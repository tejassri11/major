"""
MAPPO — Multi-Agent PPO with Centralized Critic for the Palapye triple-intersection.

This is the main training script. It orchestrates everything:
  1. Creates N_ENVS=4 parallel SUMO simulations
  2. Runs N_STEPS=512 steps in each, collecting experience (observations, actions, rewards)
  3. Uses that experience to update the shared Actor and CentralizedCritic networks
  4. Repeats until TOTAL_TIMESTEPS is reached

THE BIG PICTURE — HOW RL TRAINING WORKS:
  The agent starts knowing nothing. It takes random actions and observes outcomes.
  Over time, PPO adjusts the network weights so actions that led to good rewards
  (less waiting, no gridlock) become more likely. After millions of steps, the agent
  has "learned" a policy that coordinates the three traffic lights to minimise congestion.

ARCHITECTURE RECAP (CTDE):
  - 3 agents, one per traffic light, all SHARING THE SAME Actor network (parameter sharing)
  - CentralizedCritic takes the full global state (all 3 local obs concatenated, 66-dim)
  - Cooperative reward: same shared signal for all agents each step
  - Execution uses only local observations (decentralized — Actor sees only 22 numbers)

Run:
    python train_mappo.py

Presentation summary:
  - This file is where learning happens.
  - The Actor chooses traffic-light actions from local observations.
  - The Centralized Critic evaluates the full traffic state during training.
  - PPO updates both networks using collected SUMO experience.

Outputs:
    mappo_models/   — actor + critic checkpoints  (.pth files)
    mappo_logs/     — CSV training log  (open in Excel / plot with pandas)
"""

import os
import sys
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── SUMO path guard ───────────────────────────────────────────────────────────
# Tries to auto-detect the SUMO installation if the SUMO_HOME environment variable
# is not already set. SUMO_HOME is needed by TraCI (the Python↔SUMO bridge) to
# locate its Python tools. Without it, "import traci" fails.
_SUMO_GUESSES = [
    r"C:\Program Files (x86)\Eclipse\Sumo",
    r"C:\Program Files\Eclipse\Sumo",
    r"C:\Sumo",
]

def _ensure_sumo():
    if "SUMO_HOME" not in os.environ:
        for g in _SUMO_GUESSES:
            if os.path.isdir(g):
                os.environ["SUMO_HOME"] = g
                sys.path.append(os.path.join(g, "tools"))
                break

_ensure_sumo()

from mappo_env import (MAPPOEnv, TL_IDS,
                       LOCAL_OBS_DIM, GLOBAL_STATE_DIM, N_ACTIONS)
from mappo_networks import Actor, CentralizedCritic

# ─── Hyperparameters ──────────────────────────────────────────────────────────
# These numbers control HOW the training runs. Changing them changes both speed
# and final quality. These values are well-tested defaults for traffic RL.

# N_ENVS: how many SUMO simulations run simultaneously.
# More envs = more diverse experience per update = faster convergence, but uses more RAM and CPU.
N_ENVS          = 4

# N_STEPS: how many RL steps to collect from each env before doing a parameter update.
# Total data per update = N_STEPS × N_ENVS = 256 × 4 = 1024 transitions.
# Shorter rollouts = more frequent updates = faster learning early on.
N_STEPS         = 256

# TOTAL_TIMESTEPS: training stops when this many environment steps have been taken.
# 1.5M steps at ~100 fps ≈ 15,000 seconds ≈ 4 hours.
# Traffic signal MAPPO papers show convergence in 500K–2M steps. Start here;
# load the final checkpoint and continue training if the policy hasn't converged yet.
TOTAL_TIMESTEPS = 10240

# BATCH_SIZE: within each update, the 2048 transitions are split into mini-batches of 256.
# Mini-batching makes gradient updates smoother and allows GPU parallelism.
BATCH_SIZE      = 256

# N_EPOCHS: how many passes over the collected data we do before discarding it.
# PPO re-uses data N_EPOCHS times (unlike vanilla policy gradient which discards immediately).
# More epochs = more learning per rollout, but too many epochs risk "overfitting" to stale data.
N_EPOCHS        = 10

# GAMMA: discount factor for future rewards.
# 0.99 means a reward 100 steps in the future is worth 0.99^100 ≈ 0.37 of an immediate reward.
# High gamma = agent thinks long-term. Traffic signals need long-term thinking (queues build up slowly).
GAMMA           = 0.99

# GAE_LAMBDA: controls the bias-variance tradeoff in advantage estimation.
# λ=1.0 = unbiased but high variance (Monte Carlo return).
# λ=0.0 = low variance but biased (just the TD error, i.e., one-step advantage).
# λ=0.95 is a standard sweet spot that's nearly unbiased with much lower variance.
GAE_LAMBDA      = 0.95

# CLIP_EPS: the PPO clipping range. The ratio of new/old policy probabilities is clipped
# to [1-0.2, 1+0.2] = [0.8, 1.2]. This prevents the network from making huge jumps.
# Think of it as a safety rail: we can learn from the data but not too aggressively.
CLIP_EPS        = 0.2

# Learning rates: how large each gradient step is.
# Actor uses a smaller lr (3e-4) than Critic (1e-3) because policy changes must be careful —
# a bad policy change is harder to recover from than a bad value estimate.
LR_ACTOR        = 3e-4
LR_CRITIC       = 1e-3

# ENT_COEF: entropy bonus coefficient. A small bonus for HIGH entropy (= randomness) in
# the policy encourages the agent to keep exploring rather than getting stuck early.
# Without this, the agent can converge to a mediocre policy and stop exploring.
ENT_COEF        = 0.01

# VF_COEF: how much weight to give the critic loss relative to the actor loss in the
# combined gradient. 0.5 = critic matters half as much in the combined update.
VF_COEF         = 0.5

# MAX_GRAD_NORM: gradient clipping threshold. If the gradient vector's magnitude exceeds 0.5,
# we scale it down. Prevents explosive gradient updates, which can destabilise training.
MAX_GRAD_NORM   = 0.5

N_AGENTS  = len(TL_IDS)   # = 3
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "mappo_models"
LOG_DIR   = "mappo_logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)


# ─── Rollout Buffer ───────────────────────────────────────────────────────────
class RolloutBuffer:
    """
    Stores a complete rollout of experience before each PPO update.

    A "rollout" is a block of N_STEPS consecutive (obs, action, reward, ...) tuples
    collected from N_ENVS environments. We need to store all of this data in memory
    BEFORE doing any gradient updates, because:
      1. PPO needs to know the OLD policy's probabilities (log_probs at collection time)
         so it can compute the probability ratio new/old during the update.
      2. GAE (advantage estimation) requires going BACKWARDS through the rollout —
         you can't compute it step-by-step while collecting.

    DATA SHAPES:
      (T, E, A, obs_dim) for per-agent local observations
      (T, E, state_dim)  for global state (passed to critic)
      (T, E, A)          for actions and log_probs (one per agent per step)
      (T, E)             for rewards, values, dones (scalar per (step, env))

    where T=N_STEPS=512, E=N_ENVS=4, A=N_AGENTS=3
    """

    def __init__(self):
        T, E, A = N_STEPS, N_ENVS, N_AGENTS
        # per-agent local observations — what each Agent saw when it took its action
        self.obs          = np.zeros((T, E, A, LOCAL_OBS_DIM),  dtype=np.float32)
        # global state — what the Critic saw (all 3 agents' obs concatenated)
        self.global_state = np.zeros((T, E, GLOBAL_STATE_DIM),  dtype=np.float32)
        # actions taken by each agent at each step
        self.actions      = np.zeros((T, E, A),                 dtype=np.int64)
        # log-probabilities of those actions under the policy that collected them (OLD policy)
        self.log_probs    = np.zeros((T, E, A),                 dtype=np.float32)
        # shared cooperative reward (one value for all agents at each step)
        self.rewards      = np.zeros((T, E),                    dtype=np.float32)
        # critic's value estimate at the time of collection (used in GAE)
        self.values       = np.zeros((T, E),                    dtype=np.float32)
        # done flags: 1.0 if the episode ended at this step, 0.0 otherwise
        # used in GAE to zero out future rewards across episode boundaries
        self.dones        = np.zeros((T, E),                    dtype=np.float32)

    def add(self, t, e, obs_dict, global_state,
            agent_actions, agent_log_probs, reward, value, done):
        """
        Store one transition (t=time step, e=env index) into the buffer.
        Called once per step per environment during rollout collection.
        """
        for a, tl in enumerate(TL_IDS):
            self.obs[t, e, a]       = obs_dict[tl]
            self.actions[t, e, a]   = agent_actions[a]
            self.log_probs[t, e, a] = agent_log_probs[a]
        self.global_state[t, e] = global_state
        self.rewards[t, e]      = reward
        self.values[t, e]       = value
        self.dones[t, e]        = float(done)

    def compute_gae(self, last_values: np.ndarray):
        """
        Compute Generalised Advantage Estimation (GAE) over the stored rollout.

        WHAT IS AN "ADVANTAGE"?
          The advantage A(s, a) answers: "how much better was taking action a from state s,
          compared to what we'd TYPICALLY expect from state s?"
          A > 0: this action was better than average — increase its probability
          A < 0: this action was worse than average — decrease its probability

        WHY NOT JUST USE THE RAW REWARD?
          If we updated toward every positive reward, we'd have enormous variance because
          rewards fluctuate randomly. The advantage SUBTRACTS the baseline (the value estimate),
          which is the critic's prediction of how good the state already is. This baseline
          dramatically reduces variance without introducing bias.

        WHY GAE INSTEAD OF SIMPLE TD ERROR?
          The TD error (δ = r + γV(s') - V(s)) is a single-step estimate. It's low variance
          but "short-sighted". The full Monte Carlo return (sum all future rewards) is unbiased
          but very noisy. GAE blends the two with parameter λ:
            A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
          λ=0.95 uses mostly TD error but adds a "look-ahead tail" for better estimates.

        BACKWARD PASS:
          We compute advantages from t=T-1 down to t=0 because each step's advantage
          depends on the NEXT step's GAE value (recursion going forward = recurrence).

        Args:
          last_values: (N_ENVS,) — the critic's estimate of V(s) AFTER the last step.
                       We need this because the rollout ends mid-episode; we can't pretend
                       future rewards are zero (that would under-value the last states).

        Returns:
          advantages: (N_STEPS, N_ENVS) — how much better each step's actions were
          returns:    (N_STEPS, N_ENVS) — target values for the critic to fit
                      (returns = advantages + baseline = "what the value should have been")
        """
        advantages = np.zeros_like(self.rewards)
        last_gae   = np.zeros(N_ENVS, dtype=np.float32)   # tracks the running GAE tail

        for t in reversed(range(N_STEPS)):
            # Bootstrap next-state value:
            # At the final step of the rollout, use the post-rollout critic estimate.
            # At all other steps, use the value we already stored for step t+1.
            next_values  = last_values if t == N_STEPS - 1 else self.values[t + 1]

            # non_terminal = 0 at episode boundaries, 1 otherwise.
            # Multiplying by non_terminal zeros out the future-value term when the
            # episode ended — the "future" after a done=True is the next episode, not this one.
            non_terminal = 1.0 - self.dones[t]

            # TD error δ_t = r_t + γ * V(s_{t+1}) * non_terminal - V(s_t)
            # This is the "surprise" signal: did we get more or less than the baseline predicted?
            delta = (self.rewards[t]
                     + GAMMA * next_values * non_terminal
                     - self.values[t])

            # GAE recurrence: A_t = δ_t + γλ * non_terminal * A_{t+1}
            # The non_terminal mask cuts off the recurrence at episode boundaries.
            last_gae      = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae
            advantages[t] = last_gae

        # returns = advantages + values = "what the critic SHOULD have predicted"
        # Used as the target in the critic's mean-squared-error loss.
        returns = advantages + self.values
        return advantages, returns

    def iterate_batches(self, advantages: np.ndarray, returns: np.ndarray):
        """
        Flatten the entire rollout into one big batch, then yield randomised mini-batches.

        WHY SHUFFLE?
          Consecutive transitions are highly correlated (step t+1 is very similar to step t).
          Training on correlated batches leads to poor gradient estimates. Random shuffling
          breaks these correlations, making each mini-batch a representative sample of the
          full rollout diversity.

        WHAT GETS FLATTENED:
          (N_STEPS, N_ENVS, ...) → (N_STEPS × N_ENVS, ...) = (2048, ...)
          Then split into batches of BATCH_SIZE=256.

        ADVANTAGE NORMALISATION:
          Subtracting the mean and dividing by std keeps advantages in a consistent scale
          across different episodes and rollouts. Without this, the policy gradient
          magnitude varies wildly depending on how good or bad the current episode was,
          making the learning rate effectively inconsistent.
        """
        n_total = N_STEPS * N_ENVS   # 2048

        # Flatten time and env dimensions together
        obs_flat   = self.obs.reshape(n_total, N_AGENTS, LOCAL_OBS_DIM)
        state_flat = self.global_state.reshape(n_total, GLOBAL_STATE_DIM)
        act_flat   = self.actions.reshape(n_total, N_AGENTS)
        lp_flat    = self.log_probs.reshape(n_total, N_AGENTS)
        adv_flat   = advantages.reshape(n_total)
        ret_flat   = returns.reshape(n_total)

        # Normalise: zero mean, unit std across the entire batch
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Random permutation so each mini-batch is drawn from different envs and time steps
        idx = np.random.permutation(n_total)
        for start in range(0, n_total, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            yield (
                torch.tensor(obs_flat[b],   device=DEVICE),   # (B, A, obs_dim)
                torch.tensor(state_flat[b], device=DEVICE),   # (B, state_dim)
                torch.tensor(act_flat[b],   device=DEVICE),   # (B, A)
                torch.tensor(lp_flat[b],    device=DEVICE),   # (B, A)
                torch.tensor(adv_flat[b],   device=DEVICE),   # (B,)
                torch.tensor(ret_flat[b],   device=DEVICE),   # (B,)
            )


# ─── PPO Update ───────────────────────────────────────────────────────────────
def ppo_update(actor, critic, actor_opt, critic_opt, buffer: RolloutBuffer,
               last_values: np.ndarray):
    """
    Perform N_EPOCHS passes over the rollout buffer, updating Actor and Critic.

    This is the mathematical heart of PPO. Here's what happens:

    1. compute_gae() calculates how good or bad each past action was (advantages).
    2. For each mini-batch, we re-evaluate those actions under the CURRENT (updated) policy.
    3. We compute how much the policy has changed: ratio = new_prob / old_prob.
    4. PPO clips the ratio to [0.8, 1.2] so the update can't be too large.
    5. We combine actor loss + critic loss + entropy bonus and backpropagate.

    ACTOR LOSS (CLIPPED SURROGATE OBJECTIVE):
      Normal policy gradient: loss = -log_prob × advantage
        Problem: if advantage is large, gradient is huge → catastrophic update.
      PPO solution: multiply by ratio (new/old prob), then CLIP the ratio:
        surr1 = ratio × advantage
        surr2 = clip(ratio, 0.8, 1.2) × advantage
        loss  = -min(surr1, surr2)   ← take the CONSERVATIVE option
      This ensures: if the new policy differs too much from the old one, the update
      is automatically reduced. The agent can only move "one small step" at a time.

    CRITIC LOSS (MEAN SQUARED ERROR):
      The critic predicts the expected future return from a state.
      We train it to match the actual returns computed by GAE.
      Accurate value estimates = better advantages = better policy updates.

    ENTROPY BONUS:
      We SUBTRACT a small entropy term from the loss (because loss = -reward).
      Higher entropy = more spread-out probability distribution = more exploration.
      This prevents the policy from becoming overconfident too early.

    SHARED ADVANTAGE FOR ALL AGENTS:
      Because all agents share a cooperative reward, we use the SAME advantage for
      all three traffic lights at each step. We "broadcast" the scalar advantage
      to match the (B, A) shape of the log-probs.

    Returns mean actor_loss, critic_loss, entropy across all mini-batches.
    """
    advantages, returns = buffer.compute_gae(last_values)

    actor_losses, critic_losses, entropies = [], [], []

    for _ in range(N_EPOCHS):
        for obs_b, state_b, act_b, old_lp_b, adv_b, ret_b in \
                buffer.iterate_batches(advantages, returns):

            B = obs_b.shape[0]   # mini-batch size (typically BATCH_SIZE=256)

            # ── Actor loss (all agents, shared network) ────────────────────────
            # The Actor processes ONE agent's observation at a time. But we have
            # B transitions × A agents = B×A observations to process per mini-batch.
            # Instead of a loop, we RESHAPE to (B×A, obs_dim) and do one big forward pass.
            # Then reshape back to (B, A) to compute per-agent log-probs.
            obs_flat    = obs_b.reshape(B * N_AGENTS, LOCAL_OBS_DIM)
            act_flat    = act_b.reshape(B * N_AGENTS)
            old_lp_flat = old_lp_b.reshape(B * N_AGENTS)

            # evaluate_actions() re-runs the CURRENT policy on old (obs, action) pairs
            # returns log-probs and entropy under the CURRENT (not old) policy weights
            new_lp_flat, entropy_flat = actor.evaluate_actions(obs_flat, act_flat)

            # Reshape back to (B, A) — one value per agent per transition
            new_lp  = new_lp_flat.reshape(B, N_AGENTS)
            old_lp  = old_lp_flat.reshape(B, N_AGENTS)
            entropy = entropy_flat.reshape(B, N_AGENTS).mean()

            # MAPPO: broadcast the scalar advantage to all agents.
            # adv_b is shape (B,) — one advantage per transition (shared by all agents).
            # unsqueeze(1) → (B, 1); expand_as → (B, A) — same value for each agent.
            adv_exp = adv_b.unsqueeze(1).expand_as(new_lp)

            # Probability ratio: how much more (or less) likely is this action now vs before?
            # log(new/old) = log(new) - log(old), then exp to get the actual ratio.
            ratio = torch.exp(new_lp - old_lp)

            # Clipped surrogate loss: the PPO update
            surr1  = ratio * adv_exp
            surr2  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_exp
            a_loss = -torch.min(surr1, surr2).mean()   # negative because we MAXIMISE reward

            # ── Critic loss (centralised value function) ───────────────────────
            # state_b is shape (B, 66) — the full global state at each transition.
            # critic() predicts the expected return from that state.
            # We train it with MSE against the actual GAE returns.
            values_pred = critic(state_b)
            c_loss      = nn.functional.mse_loss(values_pred, ret_b.float())

            # ── Combined loss ──────────────────────────────────────────────────
            # The three components are added into a single scalar for backward().
            #   a_loss:           policy gradient (want to increase, so we minimise -it → already negative)
            #   VF_COEF * c_loss: value function accuracy (want to decrease MSE)
            #   ENT_COEF * entropy: exploration bonus (want to INCREASE entropy → subtract from loss)
            loss = a_loss + VF_COEF * c_loss - ENT_COEF * entropy

            # Zero gradients, backpropagate, clip, step — standard PyTorch update loop
            actor_opt.zero_grad()
            critic_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(),  MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            actor_opt.step()
            critic_opt.step()

            actor_losses.append(a_loss.item())
            critic_losses.append(c_loss.item())
            entropies.append(entropy.item())

    return (float(np.mean(actor_losses)),
            float(np.mean(critic_losses)),
            float(np.mean(entropies)))


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(actor, n_episodes=3, scenario="normal"):
    """
    Run n_episodes DETERMINISTIC episodes and return the mean episode reward.

    Deterministic = Actor always picks the HIGHEST probability action (no sampling).
    This gives a "clean" measure of what the agent has actually learned, without the
    noise that comes from exploration during training.

    We evaluate on a SEPARATE environment (not one of the training envs) to avoid
    contaminating training state. The eval environment is closed after evaluation.

    Evaluation runs every 10 updates. If the eval reward improves, we save the model
    as "best_actor.pth" — the best checkpoint seen so far in training.
    """
    env        = MAPPOEnv(use_gui=False, scenario_name=scenario)
    ep_rewards = []

    for _ in range(n_episodes):
        obs_dict, global_state = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            actions = {}
            with torch.no_grad():   # no gradients needed at evaluation time
                for tl in TL_IDS:
                    obs_t = torch.tensor(obs_dict[tl], device=DEVICE).unsqueeze(0)
                    action, _, _ = actor.get_action(obs_t, deterministic=True)   # greedy
                    actions[tl]  = action.item()
            obs_dict, global_state, reward, done, _ = env.step(actions)
            ep_r += reward
        ep_rewards.append(ep_r)

    env.close()
    return float(np.mean(ep_rewards))


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Presentation flow:
    # 1. Create Actor/Critic networks.
    # 2. Create several SUMO environments.
    # 3. Collect traffic experience.
    # 4. Update the neural networks with PPO.
    # 5. Save logs and checkpoints for evaluation.
    print("=" * 62)
    print("  MAPPO Traffic Signal Training  —  Palapye Triple Intersection")
    print(f"  Device          : {DEVICE}")
    print(f"  Parallel envs   : {N_ENVS}")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Rollout steps   : {N_STEPS} per env  ({N_STEPS * N_ENVS} total per update)")
    print(f"  Agents          : {N_AGENTS}  (parameter sharing)")
    print(f"  Local obs dim   : {LOCAL_OBS_DIM}   Global state dim: {GLOBAL_STATE_DIM}")
    print("=" * 62)

    # ── Networks & optimisers ─────────────────────────────────────────────────
    # One Actor shared across all 3 agents.
    # One CentralizedCritic that sees the full global state.
    # Separate optimisers allow different learning rates for actor vs critic.
    actor  = Actor(LOCAL_OBS_DIM, N_ACTIONS).to(DEVICE)
    critic = CentralizedCritic(GLOBAL_STATE_DIM).to(DEVICE)
    actor_opt  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # ── Environments ─────────────────────────────────────────────────────────
    # 4 independent SUMO processes. Each advances separately each step.
    # Having multiple environments gives the buffer more DIVERSE experiences —
    # different scenarios, different random traffic, different starting conditions.
    # This reduces correlation in the training data and improves stability.
    envs = [MAPPOEnv(use_gui=False) for _ in range(N_ENVS)]

    # ── Logging ──────────────────────────────────────────────────────────────
    # CSV file: every update appends one row with key metrics.
    # After training, load with pandas and plot to analyse learning progress.
    log_path = os.path.join(LOG_DIR, "mappo_train.csv")
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["timestep", "update", "mean_reward",
                     "actor_loss", "critic_loss", "entropy", "eval_reward"])

    best_eval = -float("inf")   # tracks the best evaluation reward seen so far

    # ── Initial reset ─────────────────────────────────────────────────────────
    # Start all 4 environments before entering the training loop.
    cur_obs   = [None] * N_ENVS   # current observation dict per env
    cur_state = [None] * N_ENVS   # current global state per env
    for e, env in enumerate(envs):
        cur_obs[e], cur_state[e] = env.reset()

    buffer            = RolloutBuffer()
    total_steps       = 0
    update_count      = 0
    ep_rewards        = [0.0] * N_ENVS   # running reward accumulator per env
    completed_rewards = []                # rewards of FINISHED episodes (for logging)
    start_time        = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    # The outermost loop runs until TOTAL_TIMESTEPS is reached.
    # Each iteration: collect a rollout → update networks → log → repeat.
    while total_steps < TOTAL_TIMESTEPS:

        # ── Collect rollout ───────────────────────────────────────────────────
        # For each of the N_STEPS time steps, step through ALL N_ENVS environments.
        # This fills the RolloutBuffer with 2048 transitions.
        for t in range(N_STEPS):
            for e, env in enumerate(envs):
                agent_actions   = []
                agent_log_probs = []

                with torch.no_grad():   # inference only — no gradient tracking needed here
                    # Get each agent's action from the shared Actor
                    for tl in TL_IDS:
                        obs_t = torch.tensor(cur_obs[e][tl], device=DEVICE).unsqueeze(0)
                        # deterministic=False → sample from the distribution (exploration)
                        action, log_prob, _ = actor.get_action(obs_t)
                        agent_actions.append(action.item())
                        agent_log_probs.append(log_prob.item())

                    # Critic evaluates the global state → scalar value estimate
                    state_t = torch.tensor(cur_state[e], device=DEVICE).unsqueeze(0)
                    value   = critic(state_t).item()

                # Execute the chosen actions in the simulation
                actions_dict = {tl: agent_actions[a] for a, tl in enumerate(TL_IDS)}
                new_obs, new_state, reward, done, _ = env.step(actions_dict)

                # Store this transition in the buffer
                buffer.add(t, e,
                           cur_obs[e], cur_state[e],
                           agent_actions, agent_log_probs,
                           reward, value, done)

                ep_rewards[e] += reward
                total_steps   += 1

                # Episode finished: record total reward and reset the environment
                if done:
                    completed_rewards.append(ep_rewards[e])
                    ep_rewards[e] = 0.0
                    new_obs, new_state = env.reset()

                # Advance the "current obs" pointers
                cur_obs[e]   = new_obs
                cur_state[e] = new_state

        # ── Bootstrap value for last obs ──────────────────────────────────────
        # After the rollout ends, we don't know the "true" future returns yet —
        # the episode isn't over. We ask the critic to estimate V(s) for the LAST
        # observed state. GAE uses this as a stand-in for the infinite future sum.
        with torch.no_grad():
            last_values = np.array([
                critic(torch.tensor(cur_state[e], device=DEVICE).unsqueeze(0)).item()
                for e in range(N_ENVS)
            ], dtype=np.float32)

        # ── PPO update ────────────────────────────────────────────────────────
        # Use the collected rollout to update the Actor and Critic.
        # This is where the neural network weights actually change.
        a_loss, c_loss, ent = ppo_update(
            actor, critic, actor_opt, critic_opt, buffer, last_values
        )
        update_count += 1

        # ── Logging ───────────────────────────────────────────────────────────
        # mean_rew: average episode reward over the last 20 completed episodes.
        # As training progresses, this should slowly rise toward 0 (= no congestion).
        mean_rew = float(np.mean(completed_rewards[-20:])) if completed_rewards else 0.0
        elapsed  = time.time() - start_time
        fps      = total_steps / max(elapsed, 1)
        eta_s    = int((TOTAL_TIMESTEPS - total_steps) / max(fps, 1))
        eta_h, eta_r = divmod(eta_s, 3600)
        eta_m, eta_s = divmod(eta_r, 60)

        # ASCII progress bar: fills from left as training advances
        pct     = total_steps / TOTAL_TIMESTEPS
        bar_len = 40
        filled  = int(bar_len * pct)
        bar     = "#" * filled + "." * (bar_len - filled)

        print(f"\n{'-'*62}")
        print(f"  Progress  [{bar}] {pct*100:5.1f}%")
        print(f"  Steps     {total_steps:>10,} / {TOTAL_TIMESTEPS:,}   ETA {eta_h:02d}h {eta_m:02d}m {eta_s:02d}s")
        print(f"{'-'*62}")
        print(f"  {'Metric':<18} {'Value':>10}")
        print(f"  {'------':<18} {'-----':>10}")
        print(f"  {'Update':<18} {update_count:>10,}")
        print(f"  {'Mean Reward':<18} {mean_rew:>10.3f}")
        print(f"  {'Actor Loss':<18} {a_loss:>10.4f}")
        print(f"  {'Critic Loss':<18} {c_loss:>10.4f}")
        print(f"  {'Entropy':<18} {ent:>10.3f}")
        print(f"  {'FPS':<18} {fps:>10.0f}")
        print(f"{'-'*62}")

        # ── Periodic evaluation ───────────────────────────────────────────────
        # Every 10 updates, run 3 deterministic episodes to measure "real" performance.
        # Save the model if this is the best eval reward seen so far.
        eval_rew = 0.0
        if update_count % 10 == 0:
            eval_rew = evaluate(actor, n_episodes=3, scenario="normal")
            print(f"  {'Eval Reward':<18} {eval_rew:>10.3f}  <-- deterministic")

            if eval_rew > best_eval:
                best_eval = eval_rew
                torch.save(actor.state_dict(),
                           os.path.join(MODEL_DIR, "best_actor.pth"))
                torch.save(critic.state_dict(),
                           os.path.join(MODEL_DIR, "best_critic.pth"))
                print(f"  * New best model saved  ({best_eval:.3f})")
            print(f"{'-'*62}")

        # Append this update's metrics to the CSV log
        writer.writerow([total_steps, update_count, mean_rew,
                         a_loss, c_loss, ent, eval_rew])
        log_file.flush()   # flush immediately so we can inspect the file mid-training

        # ── Periodic checkpoint ───────────────────────────────────────────────
        # Every 50 updates, save a snapshot labelled by timestep.
        # Allows rolling back to an earlier checkpoint if training diverges later.
        if update_count % 50 == 0:
            ckpt = os.path.join(MODEL_DIR, f"actor_{total_steps}.pth")
            torch.save(actor.state_dict(), ckpt)
            torch.save(critic.state_dict(),
                       ckpt.replace("actor_", "critic_"))

    # ── Final save ────────────────────────────────────────────────────────────
    torch.save(actor.state_dict(),  os.path.join(MODEL_DIR, "final_actor.pth"))
    torch.save(critic.state_dict(), os.path.join(MODEL_DIR, "final_critic.pth"))

    log_file.close()
    for env in envs:
        env.close()

    elapsed = time.time() - start_time
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)

    print("\n" + "=" * 62)
    print(f"  Training complete!  {h:02d}h {m:02d}m {s:02d}s")
    print(f"  Best eval reward  : {best_eval:.3f}")
    print(f"  Models saved to   : {MODEL_DIR}/")
    print(f"  Log saved to      : {log_path}")
    print("=" * 62)
