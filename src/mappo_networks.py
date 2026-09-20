"""
Neural network definitions for MAPPO.

In reinforcement learning, the "policy" is the brain of the agent — it decides
what action to take given what the agent currently observes.

MAPPO uses two separate networks:
  1. Actor  — the policy network. Each agent uses this to pick a traffic phase.
  2. Critic — the value network. Used ONLY during training to judge how good a
              situation is. It is thrown away at deployment time.

The key innovation in MAPPO is that the Actor only sees LOCAL information
(one intersection's data) but the Critic sees GLOBAL information (all three
intersections at once). This is called CTDE:
  Centralised Training, Decentralised Execution.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    """
    Build a 3-layer fully-connected neural network (MLP = Multi-Layer Perceptron).

    Structure:  input → hidden(256) → Tanh → hidden(256) → Tanh → output

    Tanh squashes values to (-1, 1), which helps training stay stable.
    256 hidden units is a standard size — large enough to learn complex patterns,
    small enough to train quickly on CPU.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),   # layer 1: learn low-level patterns
        nn.Linear(hidden, hidden), nn.Tanh(),   # layer 2: combine patterns
        nn.Linear(hidden, out_dim),             # layer 3: produce final output (no activation)
    )


class Actor(nn.Module):
    """
    The policy network — decides which green phase to show at a traffic light.

    INPUT  : local observation for ONE traffic light (22 numbers):
               - 8 × queue length per incoming lane   (how many cars are stopped)
               - 8 × waiting time per incoming lane   (how long cars have waited)
               - 4 × vehicle count on outgoing lanes  (is there space ahead?)
               - 1 × current phase (normalised 0–1)
               - 1 × phase state flag (0=green, 1=yellow)

    OUTPUT : 3 logits — one per possible green phase (0, 1, or 2).
             Higher logit = more likely to choose that phase.

    PARAMETER SHARING: all three traffic lights use THE SAME Actor network.
    This works because the intersections are similar in structure, so the same
    "decision rules" apply to all of them. It also means 3× less parameters
    to train, which speeds up learning.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Raw scores (logits) for each action — not probabilities yet
        return self.net(obs)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Given an observation, sample (or greedily pick) an action.

        During TRAINING  : deterministic=False → sample randomly, weighted by
                           the learned probabilities. This encourages exploration.
        During EVALUATION: deterministic=True  → always pick the highest-probability
                           action. This shows the agent's best known behaviour.

        Returns:
          action   — the chosen phase index (0, 1, or 2)
          log_prob — log probability of that action (used in PPO loss calculation)
          entropy  — randomness of the distribution (high = exploring, low = confident)
        """
        logits = self.forward(obs)
        dist   = Categorical(logits=logits)     # converts logits to a probability distribution
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Re-evaluate previously taken actions under the CURRENT (updated) policy.

        This is used inside the PPO update loop. After collecting experience with
        the OLD policy, we update the network weights and then need to know:
        "under the NEW policy, what was the probability of those same actions?"

        The ratio new_prob / old_prob tells PPO how much the policy has changed,
        which it uses to clip the update and prevent destructive large steps.

        Returns log_probs and entropy for the given (obs, action) pairs.
        """
        logits = self.forward(obs)
        dist   = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


class CentralizedCritic(nn.Module):
    """
    The value network — estimates how good the current GLOBAL situation is.

    INPUT  : global state = all three traffic lights' local observations
             concatenated into one vector (66 numbers = 3 × 22).
             The critic can see EVERYTHING — all queues, all waiting times,
             all phases across all three intersections simultaneously.

    OUTPUT : a single number (the "value") estimating the total future reward
             the agents can expect from this state if they act well from here on.

    WHY CENTRALISED? During training, giving the critic the full picture lets it
    accurately judge how good or bad a joint situation is — even if one intersection
    being green causes a knock-on queue at another. This better value estimate
    produces better advantage estimates, which makes learning more stable.

    At deployment time, the critic is NOT used — only the Actor runs.
    """

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, 66) — batch of global states
        returns: (B,)  — one value estimate per state
        squeeze(-1) removes the trailing dimension of size 1 from the MLP output.
        """
        return self.net(state).squeeze(-1)
