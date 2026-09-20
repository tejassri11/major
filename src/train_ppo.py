"""
Train a PPO agent to control traffic signals in the SUMO network.

Run:
    python train_ppo.py

Outputs:
    ppo_models/   — checkpoint files saved every 10 000 steps
    ppo_logs/     — TensorBoard logs  (tensorboard --logdir ppo_logs)

Evaluation:
    A separate env fixed to the "normal" scenario evaluates the agent
    every 20 000 steps and saves the best model as ppo_models/best_model.
"""

import os
import sys
import time

# ── SUMO path guard ───────────────────────────────────────────────────────────
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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from sumo_env import SumoEnv

MODEL_DIR = "ppo_models"
LOG_DIR   = "ppo_logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

N_ENVS          = 2
TOTAL_TIMESTEPS = 5_000_000


def make_env(rank: int, seed: int = 0):
    def _init():
        _ensure_sumo()  # each subprocess needs this on Windows (spawn)
        env = Monitor(SumoEnv(use_gui=False, scenario_name=None))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Traffic Signal Training")
    print(f"  Parallel envs   : {N_ENVS}")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Models saved to : {MODEL_DIR}/")
    print(f"  Logs saved to   : {LOG_DIR}/")
    print("=" * 60)

    _start = time.time()

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    eval_env  = Monitor(SumoEnv(use_gui=False, scenario_name="normal"))

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 1024,       # per env; total rollout = 1024 × 2 = 2048
        batch_size      = 256,        # larger batch suits more data
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.005,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = MODEL_DIR,
        name_prefix = "ppo_traffic",
        verbose     = 1,
    )

    eval_cb = EvalCallback(
        eval_env             = eval_env,
        best_model_save_path = MODEL_DIR,
        log_path             = LOG_DIR,
        eval_freq            = 20_000,
        n_eval_episodes      = 3,
        deterministic        = True,
        verbose              = 1,
    )

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_cb, eval_cb],
    )

    final_path = os.path.join(MODEL_DIR, "ppo_traffic_final")
    model.save(final_path)

    _elapsed = time.time() - _start
    _h, _rem  = divmod(int(_elapsed), 3600)
    _m, _s    = divmod(_rem, 60)
    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Total time  : {_h:02d}h {_m:02d}m {_s:02d}s")
    print(f"  Final model : {final_path}.zip")
    print("=" * 60)

    train_env.close()
    eval_env.close()
