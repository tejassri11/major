import os
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from env import SumoEnv

N_ENVS = 4          # parallel SUMO instances (tune down if RAM is tight)
SUMO_CONFIG = "map.sumocfg"


# Must be module-level (not a closure) so SubprocVecEnv can pickle it on Windows
def _make_env():
    return Monitor(SumoEnv(sumo_cfg=SUMO_CONFIG, use_gui=False))


def main():

    # ============================================================
    # ENVIRONMENT SETUP — N_ENVS parallel SUMO processes
    # ============================================================
    env = SubprocVecEnv([_make_env] * N_ENVS)

    # ============================================================
    # LOGGING + FOLDERS
    # ============================================================
    run_name = time.strftime("PPO_run_%Y%m%d_%H%M%S")
    log_dir = f"./tb_logs/{run_name}/"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ============================================================
    # DEVICE SELECTION
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    print(f"Parallel environments: {N_ENVS}\n")

    # ============================================================
    # CHECKPOINT CALLBACK
    # ============================================================
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=checkpoint_dir,
        name_prefix="ppo_traffic"
    )

    # ============================================================
    # EVAL CALLBACK — single env is fine for evaluation
    # ============================================================
    eval_env = DummyVecEnv([_make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # ============================================================
    # PPO MODEL  (set RESUME_FROM to a checkpoint path to continue)
    # With N_ENVS=4 the effective batch per update is n_steps * N_ENVS
    # = 1024 * 4 = 4096, so batch_size is scaled up accordingly.
    # ============================================================
    RESUME_FROM = None
    TOTAL_TIMESTEPS = 600000

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"\nResuming from checkpoint: {RESUME_FROM}\n")
        model = PPO.load(RESUME_FROM, env=env, device=device, tensorboard_log=log_dir)
        remaining = TOTAL_TIMESTEPS - model.num_timesteps
        print(f"Already trained: {model.num_timesteps} steps — {remaining} remaining\n")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,   # scaled from 64 → 256 to match larger effective batch
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )
        remaining = TOTAL_TIMESTEPS

    # ============================================================
    # TRAINING
    # ============================================================
    print(f"Training PPO Agent for {remaining} timesteps...")
    model.learn(
        total_timesteps=remaining,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="PPO_traffic_signal",
        reset_num_timesteps=False,
    )

    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================
    env.close()
    model.save("ppo_traffic_final")

    print("\nTraining completed successfully!")
    print("Final model saved as: ppo_traffic_final.zip")
    print(f"TensorBoard logs saved in: {log_dir}")


if __name__ == "__main__":
    main()
