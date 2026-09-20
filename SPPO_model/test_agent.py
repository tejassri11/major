import time
import torch
from stable_baselines3 import PPO
from env import SumoEnv

def main():
    print("Loading model: ppo_traffic_final.zip")
    model = PPO.load("ppo_traffic_final.zip")

    # Create SUMO GUI environment for testing
    env = SumoEnv(sumo_cfg="map.sumocfg", use_gui=True)

    # Gymnasium reset -> returns (obs, info)
    obs, info = env.reset()

    print("🚦 Running trained agent in SUMO GUI...")

    while True:
        # SB3 expects only the obs (array), not the tuple
        action, _ = model.predict(obs, deterministic=True)

        # Also Gymnasium step returns 5 values
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, info = env.reset()

if __name__ == "__main__":
    main()
