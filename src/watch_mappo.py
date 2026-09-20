"""
Continuous Live Playback: Watch the trained MAPPO policy control the triple intersection in SUMO-GUI.
Runs continuously in a single window without stopping, resetting, or closing.

Change SCENARIO below to any of: low, normal, rush_hour_am, rush_hour_pm, holiday, incident
"""

import time
import torch
import traci

import mappo_env
from mappo_env import MAPPOEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
from mappo_networks import Actor

SCENARIO = "rush_hour_am"   # <-- change this to try different scenarios: low, normal, rush_hour_am, etc.
MODEL    = "mappo_models/best_actor.pth"

# Infinite episode limit so the simulation runs indefinitely without ever stopping or closing the window
mappo_env.MAX_SIM_STEPS = 10**9

actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
actor.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
actor.eval()

# Start SUMO-GUI with a smooth animation delay (30 ms) so vehicles, 2-wheelers, and auto-rickshaws are observable
env = MAPPOEnv(use_gui=True, scenario_name=SCENARIO, gui_delay=30)

print(f"\n==============================================================")
print(f" [CONTINUOUS SIMULATION] Running MAPPO on scenario: {SCENARIO}")
print(f" Single persistent SUMO-GUI window — will run indefinitely.")
print(f" Close the SUMO window or press Ctrl+C to stop.")
print(f"==============================================================\n")

# Single initialization: opens the SUMO-GUI window ONCE
obs_dict, _ = env.reset()
step = 0
total_reward = 0.0

try:
    while True:
        actions = {}
        for tl in TL_IDS:
            obs_t = torch.tensor(obs_dict[tl], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_t, deterministic=True)
            actions[tl] = int(action.item())

        obs_dict, _, reward, done, _ = env.step(actions)
        total_reward += reward
        step += 1

        if step % 50 == 0:
            avg_rew = total_reward / step
            print(f"  step {step:5d} | sim time: {step * 3:5d}s | mean reward: {avg_rew:+.3f}")

        # Small pacing sleep so SUMO-GUI can render frames smoothly at ~25-30 FPS
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nSimulation stopped by user (Ctrl+C).")
except Exception as e:
    # Triggered cleanly if the user closes the SUMO window directly
    print(f"\nSimulation ended (window closed: {e}).")
finally:
    env.close()
    print("SUMO closed cleanly.")
