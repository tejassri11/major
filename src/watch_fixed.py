"""
Watch the Fixed-Time controller run the Palapye triple intersection in SUMO-GUI.
Phases cycle on a fixed timer (no RL) — the same baseline used in the compare scripts.
Change SCENARIO below to any of: low, normal, rush_hour_am, rush_hour_pm, holiday, incident
"""

from mappo_env import MAPPOEnv, TL_IDS, N_ACTIONS

SCENARIO          = "rush_hour_am"   # <-- change this to try different scenarios
FIXED_PHASE_STEPS = 15               # RL steps per phase before advancing (= 45 sim-seconds)

env = MAPPOEnv(use_gui=True, scenario_name=SCENARIO)
obs_dict, _ = env.reset()

print(f"\nRunning Fixed-Time on scenario: {SCENARIO}")
print("Watch the SUMO window. Close it or press Ctrl+C to stop.\n")

step        = 0
done        = False
phase_timer = 0
fixed_phase = 0

while not done:
    if phase_timer >= FIXED_PHASE_STEPS:
        fixed_phase = (fixed_phase + 1) % N_ACTIONS
        phase_timer = 0

    actions = {tl: fixed_phase for tl in TL_IDS}
    obs_dict, _, _, done, _ = env.step(actions)

    phase_timer += 1
    step        += 1

    if step % 50 == 0:
        print(f"  step {step:4d} | phase {fixed_phase}")

print(f"\nEpisode finished — {step} steps")
