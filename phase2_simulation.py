import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_REPO_ROOT)

_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if "SUMO_HOME" not in os.environ:
    try:
        import sumolib
        _sumo_bin = sumolib.checkBinary("sumo")
        os.environ["SUMO_HOME"] = os.path.dirname(
            os.path.dirname(os.path.abspath(_sumo_bin))
        )
        print(f"[SUMO] Auto-detected SUMO_HOME: {os.environ['SUMO_HOME']}")
    except Exception as _e:
        print(f"[SUMO] Warning: could not auto-detect SUMO_HOME ({_e}).")

import mappo_env
from mappo_env import MAPPOEnv, TL_IDS, N_ACTIONS
import traci

mappo_env.MAX_SIM_STEPS = 10 ** 9

PHASE_HOLD_STEPS = 15


def fixed_time_actions(phase: int) -> dict:
    return {tl: phase for tl in TL_IDS}


def read_live_metrics(env_label: str) -> dict:
    try:
        traci.switch(env_label)
        n_vehicles   = traci.vehicle.getIDCount()
        n_teleports  = traci.simulation.getStartingTeleportNumber()
        n_collisions = traci.simulation.getCollidingVehiclesNumber()
        return {
            "vehicles":   n_vehicles,
            "teleports":  n_teleports,
            "collisions": n_collisions,
        }
    except Exception:
        return {"vehicles": 0, "teleports": 0, "collisions": 0}


def run_phase2(scenario: str = "rush_hour_am",
               total_steps: int = 1500,
               use_gui: bool = True):
    print()
    print("=" * 65)
    print("  PHASE 2 -- NH48 Delhi-Gurgaon Corridor: SUMO Digital Twin")
    print("=" * 65)
    print(f"  Network      : network/gurgaon_iffco.net.xml")
    print(f"  Config       : network/triple.sumocfg")
    print(f"  Intersections: TL_A | TL_B | TL_C  (3 signalized junctions)")
    print(f"  Scenario     : {scenario}")
    print(f"  Steps        : {total_steps}  ({total_steps * 3} sim-seconds)")
    print(f"  GUI          : {'SUMO-GUI' if use_gui else 'Headless'}")
    print(f"  Controller   : Fixed-time cycling (no RL)")
    print()
    print("  Vehicle types: two-wheelers (50%) | auto-rickshaws (15%)")
    print("                 hatchbacks (12%)   | motorcycles (10%)")
    print("                 sedans (5%)        | SUVs (3%) | buses (3%) | LCVs (2%)")
    print()
    print("  Sublane physics  : ENABLED (lateral-resolution = 0.8 m)")
    print("  Left-hand traffic: YES (Indian road network from OpenStreetMap)")
    print("=" * 65)
    print()

    env = MAPPOEnv(use_gui=use_gui, scenario_name=scenario, gui_delay=20)

    print(f"[Phase 2] Starting SUMO {'(GUI)' if use_gui else '(headless)'}...")
    obs_dict, _ = env.reset()
    env_label   = env._label
    print(f"[Phase 2] SUMO started successfully (TraCI label: {env_label})")
    print(f"[Phase 2] All 3 intersections loaded: {TL_IDS}")
    print()
    print(f"{'Step':>6}  {'SimTime':>8}  {'Vehicles':>9}  {'Phase':>5}  Note")
    print("-" * 55)

    step         = 0
    phase_timer  = 0
    cur_phase    = 0
    total_reward = 0.0
    t_start      = time.time()

    try:
        while step < total_steps:
            if phase_timer >= PHASE_HOLD_STEPS:
                cur_phase   = (cur_phase + 1) % N_ACTIONS
                phase_timer = 0

            actions = fixed_time_actions(cur_phase)
            obs_dict, _, reward, done, _ = env.step(actions)

            total_reward += reward
            phase_timer  += 1
            step         += 1

            if step % 50 == 0 or step == 1:
                m        = read_live_metrics(env_label)
                sim_secs = step * 3
                note     = ""
                if m["teleports"] > 0:
                    note = f"  ! {m['teleports']} teleport(s)"
                if m["collisions"] > 0:
                    note += f"  ! {m['collisions']} collision(s)"
                print(f"{step:>6}  {sim_secs:>6}s    {m['vehicles']:>6} veh   Ph-{cur_phase}{note}")


    except KeyboardInterrupt:
        print(f"\n[Phase 2] Stopped by user at step {step}.")

    except Exception as e:
        print(f"\n[Phase 2] Simulation error at step {step}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print()
        print("=" * 65)
        elapsed = time.time() - t_start
        print(f"[Phase 2] Simulation complete.")
        print(f"  Steps run   : {step} / {total_steps}")
        print(f"  Sim time    : {step * 3} seconds ({step * 3 / 60:.1f} min)")
        print(f"  Wall time   : {elapsed:.1f}s")
        print(f"  Mean reward : {total_reward / max(step, 1):.4f}")
        print("=" * 65)
        try:
            env.close()
            print("[Phase 2] SUMO closed cleanly.")
        except Exception as ce:
            print(f"[Phase 2] Close warning (non-fatal): {ce}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 NH48 SUMO Digital Twin -- Fixed-Time Signal Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  low           Quiet off-peak (Sunday morning)
  normal        Typical weekday mid-morning
  rush_hour_am  Morning rush -- heavy inbound (default)
  rush_hour_pm  Evening rush -- heavy outbound
  holiday       Sustained transit through-traffic
  incident      Normal demand with two entries blocked

Examples:
  python phase2_simulation.py
  python phase2_simulation.py --scenario holiday --steps 2000
  python phase2_simulation.py --no-gui --steps 500
        """,
    )
    parser.add_argument(
        "--scenario",
        default="rush_hour_am",
        choices=["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"],
        help="Traffic demand scenario (default: rush_hour_am)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Number of RL steps to simulate (1 step = 3 sim-seconds, default: 1500 = 75 min sim)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        dest="no_gui",
        help="Run headless without SUMO-GUI (for testing without a display)",
    )

    args = parser.parse_args()
    run_phase2(
        scenario    = args.scenario,
        total_steps = args.steps,
        use_gui     = not args.no_gui,
    )
