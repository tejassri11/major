"""
mappo_vs_fixed_network_live.py

Time-series comparison of MAPPO vs Fixed-Time on the full triple-intersection
A1 corridor network. Records per-step metrics averaged across all three junctions,
chains all six demand scenarios back-to-back, and saves one line chart per metric
in the same style as compare_tla_live.py.

Metrics: waiting time, queue length, throughput, stop ratio, pressure

Output:
  output/compare_network_waiting_time.png
  output/compare_network_queue_length.png
  output/compare_network_throughput.png
  output/compare_network_stop_ratio.png
  output/compare_network_pressure.png
"""

# --- Performance Analysis/ lives outside src/; add ../src to the path so the
# flat project imports (mappo_env, mappo_networks, sumo_env) resolve. Run these
# scripts from the repository root so data paths (network/, mappo_models/) work.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))


import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_SUMO_GUESSES = [
    r"C:\Program Files (x86)\Eclipse\Sumo",
    r"C:\Program Files\Eclipse\Sumo",
    r"C:\Sumo",
]
if "SUMO_HOME" not in os.environ:
    for g in _SUMO_GUESSES:
        if os.path.isdir(g):
            os.environ["SUMO_HOME"] = g
            sys.path.append(os.path.join(g, "tools"))
            break

import traci
from mappo_env import MAPPOEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
from mappo_networks import Actor

os.makedirs("output", exist_ok=True)

# ── config ────────────────────────────────────────────────────
MAPPO_MODEL_PATH  = "mappo_models/best_actor.pth"
SCENARIOS         = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]
FIXED_PHASE_STEPS = 10
MAPPO_DELTA_T     = 3       # sim seconds per RL step
TARGET_SIM_SECS   = 1800    # total sim seconds per controller

METRIC_INFO = [
    # (key,         suffix,          title,                    ylabel,                         lower_better)
    ("wait",        "waiting_time",  "Network Waiting Time",   "Waiting Time (s/lane)",        True),
    ("queue",       "queue_length",  "Network Queue Length",   "Halting Vehicles / Lane",      True),
    ("throughput",  "throughput",    "Network Throughput",     "Vehicles Cleared",             False),
    ("stop_ratio",  "stop_ratio",    "Network Stop Ratio",     "Fraction of Vehicles Stopped", True),
    ("pressure",    "pressure",      "Network Pressure",       "Incoming - Outgoing Vehicles", True),
]

COLORS = {"MAPPO": "steelblue", "Fixed": "tomato"}

# ── metric collection ─────────────────────────────────────────

_seen_passed: set = set()

def _reset_throughput():
    _seen_passed.clear()

def _read_network(env, label):
    """Read per-step metrics averaged across all three junctions."""
    traci.switch(label)

    wait_vals, queue_vals, total_vals, halting_vals = [], [], [], []
    out_veh_total = 0
    passed_step   = 0

    for tl in TL_IDS:
        in_lanes  = env._controlled_lanes.get(tl, [])
        out_lanes = env._outgoing_lanes.get(tl, [])

        for ln in in_lanes:
            try:
                wait_vals.append(traci.lane.getWaitingTime(ln))
                h = traci.lane.getLastStepHaltingNumber(ln)
                v = traci.lane.getLastStepVehicleNumber(ln)
                halting_vals.append(h)
                total_vals.append(v)
            except Exception:
                wait_vals.append(0.0)
                halting_vals.append(0)
                total_vals.append(0)

        for ln in out_lanes:
            try:
                out_veh_total += traci.lane.getLastStepVehicleNumber(ln)
                for vid in traci.lane.getLastStepVehicleIDs(ln):
                    if vid not in _seen_passed:
                        _seen_passed.add(vid)
                        passed_step += 1
            except Exception:
                pass

    n = max(len(env._controlled_lanes.get(TL_IDS[0], [])) * len(TL_IDS), 1)
    total_veh = sum(total_vals)
    total_halt = sum(halting_vals)

    return {
        "wait":       sum(wait_vals) / max(len(TL_IDS), 1),
        "queue":      total_halt / max(len(TL_IDS), 1),
        "throughput": float(passed_step),
        "stop_ratio": total_halt / max(total_veh, 1),
        "pressure":   float(total_veh - out_veh_total),
    }


# ── controller runners ────────────────────────────────────────

def run_mappo(actor):
    print("\n" + "=" * 60)
    print("MAPPO — triple network, scenario rotation")
    print("=" * 60)

    _reset_throughput()
    times, metrics = [], {k: [] for k in ("wait", "queue", "throughput", "stop_ratio", "pressure")}
    scen_marks = []
    cum_tp   = 0.0
    sim_secs = 0
    rl_step  = 0
    scen_idx = 0

    while sim_secs < TARGET_SIM_SECS:
        scenario = SCENARIOS[scen_idx % len(SCENARIOS)]
        scen_idx += 1
        scen_marks.append((sim_secs, scenario))
        print(f"  [MAPPO] scenario '{scenario}' at t={sim_secs}s")

        env = MAPPOEnv(use_gui=False, scenario_name=scenario)
        obs_dict, _ = env.reset()
        done = False

        while not done and sim_secs < TARGET_SIM_SECS:
            m = _read_network(env, env._label)
            cum_tp += m["throughput"]
            times.append(float(sim_secs))
            for k in metrics:
                metrics[k].append(cum_tp if k == "throughput" else m[k])

            actions = {}
            for tl in TL_IDS:
                obs_t = torch.tensor(obs_dict[tl], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = actor.get_action(obs_t, deterministic=True)
                actions[tl] = int(action.item())

            obs_dict, _, _, done, _ = env.step(actions)
            rl_step  += 1
            sim_secs += MAPPO_DELTA_T

            if rl_step % 100 == 0:
                print(f"  [MAPPO] t={sim_secs}s  wait={metrics['wait'][-1]:.2f}  queue={metrics['queue'][-1]:.2f}")

        env.close()
        time.sleep(1)

    print(f"  [MAPPO] done — {rl_step} steps / {sim_secs}s / {scen_idx} scenarios")
    return np.array(times), {k: np.array(v) for k, v in metrics.items()}, scen_marks


def run_fixed():
    print("\n" + "=" * 60)
    print("Fixed-Time — triple network, scenario rotation")
    print("=" * 60)

    _reset_throughput()
    times, metrics = [], {k: [] for k in ("wait", "queue", "throughput", "stop_ratio", "pressure")}
    scen_marks = []
    cum_tp     = 0.0
    sim_secs   = 0
    rl_step    = 0
    scen_idx   = 0
    fixed_phase  = 0
    phase_timer  = 0

    while sim_secs < TARGET_SIM_SECS:
        scenario = SCENARIOS[scen_idx % len(SCENARIOS)]
        scen_idx += 1
        scen_marks.append((sim_secs, scenario))
        print(f"  [Fixed] scenario '{scenario}' at t={sim_secs}s")

        env = MAPPOEnv(use_gui=False, scenario_name=scenario)
        obs_dict, _ = env.reset()
        done = False

        while not done and sim_secs < TARGET_SIM_SECS:
            m = _read_network(env, env._label)
            cum_tp += m["throughput"]
            times.append(float(sim_secs))
            for k in metrics:
                metrics[k].append(cum_tp if k == "throughput" else m[k])

            if phase_timer >= FIXED_PHASE_STEPS:
                fixed_phase = (fixed_phase + 1) % N_ACTIONS
                phase_timer = 0
            actions     = {tl: fixed_phase for tl in TL_IDS}
            phase_timer += 1

            obs_dict, _, _, done, _ = env.step(actions)
            rl_step  += 1
            sim_secs += MAPPO_DELTA_T

            if rl_step % 100 == 0:
                print(f"  [Fixed] t={sim_secs}s  wait={metrics['wait'][-1]:.2f}  queue={metrics['queue'][-1]:.2f}")

        env.close()
        time.sleep(1)

    print(f"  [Fixed] done — {rl_step} steps / {sim_secs}s / {scen_idx} scenarios")
    return np.array(times), {k: np.array(v) for k, v in metrics.items()}, scen_marks


# ── plotting ──────────────────────────────────────────────────

def smooth(y, w=20):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def plot(results, scen_marks):
    """
    results    : {"MAPPO": (times, metrics), "Fixed": (times, metrics)}
    scen_marks : from MAPPO run (both controllers see same scenario sequence)
    """
    for metric, suffix, title, ylabel, _ in METRIC_INFO:
        fig, ax = plt.subplots(figsize=(12, 5))

        for name, (times, mdict) in results.items():
            raw = mdict[metric]
            smo = smooth(raw)
            ax.plot(times, raw, alpha=0.18, color=COLORS[name], linewidth=1.0)
            ax.plot(times, smo, label=name, color=COLORS[name], linewidth=2.2)

        for t_mark, _ in scen_marks:
            if t_mark > 0:
                ax.axvline(t_mark, color="gray", linestyle=":", linewidth=1.0, alpha=0.55)

        ymin, ymax = ax.get_ylim()
        label_y = ymax * 0.95 if ymax > 0 else 0.95
        for t_mark, scen_name in scen_marks:
            ax.text(t_mark + 5, label_y, scen_name,
                    fontsize=8, rotation=90, va="top", color="dimgray", alpha=0.8)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Simulation Time (s)", fontsize=11)
        ax.legend(fontsize=10, loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        plt.tight_layout()
        out = f"output/compare_network_{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────

def main():
    print("Loading MAPPO actor ...")
    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    actor.load_state_dict(torch.load(MAPPO_MODEL_PATH, map_location="cpu", weights_only=True))
    actor.eval()

    mappo_t, mappo_m, scen_marks = run_mappo(actor)
    fixed_t, fixed_m, _          = run_fixed()

    results = {
        "MAPPO": (mappo_t, mappo_m),
        "Fixed": (fixed_t, fixed_m),
    }
    plot(results, scen_marks)
    print("\nDone.")


if __name__ == "__main__":
    main()
