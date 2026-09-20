"""
mappo_vs_ppo_vs_fixed_junction_live.py
Step-by-step line graph comparison at junction 6073919354 (TL_A).

Runs three controllers sequentially and records per-step metrics at TL_A:
  1. SPPO   — single-agent PPO on SPPO_model/map.sumocfg
  2. MAPPO  — multi-agent PPO on network/triple.sumocfg (TL_A metrics only)
  3. Fixed  — fixed-time cycling on SPPO_model/map.sumocfg (same network as SPPO,
              for a fair single-intersection baseline)

X-axis is simulation time in seconds (SPPO: 1 step = 1 s, MAPPO/Fixed: 1 RL step = 3 s).
Five metrics plotted as separate line graphs, matching the PPO vs Fixed report style.

Output:
  output/compare_tla_live.png
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
import traci

# ── SUMO path guard ───────────────────────────────────────────────────────────
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

os.makedirs("output", exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
SPPO_MODEL_PATH  = "SPPO_model/ppo_traffic_final.zip"
SPPO_SUMOCFG     = os.path.abspath("SPPO_model/map.sumocfg")
MAPPO_MODEL_PATH = "mappo_models/best_actor.pth"
MAPPO_SCENARIOS  = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]
FIXED_PHASE_STEPS = 10            # RL steps per phase
TARGET_SIM_SECS  = 1800           # total sim seconds for MAPPO (chain episodes if needed)
MAPPO_DELTA_T    = 3              # sim seconds per MAPPO RL step

# TL_A lane lists (from SPPO_model/env.py — same physical junction in both networks)
TLA_INCOMING = [
    "-E5_0",
    "-465932558#1.34_0",
    "-465932558#1.34_1",
    "-465932558#1.34_2",
    "470773638#1_0",
    "465932558#0_0",
    "465932558#0_1",
]
TLA_OUTGOING = [
    "E5_0",
    "-465932558#0_0",
    "-465932558#0_1",
    "-470773638#1_0",
    "465932558#1_0",
]

METRIC_INFO = [
    # (key,         file_suffix,       title,                       y-axis label,              lower_better)
    ("wait",        "waiting_time",    "Waiting Time at TL_A",       "Waiting Time (s/lane)",   True),
    ("queue",       "queue_length",    "Queue Length at TL_A",       "Halting Vehicles / Lane", True),
    ("throughput",  "throughput",      "Cumulative Throughput at TL_A", "Vehicles Cleared",     False),
    ("stop_ratio",  "stop_ratio",      "Stop Ratio at TL_A",         "Fraction of Vehicles Stopped", True),
    ("pressure",    "pressure",        "Pressure at TL_A",           "Incoming − Outgoing Vehicles", True),
]

COLORS = {
    "MAPPO": "steelblue",
    "SPPO":  "darkorange",
    "Fixed": "tomato",
}


# ─────────────────────────────────────────────────────────────
# PER-STEP METRIC COLLECTION
# ─────────────────────────────────────────────────────────────

# Tracks vehicle IDs we've already counted as passing through TL_A's outgoing
# lanes. Reset at the start of each controller run via _reset_tla_throughput().
_seen_passed: set[str] = set()


def _reset_tla_throughput():
    """Call once before each controller's run so throughput counts start fresh."""
    _seen_passed.clear()


def _read_tla(label=None):
    """Read TL_A lane metrics from the active SUMO instance.

    Throughput is now TL_A-scoped: a vehicle is counted the FIRST time it is
    observed on any of TL_A's outgoing lanes. Summed across the run this is a
    direct count of vehicles that have crossed TL_A — comparable across
    single-intersection and triple-intersection setups.
    """
    if label:
        traci.switch(label)
    n_in = max(len(TLA_INCOMING), 1)
    try:
        halting = [traci.lane.getLastStepHaltingNumber(ln) for ln in TLA_INCOMING]
        total   = [traci.lane.getLastStepVehicleNumber(ln) for ln in TLA_INCOMING]
        veh_out = sum(traci.lane.getLastStepVehicleNumber(ln) for ln in TLA_OUTGOING)

        # Per-step throughput increment: vehicles newly seen on TL_A's exit lanes.
        passed_this_step = 0
        for ln in TLA_OUTGOING:
            for vid in traci.lane.getLastStepVehicleIDs(ln):
                if vid not in _seen_passed:
                    _seen_passed.add(vid)
                    passed_this_step += 1

        return {
            "wait":       sum(traci.lane.getWaitingTime(ln) for ln in TLA_INCOMING) / n_in,
            "queue":      sum(halting) / n_in,
            "throughput": float(passed_this_step),
            "stop_ratio": sum(halting) / max(sum(total), 1),
            "pressure":   float(sum(total) - veh_out),
        }
    except Exception:
        return {k: 0.0 for k in ("wait", "queue", "throughput", "stop_ratio", "pressure")}


# ─────────────────────────────────────────────────────────────
# CONTROLLER RUNNERS
# ─────────────────────────────────────────────────────────────

def run_sppo():
    print("\n" + "=" * 60)
    print("PHASE 1: SPPO on single-intersection")
    print("=" * 60)

    sys.path.insert(0, os.path.abspath("SPPO_model"))
    from stable_baselines3 import PPO as SB3PPO
    from env import SumoEnv as SppoEnv

    model = SB3PPO.load(SPPO_MODEL_PATH)
    env   = SppoEnv(sumo_cfg=SPPO_SUMOCFG, use_gui=False)
    obs, _ = env.reset()
    _reset_tla_throughput()

    times, metrics = [], {k: [] for k in ("wait", "queue", "throughput", "stop_ratio", "pressure")}
    cum_tp = 0.0
    t = 0
    done = False

    while not done:
        m = _read_tla(label=None)
        cum_tp += m["throughput"]
        times.append(float(t))
        for k in metrics:
            metrics[k].append(cum_tp if k == "throughput" else m[k])

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        t += 1   # 1 sim second per step

        if t % 200 == 0:
            print(f"  [SPPO] step {t}  wait={metrics['wait'][-1]:.2f}s  "
                  f"queue={metrics['queue'][-1]:.2f}")

    env.close()
    sys.path.pop(0)
    time.sleep(2)
    print(f"  [SPPO] done — {t} steps ({t}s simulation)")
    return np.array(times), {k: np.array(v) for k, v in metrics.items()}


def run_mappo():
    """
    MAPPO on triple-intersection. Chains multiple episodes back-to-back, cycling
    through MAPPO_SCENARIOS until total simulation time reaches TARGET_SIM_SECS.
    This matches SPPO's ~1800s episode duration and exposes MAPPO to varied demand.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: MAPPO on triple-intersection (TL_A metrics, scenario rotation)")
    print("=" * 60)

    import torch
    from mappo_env import MAPPOEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
    from mappo_networks import Actor

    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    actor.load_state_dict(torch.load(MAPPO_MODEL_PATH, map_location="cpu", weights_only=True))
    actor.eval()
    _reset_tla_throughput()

    times, metrics = [], {k: [] for k in ("wait", "queue", "throughput", "stop_ratio", "pressure")}
    scenario_marks = []   # list of (sim_seconds, scenario_name) for plot annotation
    cum_tp     = 0.0
    sim_secs   = 0
    rl_step    = 0
    scen_idx   = 0

    while sim_secs < TARGET_SIM_SECS:
        scenario = MAPPO_SCENARIOS[scen_idx % len(MAPPO_SCENARIOS)]
        scen_idx += 1
        scenario_marks.append((sim_secs, scenario))
        print(f"  [MAPPO] starting scenario '{scenario}' at t={sim_secs}s")

        env = MAPPOEnv(use_gui=False, scenario_name=scenario)
        obs_dict, _ = env.reset()
        done = False

        while not done and sim_secs < TARGET_SIM_SECS:
            m = _read_tla(label=env._label)
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
                print(f"  [MAPPO] t={sim_secs}s  wait={metrics['wait'][-1]:.2f}s  "
                      f"queue={metrics['queue'][-1]:.2f}")

        env.close()
        time.sleep(1)

    print(f"  [MAPPO] done — {rl_step} RL steps ({sim_secs}s simulation, "
          f"{scen_idx} scenarios)")
    return np.array(times), {k: np.array(v) for k, v in metrics.items()}, scenario_marks


def run_fixed():
    """Fixed-time on the SAME single-intersection network as SPPO (fair baseline)."""
    print("\n" + "=" * 60)
    print("PHASE 3: Fixed-Time on single-intersection (same network as SPPO)")
    print("=" * 60)

    sys.path.insert(0, os.path.abspath("SPPO_model"))
    from env import SumoEnv as SppoEnv

    # SPPO phase_map = [0, 6, 9, 12] — cycle through all 4 actions at fixed intervals
    N_SPPO_ACTIONS  = 4
    FIXED_HOLD_STEPS = 15   # steps per phase (~15s each at 1s/step)

    env = SppoEnv(sumo_cfg=SPPO_SUMOCFG, use_gui=False)
    obs, _ = env.reset()
    _reset_tla_throughput()

    times, metrics = [], {k: [] for k in ("wait", "queue", "throughput", "stop_ratio", "pressure")}
    cum_tp     = 0.0
    phase_idx  = 0
    hold_count = 0
    t = 0
    done = False

    while not done:
        m = _read_tla(label=None)
        cum_tp += m["throughput"]
        times.append(float(t))
        for k in metrics:
            metrics[k].append(cum_tp if k == "throughput" else m[k])

        # Cycle phases at fixed intervals
        if hold_count >= FIXED_HOLD_STEPS:
            phase_idx  = (phase_idx + 1) % N_SPPO_ACTIONS
            hold_count = 0
        obs, _, terminated, truncated, _ = env.step(phase_idx)
        done = terminated or truncated
        hold_count += 1
        t += 1

        if t % 200 == 0:
            print(f"  [Fixed] step {t}  wait={metrics['wait'][-1]:.2f}s  "
                  f"queue={metrics['queue'][-1]:.2f}")

    env.close()
    sys.path.pop(0)
    time.sleep(2)
    print(f"  [Fixed] done — {t} steps ({t}s simulation)")
    return np.array(times), {k: np.array(v) for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def smooth(y, w=15):
    """Simple moving-average smoothing."""
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def plot(results, scen_marks):
    """
    Save one figure per metric.

    results    : {"MAPPO": (times, metrics_dict), "SPPO": ..., "Fixed": ...}
    scen_marks : [(sim_seconds, scenario_name), ...] — MAPPO scenario boundaries
    """
    for metric, suffix, title, ylabel, lower_better in METRIC_INFO:
        fig, ax = plt.subplots(figsize=(12, 5))

        for name, (times, metrics) in results.items():
            raw = metrics[metric]
            smo = smooth(raw)
            ax.plot(times, raw, alpha=0.20, color=COLORS[name], linewidth=1.0)
            ax.plot(times, smo, label=name, color=COLORS[name], linewidth=2.2)

        # Dotted scenario boundary markers (MAPPO rotation)
        for t_mark, _ in scen_marks:
            if t_mark > 0:
                ax.axvline(t_mark, color="gray", linestyle=":", linewidth=1.0, alpha=0.55)

        # Scenario name labels along the top of the axes
        ymin, ymax = ax.get_ylim()
        label_y = ymax * 0.95 if ymax > 0 else 0.95
        for t_mark, scen_name in scen_marks:
            ax.text(t_mark + 5, label_y, scen_name,
                    fontsize=8, rotation=90, va="top",
                    color="dimgray", alpha=0.8)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Simulation Time (s)", fontsize=11)
        ax.legend(fontsize=10, loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        plt.tight_layout()
        out = f"output/compare_tla_{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    sppo_t,  sppo_m              = run_sppo()
    mappo_t, mappo_m, scen_marks = run_mappo()
    fixed_t, fixed_m             = run_fixed()

    results = {
        "MAPPO": (mappo_t, mappo_m),
        "SPPO":  (sppo_t,  sppo_m),
        "Fixed": (fixed_t, fixed_m),
    }
    plot(results, scen_marks)
    print("\nDone.")


if __name__ == "__main__":
    main()
