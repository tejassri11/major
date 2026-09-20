"""
mappo_vs_ppo_vs_fixed_junction.py
Direct comparison of SPPO vs MAPPO vs Fixed-Time at junction 6073919354 (TL_A).

  SPPO   — single-agent PPO trained on the single-intersection network (SPPO_model/map.sumocfg).
            Evaluated in its native environment. Junction 6073919354 is the only TL.
  MAPPO  — multi-agent PPO trained on the triple-intersection network.
            Only junction 6073919354 metrics are extracted from the triple-network run.
  Fixed  — fixed-time cycling, evaluated alongside MAPPO on the triple-intersection network.
            Only junction 6073919354 metrics are extracted.

Metrics collected for junction 6073919354:
  wait      — mean waiting time (s/lane) across incoming lanes
  queue     — mean queue length (halting vehicles/lane)
  throughput— total vehicles that cleared the junction per episode
  stop_ratio— fraction of incoming vehicles that are stopped each step (mean)
  pressure  — mean (incoming vehicles − outgoing vehicles) per step

Outputs:
  output/compare_tla_all_metrics.png   — 5-metric grouped bar chart
  output/compare_tla_summary.csv       — raw numbers
"""

# --- Performance Analysis/ lives outside src/; add ../src to the path so the
# flat project imports (mappo_env, mappo_networks, sumo_env) resolve. Run these
# scripts from the repository root so data paths (network/, mappo_models/) work.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))


import os
import sys
import numpy as np
import pandas as pd
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
SCENARIOS        = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]
N_EPISODES       = 3           # per scenario for MAPPO/Fixed; total for SPPO
FIXED_PHASE_STEPS = 10         # RL steps per phase for fixed-time
TL_A             = "6073919354"
EPISODE_STEPS    = 500

# Hard-coded TL_A lanes (from SPPO_model/env.py — same junction in both networks)
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

_COLORS = {"mappo": "steelblue", "sppo": "darkorange", "fixed": "tomato"}
_LABELS = {"mappo": "MAPPO", "sppo": "SPPO (single-intersection)", "fixed": "Fixed-Time"}


# ─────────────────────────────────────────────────────────────
# METRIC COLLECTION — works for any running SUMO instance
# ─────────────────────────────────────────────────────────────

def collect_tla(label=None):
    """
    Pull all 5 TL_A metrics from the running SUMO instance.
    Pass label=None for SPPO (default TraCI connection) or label string for MAPPO/Fixed.
    """
    if label:
        traci.switch(label)

    n_in  = max(len(TLA_INCOMING), 1)
    n_out = max(len(TLA_OUTGOING), 1)

    try:
        # Waiting time (s) averaged across incoming lanes
        wait = sum(traci.lane.getWaitingTime(ln) for ln in TLA_INCOMING) / n_in

        # Queue: halting vehicles averaged across incoming lanes
        halting_in = [traci.lane.getLastStepHaltingNumber(ln) for ln in TLA_INCOMING]
        queue = sum(halting_in) / n_in

        # Stop ratio: stopped / total on incoming lanes
        total_in = sum(traci.lane.getLastStepVehicleNumber(ln) for ln in TLA_INCOMING)
        stopped  = sum(halting_in)
        stop_ratio = stopped / total_in if total_in > 0 else 0.0

        # Pressure: incoming vehicles − outgoing vehicles
        veh_in  = sum(traci.lane.getLastStepVehicleNumber(ln) for ln in TLA_INCOMING)
        veh_out = sum(traci.lane.getLastStepVehicleNumber(ln) for ln in TLA_OUTGOING)
        pressure = float(veh_in - veh_out)

        # Throughput: vehicles that completed their trip this step (whole sim step)
        throughput = float(traci.simulation.getArrivedNumber())

    except Exception:
        wait, queue, stop_ratio, pressure, throughput = 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        "wait":       wait,
        "queue":      queue,
        "stop_ratio": stop_ratio,
        "pressure":   pressure,
        "throughput": throughput,
    }


def summarise(step_records):
    """Aggregate per-step records → episode summary dict."""
    keys = step_records[0].keys()
    return {
        k: float(np.sum([r[k] for r in step_records])
                 if k == "throughput"
                 else np.mean([r[k] for r in step_records]))
        for k in keys
    }


# ─────────────────────────────────────────────────────────────
# SPPO RUNNER
# ─────────────────────────────────────────────────────────────

def run_sppo(n_eps):
    sys.path.insert(0, os.path.abspath("SPPO_model"))
    from stable_baselines3 import PPO as SB3PPO
    from env import SumoEnv as SppoEnv

    model = SB3PPO.load(SPPO_MODEL_PATH)
    results = []

    for ep in range(n_eps):
        print(f"  [SPPO] episode {ep+1}/{n_eps}", end="\r")
        env = SppoEnv(sumo_cfg=SPPO_SUMOCFG, use_gui=False)
        obs, _ = env.reset()
        records = []
        done = False

        while not done:
            records.append(collect_tla(label=None))   # default TraCI connection
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.close()
        results.append(summarise(records))

    sys.path.pop(0)
    return results


# ─────────────────────────────────────────────────────────────
# MAPPO / FIXED RUNNERS
# ─────────────────────────────────────────────────────────────

def run_mappo(scenario, actor):
    from mappo_env import MAPPOEnv, TL_IDS, N_ACTIONS
    import torch

    env = MAPPOEnv(use_gui=False, scenario_name=scenario)
    obs_dict, _ = env.reset()
    records = []
    done = False
    step = 0

    while not done and step < EPISODE_STEPS:
        records.append(collect_tla(label=env._label))

        actions = {}
        for tl in TL_IDS:
            obs_t = torch.tensor(obs_dict[tl], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_t, deterministic=True)
            actions[tl] = int(action.item())

        obs_dict, _, _, done, _ = env.step(actions)
        step += 1

    env.close()
    return summarise(records)


def run_fixed(scenario):
    from mappo_env import MAPPOEnv, TL_IDS, N_ACTIONS

    env = MAPPOEnv(use_gui=False, scenario_name=scenario)
    obs_dict, _ = env.reset()
    records = []
    done = False
    phase_timer = 0
    fixed_phase = 0
    step = 0

    while not done and step < EPISODE_STEPS:
        records.append(collect_tla(label=env._label))

        if phase_timer >= FIXED_PHASE_STEPS:
            fixed_phase = (fixed_phase + 1) % N_ACTIONS
            phase_timer = 0
        obs_dict, _, _, done, _ = env.step({tl: fixed_phase for tl in TL_IDS})
        phase_timer += 1
        step += 1

    env.close()
    return summarise(records)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # ── Load MAPPO actor ──────────────────────────────────────
    import torch
    from mappo_env import LOCAL_OBS_DIM, N_ACTIONS
    from mappo_networks import Actor

    print("Loading MAPPO actor...")
    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    actor.load_state_dict(torch.load(MAPPO_MODEL_PATH, map_location="cpu", weights_only=True))
    actor.eval()

    # ── Run MAPPO and Fixed across all scenarios ──────────────
    mappo_eps, fixed_eps = [], []
    for scenario in SCENARIOS:
        for ep in range(N_EPISODES):
            print(f"  [MAPPO] {scenario} ep {ep+1}/{N_EPISODES}", end="\r")
            mappo_eps.append(run_mappo(scenario, actor))
            print(f"  [Fixed] {scenario} ep {ep+1}/{N_EPISODES}", end="\r")
            fixed_eps.append(run_fixed(scenario))
    print()

    # ── Run SPPO ──────────────────────────────────────────────
    n_sppo = N_EPISODES * len(SCENARIOS)
    print(f"Running SPPO ({n_sppo} episodes)...")
    sppo_eps = run_sppo(n_sppo)
    print()

    # ── Aggregate ─────────────────────────────────────────────
    def agg(eps):
        keys = eps[0].keys()
        return {k: (np.mean([e[k] for e in eps]), np.std([e[k] for e in eps])) for k in keys}

    mappo_agg = agg(mappo_eps)
    fixed_agg = agg(fixed_eps)
    sppo_agg  = agg(sppo_eps)

    # ── Save CSV ──────────────────────────────────────────────
    rows = []
    for ctrl, a in [("mappo", mappo_agg), ("sppo", sppo_agg), ("fixed", fixed_agg)]:
        row = {"controller": ctrl}
        for k, (m, s) in a.items():
            row[k] = m
            row[k + "_std"] = s
        rows.append(row)
    pd.DataFrame(rows).to_csv("output/compare_tla_summary.csv", index=False)
    print("Saved: output/compare_tla_summary.csv")

    _plot(mappo_agg, sppo_agg, fixed_agg)
    _print_table(mappo_agg, sppo_agg, fixed_agg)


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def _plot(mappo, sppo, fixed):
    metrics = [
        ("wait",       "Mean Waiting Time\n(s/lane)",            True),
        ("queue",      "Mean Queue Length\n(veh/lane)",          True),
        ("throughput", "Total Throughput\n(vehicles/episode)",   False),
        ("stop_ratio", "Mean Stop Ratio\n(0=none, 1=all stopped)", True),
        ("pressure",   "Mean Pressure\n(incoming − outgoing veh)", True),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(
        "Junction 6073919354 (TL_A): SPPO vs MAPPO vs Fixed-Time\n"
        "SPPO evaluated on single-intersection network · MAPPO & Fixed on triple-intersection network",
        fontsize=11, fontweight="bold"
    )

    ctrls = ["mappo", "sppo", "fixed"]
    aggs  = {"mappo": mappo, "sppo": sppo, "fixed": fixed}

    for ax, (metric, ylabel, lower_better) in zip(axes, metrics):
        means = [aggs[c][metric][0] for c in ctrls]
        stds  = [aggs[c][metric][1] for c in ctrls]
        x     = np.arange(3)

        bars = ax.bar(x, means, width=0.55,
                      color=[_COLORS[c] for c in ctrls],
                      alpha=0.85, yerr=stds, capsize=5,
                      error_kw={"elinewidth": 1.5})

        ax.set_title(ylabel, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["MAPPO", "SPPO", "Fixed"], fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # % vs Fixed labels
        fixed_val = aggs["fixed"][metric][0]
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            if ctrls[i] == "fixed" or fixed_val == 0:
                continue
            pct = (fixed_val - mean) / abs(fixed_val) * 100
            if not lower_better:
                pct = -pct
            sign = "+" if pct > 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + std + 0.02 * max(means),
                    f"{sign}{pct:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="green" if pct > 0 else "red")

    plt.tight_layout()
    plt.savefig("output/compare_tla_all_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: output/compare_tla_all_metrics.png")
    plt.close()


def _print_table(mappo, sppo, fixed):
    metrics = [
        ("wait",       "Waiting time (s/lane)",     True),
        ("queue",      "Queue (veh/lane)",           True),
        ("throughput", "Throughput (veh/episode)",   False),
        ("stop_ratio", "Stop ratio",                 True),
        ("pressure",   "Pressure (veh diff)",        True),
    ]

    print("\n" + "=" * 72)
    print("TL_A (6073919354) COMPARISON — SPPO vs MAPPO vs Fixed-Time")
    print("=" * 72)
    print(f"{'Metric':<28} {'MAPPO':>10} {'SPPO':>10} {'Fixed':>10}  "
          f"{'vs Fixed\n(MAPPO)':>12} {'vs Fixed\n(SPPO)':>12}")
    print("-" * 72)

    aggs = {"mappo": mappo, "sppo": sppo, "fixed": fixed}
    for metric, label, lower_better in metrics:
        m = aggs["mappo"][metric][0]
        s = aggs["sppo"][metric][0]
        f = aggs["fixed"][metric][0]
        def pct(x):
            if f == 0:
                return 0.0
            return (f - x) / abs(f) * 100 if lower_better else (x - f) / abs(f) * 100
        print(f"  {label:<26} {m:>10.3f} {s:>10.3f} {f:>10.3f}  "
              f"{pct(m):>+10.1f}%  {pct(s):>+10.1f}%")

    print("=" * 72)
    print("\nNote: SPPO evaluated on single-intersection network (map.sumocfg).")
    print("      MAPPO and Fixed evaluated on triple-intersection network.")


if __name__ == "__main__":
    main()
