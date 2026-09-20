"""
evaluate_network.py

MAPPO vs Fixed-time on the full triple-intersection A1 corridor network.

Collects metrics at every junction across all six demand scenarios:
  - Network-wide aggregates (all three junctions combined)
  - Per-junction breakdown (TL_A, TL_B, TL_C individually)

Metrics: mean waiting time, mean queue length, total throughput, mean pressure

Output:
  output/network_eval_summary.csv        — per-controller per-scenario numbers
  output/network_eval_network.png        — network-wide bar charts
  output/network_eval_per_junction.png   — per-junction breakdown
"""

import os
import numpy as np
import pandas as pd
import torch
import traci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from mappo_env import MAPPOEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
from mappo_networks import Actor

os.makedirs("output", exist_ok=True)

# ─────────────────────────────────────────────────────────────
MODEL_PATH        = "mappo_models/best_actor.pth"
SCENARIOS         = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]
N_EPISODES        = 3
FIXED_PHASE_STEPS = 10       # steps per phase (10 × 3s = 30s)

COLORS = {"mappo": "steelblue", "fixed": "tomato"}
TL_LABELS = {
    "cluster_266437400_266437558": "TL_A",
    "317145283":                 "TL_B",
    "cluster_4162190061_4162190645": "TL_C",
}

# ─────────────────────────────────────────────────────────────
# METRIC COLLECTION
# ─────────────────────────────────────────────────────────────

_current_env = None


def _get_metrics(label):
    """
    Collect per-junction and network-wide metrics from the running SUMO instance.
    Returns a dict with per-TL values and network-wide aggregates.
    """
    traci.switch(label)
    env = _current_env

    per_tl = {}
    for tl in TL_IDS:
        in_lanes  = env._controlled_lanes.get(tl, [])
        out_lanes = env._outgoing_lanes.get(tl, [])
        n_in      = max(len(in_lanes), 1)

        try:
            halting   = [traci.lane.getLastStepHaltingNumber(ln) for ln in in_lanes]
            total_veh = [traci.lane.getLastStepVehicleNumber(ln) for ln in in_lanes]
            wait      = [traci.lane.getWaitingTime(ln)           for ln in in_lanes]
            out_veh   = sum(traci.lane.getLastStepVehicleNumber(ln) for ln in out_lanes)

            per_tl[tl] = {
                "wait":       sum(wait)    / n_in,
                "queue":      sum(halting) / n_in,
                "throughput": float(out_veh),
                "pressure":   float(sum(total_veh) - out_veh),
            }
        except Exception:
            per_tl[tl] = {"wait": 0.0, "queue": 0.0, "throughput": 0.0, "pressure": 0.0}

    # Network-wide: average wait/queue/pressure across junctions, sum throughput
    network = {
        "wait":       np.mean([per_tl[tl]["wait"]     for tl in TL_IDS]),
        "queue":      np.mean([per_tl[tl]["queue"]    for tl in TL_IDS]),
        "throughput": sum(    per_tl[tl]["throughput"] for tl in TL_IDS),
        "pressure":   np.mean([per_tl[tl]["pressure"] for tl in TL_IDS]),
    }

    return {"per_tl": per_tl, "network": network}


# ─────────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────────

def run_episode(scenario, controller, actor=None):
    global _current_env

    env = MAPPOEnv(use_gui=False, scenario_name=scenario)
    _current_env = env
    obs_dict, _ = env.reset()

    step_metrics  = []
    phase_timer   = 0
    fixed_phase   = 0
    done          = False

    while not done:
        m = _get_metrics(env._label)
        step_metrics.append(m)

        if controller == "mappo":
            actions = {}
            for tl in TL_IDS:
                obs_t = torch.tensor(obs_dict[tl], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = actor.get_action(obs_t, deterministic=True)
                actions[tl] = int(action.item())

        elif controller == "fixed":
            if phase_timer >= FIXED_PHASE_STEPS:
                fixed_phase = (fixed_phase + 1) % N_ACTIONS
                phase_timer = 0
            actions     = {tl: fixed_phase for tl in TL_IDS}
            phase_timer += 1

        obs_dict, _, _, done, _ = env.step(actions)

    env.close()
    _current_env = None

    # Aggregate over steps
    result = {
        "network": {
            "mean_wait":   np.mean([m["network"]["wait"]      for m in step_metrics]),
            "mean_queue":  np.mean([m["network"]["queue"]     for m in step_metrics]),
            "throughput":  np.sum( [m["network"]["throughput"] for m in step_metrics]),
            "mean_pressure": np.mean([m["network"]["pressure"] for m in step_metrics]),
        },
        "per_tl": {
            tl: {
                "mean_wait":     np.mean([m["per_tl"][tl]["wait"]      for m in step_metrics]),
                "mean_queue":    np.mean([m["per_tl"][tl]["queue"]     for m in step_metrics]),
                "throughput":    np.sum( [m["per_tl"][tl]["throughput"] for m in step_metrics]),
                "mean_pressure": np.mean([m["per_tl"][tl]["pressure"]  for m in step_metrics]),
            }
            for tl in TL_IDS
        },
    }
    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading MAPPO actor ...")
    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    actor.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    actor.eval()

    records = []

    for scenario in SCENARIOS:
        for controller in ["mappo", "fixed"]:
            ep_results = []
            for ep in range(N_EPISODES):
                print(f"  {controller:6s} | {scenario:15s} | ep {ep+1}/{N_EPISODES}", end="\r")
                ep_results.append(
                    run_episode(scenario, controller, actor if controller == "mappo" else None)
                )

            # Average network-wide metrics across episodes
            net = {
                "controller": controller,
                "scenario":   scenario,
                "mean_wait":      np.mean([r["network"]["mean_wait"]     for r in ep_results]),
                "mean_queue":     np.mean([r["network"]["mean_queue"]    for r in ep_results]),
                "throughput":     np.mean([r["network"]["throughput"]    for r in ep_results]),
                "mean_pressure":  np.mean([r["network"]["mean_pressure"] for r in ep_results]),
            }
            records.append(net)

            # Per-junction records
            for tl in TL_IDS:
                tl_rec = {
                    "controller": controller,
                    "scenario":   scenario,
                    "junction":   TL_LABELS[tl],
                    "mean_wait":      np.mean([r["per_tl"][tl]["mean_wait"]     for r in ep_results]),
                    "mean_queue":     np.mean([r["per_tl"][tl]["mean_queue"]    for r in ep_results]),
                    "throughput":     np.mean([r["per_tl"][tl]["throughput"]    for r in ep_results]),
                    "mean_pressure":  np.mean([r["per_tl"][tl]["mean_pressure"] for r in ep_results]),
                }
                records.append(tl_rec)

            print(f"  {controller:6s} | {scenario:15s} | "
                  f"wait={net['mean_wait']:6.2f}s  "
                  f"queue={net['mean_queue']:5.2f}  "
                  f"throughput={net['throughput']:7.1f}  "
                  f"pressure={net['mean_pressure']:5.2f}")

    df = pd.DataFrame(records)
    df.to_csv("output/network_eval_summary.csv", index=False)
    print("\nSaved: output/network_eval_summary.csv")

    _plot_network(df)
    _plot_per_junction(df)
    _print_table(df)


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def _plot_network(df):
    net_df = df[~df.get("junction", pd.Series(dtype=str)).notna()].copy()

    metrics = ["mean_wait", "mean_queue", "throughput", "mean_pressure"]
    titles  = [
        "Mean Waiting Time (s/lane)",
        "Mean Queue Length (veh/lane)",
        "Total Throughput (vehicles)",
        "Mean Pressure (veh)",
    ]

    x      = np.arange(len(SCENARIOS))
    labels = [s.replace("_", " ") for s in SCENARIOS]

    for metric, title in zip(metrics, titles):
        fig, ax = plt.subplots(figsize=(10, 5))

        for ctrl in ["mappo", "fixed"]:
            vals = [
                net_df[(net_df.controller == ctrl) & (net_df.scenario == s)][metric].values[0]
                for s in SCENARIOS
            ]
            ax.plot(x, vals, marker="o", linewidth=2.2, markersize=6,
                    color=COLORS[ctrl], label=ctrl.upper())
            ax.fill_between(x, vals, alpha=0.08, color=COLORS[ctrl])

        ax.set_title(f"Network-Wide {title} — MAPPO vs Fixed-Time",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=11)
        ax.set_xlabel("Demand Scenario", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.3, len(SCENARIOS) - 0.7)

        plt.tight_layout()
        suffix = metric.replace("mean_", "")
        out = f"output/network_eval_{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def _plot_per_junction(df):
    junc_df = df[df.get("junction", pd.Series(dtype=str)).notna()].copy()
    if junc_df.empty:
        return

    junctions = ["TL_A", "TL_B", "TL_C"]
    metrics   = ["mean_wait", "mean_queue", "throughput", "mean_pressure"]
    titles    = ["Waiting Time (s/lane)", "Queue (veh/lane)", "Throughput (veh)", "Pressure (veh)"]

    x      = np.arange(len(SCENARIOS))
    labels = [s.replace("_", " ") for s in SCENARIOS]

    for metric, title in zip(metrics, titles):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
        fig.suptitle(f"{title} per Junction — MAPPO vs Fixed-Time",
                     fontsize=12, fontweight="bold")

        for ax, junc in zip(axes, junctions):
            sub = junc_df[junc_df.junction == junc]

            for ctrl in ["mappo", "fixed"]:
                vals = [
                    sub[(sub.controller == ctrl) & (sub.scenario == s)][metric].values[0]
                    if len(sub[(sub.controller == ctrl) & (sub.scenario == s)]) > 0 else 0
                    for s in SCENARIOS
                ]
                ax.plot(x, vals, marker="o", linewidth=2.2, markersize=6,
                        color=COLORS[ctrl], label=ctrl.upper())
                ax.fill_between(x, vals, alpha=0.08, color=COLORS[ctrl])

            ax.set_title(junc, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.3, len(SCENARIOS) - 0.7)

        axes[0].set_ylabel(title, fontsize=10)
        plt.tight_layout()
        suffix = metric.replace("mean_", "")
        out = f"output/network_eval_junctions_{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────

def _print_table(df):
    net_df = df[~df.get("junction", pd.Series(dtype=str)).notna()].copy()

    print("\n" + "=" * 75)
    print("NETWORK-WIDE RESULTS — MAPPO vs Fixed-Time")
    print("=" * 75)

    for metric, label, lb in [
        ("mean_wait",     "Mean Waiting Time (s/lane)",    True),
        ("mean_queue",    "Mean Queue (veh/lane)",          True),
        ("throughput",    "Total Throughput (vehicles)",    False),
        ("mean_pressure", "Mean Pressure (veh)",            True),
    ]:
        print(f"\n  {label}")
        print(f"  {'Scenario':<18} {'MAPPO':>10} {'Fixed':>10}  {'vs Fixed':>10}")
        print("  " + "-" * 55)
        for s in SCENARIOS:
            m = net_df[(net_df.controller == "mappo") & (net_df.scenario == s)][metric].values
            f = net_df[(net_df.controller == "fixed") & (net_df.scenario == s)][metric].values
            if len(m) == 0 or len(f) == 0:
                continue
            m, f = m[0], f[0]
            pct  = ((f - m) / max(abs(f), 1e-6) * 100) if lb else ((m - f) / max(abs(f), 1e-6) * 100)
            sign = "+" if pct > 0 else ""
            print(f"  {s:<18} {m:>10.2f} {f:>10.2f}  {sign}{pct:>8.1f}%")


if __name__ == "__main__":
    main()
