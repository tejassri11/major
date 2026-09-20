"""
mappo_vs_ppo_vs_fixed_network.py
Compare MAPPO, SA-PPO, and Fixed-time controllers on the triple-intersection network.

Outputs:
  output/compare_summary.csv     — mean per controller per scenario (overall + TL_A)
  output/compare_overall.png     — grouped bar chart, all 3 TLs
  output/compare_tla.png         — same metrics, TL_A lanes only
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
import torch
import traci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

from stable_baselines3 import PPO as SB3PPO
from mappo_env import MAPPOEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
from mappo_networks import Actor
from sumo_env import SumoEnv

# ── config ────────────────────────────────────────────────────────────────────
MAPPO_MODEL      = "mappo_models/best_actor.pth"
PPO_MODEL        = "ppo_models/best_model.zip"
SCENARIOS        = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]
N_EPISODES       = 3
FIXED_PHASE_STEPS = 10   # steps per phase for fixed-time
EPISODE_STEPS    = 500   # uniform episode length for all controllers
TL_A             = "6073919354"

os.makedirs("output", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_mappo_actor():
    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    actor.load_state_dict(torch.load(MAPPO_MODEL, map_location="cpu", weights_only=True))
    actor.eval()
    return actor


def load_ppo_model():
    return SB3PPO.load(PPO_MODEL)


# ─────────────────────────────────────────────────────────────
# METRIC COLLECTION
# ─────────────────────────────────────────────────────────────

def _collect(label, controlled_lanes):
    """Pull overall and TL_A metrics from the live SUMO instance."""
    traci.switch(label)
    result = {"overall": {}, "tla": {}}
    try:
        all_lanes = [ln for tl in TL_IDS for ln in controlled_lanes.get(tl, [])]
        tla_lanes = controlled_lanes.get(TL_A, [])
        n_all = max(len(all_lanes), 1)
        n_tla = max(len(tla_lanes), 1)

        result["overall"] = {
            "wait":       sum(traci.lane.getWaitingTime(ln) for ln in all_lanes) / n_all,
            "queue":      sum(traci.lane.getLastStepHaltingNumber(ln) for ln in all_lanes) / n_all,
            "throughput": float(traci.simulation.getArrivedNumber()),
        }
        result["tla"] = {
            "wait":  sum(traci.lane.getWaitingTime(ln) for ln in tla_lanes) / n_tla,
            "queue": sum(traci.lane.getLastStepHaltingNumber(ln) for ln in tla_lanes) / n_tla,
        }
    except Exception:
        result["overall"] = {"wait": 0.0, "queue": 0.0, "throughput": 0.0}
        result["tla"]     = {"wait": 0.0, "queue": 0.0}
    return result


def _ep_summary(step_records):
    """Aggregate per-step records into a single episode dict."""
    overall = step_records[0]["overall"].keys()
    tla     = step_records[0]["tla"].keys()
    return {
        **{f"overall_{k}": np.mean([r["overall"][k] for r in step_records]) for k in overall},
        **{f"tla_{k}":     np.mean([r["tla"][k]     for r in step_records]) for k in tla},
        "overall_throughput": float(np.sum([r["overall"]["throughput"] for r in step_records])),
    }


# ─────────────────────────────────────────────────────────────
# EPISODE RUNNERS
# ─────────────────────────────────────────────────────────────

def run_mappo_episode(scenario, actor):
    env = MAPPOEnv(use_gui=False, scenario_name=scenario)
    obs_dict, _ = env.reset()
    records = []
    done = False
    step = 0

    while not done and step < EPISODE_STEPS:
        records.append(_collect(env._label, env._controlled_lanes))

        actions = {}
        for tl in TL_IDS:
            obs_t = torch.tensor(obs_dict[tl], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_t, deterministic=True)
            actions[tl] = int(action.item())

        obs_dict, _, _, done, _ = env.step(actions)
        step += 1

    env.close()
    return _ep_summary(records)


def run_fixed_episode(scenario):
    env = MAPPOEnv(use_gui=False, scenario_name=scenario)
    obs_dict, _ = env.reset()
    records = []
    done = False
    phase_timer = 0
    fixed_phase = 0
    step = 0

    while not done and step < EPISODE_STEPS:
        records.append(_collect(env._label, env._controlled_lanes))

        if phase_timer >= FIXED_PHASE_STEPS:
            fixed_phase = (fixed_phase + 1) % N_ACTIONS
            phase_timer = 0
        actions = {tl: fixed_phase for tl in TL_IDS}
        phase_timer += 1

        obs_dict, _, _, done, _ = env.step(actions)
        step += 1

    env.close()
    return _ep_summary(records)


def run_ppo_episode(scenario, ppo_model):
    env = SumoEnv(use_gui=False, scenario_name=scenario)
    obs, _ = env.reset()
    records = []
    done = False
    step = 0

    while not done and step < EPISODE_STEPS:
        records.append(_collect(env._label, env._controlled_lanes))
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    env.close()
    return _ep_summary(records)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading models...")
    actor     = load_mappo_actor()
    ppo_model = load_ppo_model()

    rows = []
    for scenario in SCENARIOS:
        for ctrl_name, runner in [
            ("mappo", lambda s: run_mappo_episode(s, actor)),
            ("ppo",   lambda s: run_ppo_episode(s, ppo_model)),
            ("fixed", run_fixed_episode),
        ]:
            ep_data = []
            for ep in range(N_EPISODES):
                print(f"  {ctrl_name:6s} | {scenario:15s} | ep {ep+1}/{N_EPISODES}", end="\r")
                ep_data.append(runner(scenario))

            row = {"controller": ctrl_name, "scenario": scenario}
            for key in ep_data[0]:
                vals = [e[key] for e in ep_data]
                row[key]             = float(np.mean(vals))
                row[key + "_std"]    = float(np.std(vals))
            rows.append(row)
            print(f"  {ctrl_name:6s} | {scenario:15s} | "
                  f"wait={row['overall_wait']:6.1f}s  "
                  f"queue={row['overall_queue']:5.2f}  "
                  f"tla_wait={row['tla_wait']:6.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv("output/compare_summary.csv", index=False)
    print("\nSaved: output/compare_summary.csv")

    _plot_overall(df)
    _plot_tla(df)
    _print_table(df)


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

_COLORS = {"mappo": "steelblue", "ppo": "darkorange", "fixed": "tomato"}
_LABELS = {"mappo": "MAPPO", "ppo": "SA-PPO", "fixed": "Fixed-Time"}


def _grouped_bar(ax, df, metric, title, ylabel, lower_is_better=True):
    x   = np.arange(len(SCENARIOS))
    w   = 0.25
    for j, ctrl in enumerate(["mappo", "ppo", "fixed"]):
        vals = [df[(df.controller == ctrl) & (df.scenario == s)][metric].values[0]
                for s in SCENARIOS]
        stds = [df[(df.controller == ctrl) & (df.scenario == s)].get(metric + "_std", pd.Series([0])).values[0]
                for s in SCENARIOS]
        ax.bar(x + (j - 1) * w, vals, width=w,
               color=_COLORS[ctrl], label=_LABELS[ctrl], alpha=0.85,
               yerr=stds, capsize=3, error_kw={"elinewidth": 1})
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _plot_overall(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("MAPPO vs SA-PPO vs Fixed-Time — All Intersections", fontsize=13, fontweight="bold")
    _grouped_bar(axes[0], df, "overall_wait",       "Mean Waiting Time (s/lane)",    "seconds/lane")
    _grouped_bar(axes[1], df, "overall_queue",      "Mean Queue Length (veh/lane)",  "vehicles/lane")
    _grouped_bar(axes[2], df, "overall_throughput", "Total Throughput (vehicles)",   "vehicles", lower_is_better=False)
    plt.tight_layout()
    plt.savefig("output/compare_overall.png", dpi=150, bbox_inches="tight")
    print("Saved: output/compare_overall.png")
    plt.close()


def _plot_tla(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"MAPPO vs SA-PPO vs Fixed-Time — TL_A ({TL_A}) Only", fontsize=13, fontweight="bold")
    _grouped_bar(axes[0], df, "tla_wait",  "Mean Waiting Time (s/lane)",   "seconds/lane")
    _grouped_bar(axes[1], df, "tla_queue", "Mean Queue Length (veh/lane)", "vehicles/lane")
    plt.tight_layout()
    plt.savefig("output/compare_tla.png", dpi=150, bbox_inches="tight")
    print("Saved: output/compare_tla.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────

def _print_table(df):
    metrics = [
        ("overall_wait",       "Overall Mean Wait (s/lane)",         True),
        ("overall_queue",      "Overall Mean Queue (veh/lane)",       True),
        ("overall_throughput", "Overall Total Throughput (vehicles)", False),
        ("tla_wait",           "TL_A Mean Wait (s/lane)",            True),
        ("tla_queue",          "TL_A Mean Queue (veh/lane)",         True),
    ]
    ctrls = ["mappo", "ppo", "fixed"]

    print("\n" + "=" * 90)
    print("CONTROLLER COMPARISON RESULTS")
    print("=" * 90)

    for metric, label, lower_better in metrics:
        print(f"\n  {label}")
        print(f"  {'Scenario':<18} {'MAPPO':>10} {'SA-PPO':>10} {'Fixed':>10}  "
              f"{'vs Fixed (MAPPO)':>18} {'vs Fixed (PPO)':>16}")
        print("  " + "-" * 84)
        for s in SCENARIOS:
            vals = {}
            for c in ctrls:
                row = df[(df.controller == c) & (df.scenario == s)]
                vals[c] = row[metric].values[0] if len(row) else 0.0
            f = vals["fixed"]
            def pct(x):
                if lower_better:
                    return (f - x) / max(abs(f), 1e-6) * 100
                return (x - f) / max(abs(f), 1e-6) * 100
            pm = pct(vals["mappo"])
            pp = pct(vals["ppo"])
            print(f"  {s:<18} {vals['mappo']:>10.2f} {vals['ppo']:>10.2f} {vals['fixed']:>10.2f}  "
                  f"{pm:>+16.1f}%  {pp:>+13.1f}%")


if __name__ == "__main__":
    main()
