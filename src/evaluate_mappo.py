"""
Evaluation script for the trained MAPPO policy.

Runs three controllers across all six traffic scenarios and collects
per-step metrics pulled directly from TraCI after each env step.

Controllers:
  mappo      — best_actor.pth, deterministic greedy
  fixed      — cycles phases 0→1→2→0 every FIXED_PHASE_STEPS steps (fixed-time)
  random     — uniform random phase selection (sanity baseline)

Output:
  eval_summary.csv      — mean per controller per scenario
  eval_comparison.png   — grouped bar chart
"""

import numpy as np
import pandas as pd
import torch
import traci
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from mappo_env import MAPPOEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
from mappo_networks import Actor

# ─────────────────────────────────────────────────────────────
MODEL_PATH        = "mappo_models/best_actor.pth"
SCENARIOS         = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]
N_EPISODES        = 3        # episodes averaged per controller per scenario
FIXED_PHASE_STEPS = 10       # steps per phase for fixed-time (10 × 3s = 30s each)

# Junction we report on. TL_A (6073919354) is physically the same junction
# present in the single-intersection Fixed/SPPO network, so reporting only
# on this TL gives an apples-to-apples comparison.
EVAL_TL           = "cluster_266437400_266437558"


def load_actor():
    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    actor.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    actor.eval()
    return actor


def _get_metrics(label):
    """Pull raw traffic metrics from the running SUMO instance via TraCI.

    Metrics are collected ONLY at EVAL_TL (junction 6073919354 / TL_A), which
    is the same physical junction the Fixed/SPPO single-intersection baselines
    measure. MAPPO still controls all three TLs — only reporting is scoped.
    """
    traci.switch(label)
    env_obj = _current_env          # set before each episode

    try:
        in_lanes  = env_obj._controlled_lanes.get(EVAL_TL, [])
        out_lanes = env_obj._outgoing_lanes.get(EVAL_TL, []) \
                    if hasattr(env_obj, "_outgoing_lanes") else []

        total_wait  = sum(traci.lane.getWaitingTime(ln)         for ln in in_lanes)
        total_queue = sum(traci.lane.getLastStepHaltingNumber(ln) for ln in in_lanes)
        # Per-step throughput at TL_A = vehicles currently on its outgoing lanes.
        # Summed across the episode this is comparable to the Fixed/SPPO chart
        # which counts vehicles passing through the same junction.
        throughput  = sum(traci.lane.getLastStepVehicleNumber(ln) for ln in out_lanes)
        collisions  = traci.simulation.getCollidingVehiclesNumber()
        teleports   = traci.simulation.getStartingTeleportNumber()
        n_lanes     = max(len(in_lanes), 1)
        return {
            "wait":       total_wait  / n_lanes,
            "queue":      total_queue / n_lanes,
            "throughput": float(throughput),
            "collisions": float(collisions),
            "teleports":  float(teleports),
        }
    except Exception:
        return {"wait": 0.0, "queue": 0.0, "throughput": 0.0,
                "collisions": 0.0, "teleports": 0.0}


_current_env = None   # global reference so _get_metrics can reach _controlled_lanes


def run_episode(scenario, controller, actor=None):
    global _current_env

    env = MAPPOEnv(use_gui=False, scenario_name=scenario)
    _current_env = env
    obs_dict, _ = env.reset()

    step_metrics = []
    step         = 0
    phase_timer  = 0
    fixed_phase  = 0
    done = False

    while not done:
        # ── collect metrics from current state (before stepping) ───
        # Must happen here — env.step() closes the TraCI connection
        # on the final step (done=True), so querying after step fails.
        m = _get_metrics(env._label)
        step_metrics.append(m)

        # ── choose action ──────────────────────────────────────────
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
            actions = {tl: fixed_phase for tl in TL_IDS}
            phase_timer += 1

        elif controller == "random":
            actions = {tl: np.random.randint(N_ACTIONS) for tl in TL_IDS}

        obs_dict, _, _, done, _ = env.step(actions)
        step += 1

    env.close()
    _current_env = None

    return {
        "mean_wait":   np.mean([m["wait"]       for m in step_metrics]),
        "mean_queue":  np.mean([m["queue"]       for m in step_metrics]),
        "throughput":  np.sum( [m["throughput"]  for m in step_metrics]),
        "collisions":  np.sum( [m["collisions"]  for m in step_metrics]),
        "teleports":   np.sum( [m["teleports"]   for m in step_metrics]),
        "steps":       step,
    }


def main():
    print("Loading best actor …")
    actor = load_actor()

    records = []
    for scenario in SCENARIOS:
        for controller in ["mappo", "fixed", "random"]:
            ep_actor = actor if controller == "mappo" else None
            ep_data  = []
            for ep in range(N_EPISODES):
                print(f"  {controller:8s} | {scenario:15s} | ep {ep+1}/{N_EPISODES}", end="\r")
                ep_data.append(run_episode(scenario, controller, ep_actor))

            rec = {
                "controller":  controller,
                "scenario":    scenario,
                "mean_wait":   np.mean([r["mean_wait"]  for r in ep_data]),
                "std_wait":    np.std( [r["mean_wait"]  for r in ep_data]),
                "mean_queue":  np.mean([r["mean_queue"] for r in ep_data]),
                "std_queue":   np.std( [r["mean_queue"] for r in ep_data]),
                "throughput":  np.mean([r["throughput"] for r in ep_data]),
                "collisions":  np.mean([r["collisions"] for r in ep_data]),
                "teleports":   np.mean([r["teleports"]  for r in ep_data]),
            }
            records.append(rec)
            print(f"  {controller:8s} | {scenario:15s} | "
                  f"wait={rec['mean_wait']:6.1f}s  "
                  f"queue={rec['mean_queue']:5.2f}  "
                  f"throughput={rec['throughput']:6.1f}")

    df = pd.DataFrame(records)
    df.to_csv("eval_summary.csv", index=False)
    print("\nSaved: eval_summary.csv")

    _plot(df)
    _print_table(df)


def _plot(df):
    metrics = ["mean_wait",  "mean_queue", "throughput"]
    titles  = ["Mean Waiting Time (s/lane)", "Mean Queue Length (veh/lane)", "Total Throughput (vehicles)"]
    colors  = {"mappo": "steelblue", "fixed": "tomato", "random": "#aaaaaa"}

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("MAPPO vs Fixed-Time vs Random — All Scenarios", fontsize=13, fontweight="bold")

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax    = fig.add_subplot(1, 3, idx + 1)
        x     = np.arange(len(SCENARIOS))
        w     = 0.25

        for j, ctrl in enumerate(["mappo", "fixed", "random"]):
            vals = [df[(df.controller == ctrl) & (df.scenario == s)][metric].values[0]
                    for s in SCENARIOS]
            stds = []
            if metric in ("mean_wait", "mean_queue"):
                stds = [df[(df.controller == ctrl) & (df.scenario == s)][f"std_{metric.split('_')[1]}"].values[0]
                        for s in SCENARIOS]
            ax.bar(x + (j - 1) * w, vals, width=w,
                   color=colors[ctrl], label=ctrl.upper(), alpha=0.85,
                   yerr=stds if stds else None, capsize=3, error_kw={"elinewidth": 1})

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/eval_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: output/eval_comparison.png")


def _print_table(df):
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)
    for metric, label, lower_is_better in [
        ("mean_wait",  "Mean Wait (s/lane)",          True),
        ("mean_queue", "Mean Queue (veh/lane)",        True),
        ("throughput", "Total Throughput (vehicles)",  False),
    ]:
        print(f"\n  {label}")
        print(f"  {'Scenario':<18} {'MAPPO':>9} {'Fixed':>9} {'Random':>9}  {'vs Fixed':>10}")
        print("  " + "-" * 60)
        for s in SCENARIOS:
            m = df[(df.controller == "mappo") & (df.scenario == s)][metric].values[0]
            f = df[(df.controller == "fixed") & (df.scenario == s)][metric].values[0]
            r = df[(df.controller == "random") & (df.scenario == s)][metric].values[0]
            if lower_is_better:
                pct = (f - m) / max(abs(f), 1e-6) * 100
            else:
                pct = (m - f) / max(abs(f), 1e-6) * 100
            sign = "+" if pct > 0 else ""
            print(f"  {s:<18} {m:>9.2f} {f:>9.2f} {r:>9.2f}  {sign}{pct:>8.1f}%")


if __name__ == "__main__":
    main()
