"""
ppo_vs_mappo_live_gui.py
Live animated comparison dashboard at junction 6073919354 (TL_A).

Opens TWO SUMO-GUI windows simultaneously:
  - SPPO  → single-intersection network (SPPO_model/map.sumocfg), driven by ppo_traffic_final.zip
  - MAPPO → triple-intersection network (network/triple.sumocfg), driven by best_actor.pth

A matplotlib window shows 5 stacked subplots that update LIVE as both simulations step:
  Delay · Queue · Throughput · Stop Ratio · Pressure
Each chart has two lines (SPPO orange, MAPPO blue) growing in real time.

Close the matplotlib window to stop both simulations cleanly.
"""

# --- Performance Analysis/ lives outside src/; add ../src to the path so the
# flat project imports (mappo_env, mappo_networks, sumo_env) resolve. Run these
# scripts from the repository root so data paths (network/, mappo_models/) work.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))


import os
import sys
import numpy as np
import traci
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# ── config ────────────────────────────────────────────────────────────────────
SPPO_MODEL_PATH  = "SPPO_model/ppo_traffic_final.zip"
SPPO_SUMOCFG     = os.path.abspath("SPPO_model/map.sumocfg")
MAPPO_MODEL_PATH = "mappo_models/best_actor.pth"
MAPPO_SCENARIOS  = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]

# TL_A lanes (same physical junction in both networks)
INCOMING = [
    "1314322247#13_0",
    "1314322247#13_1",
    "28534416#6_0",
    "415417667#7_0",
    "415417667#7_1",
]
OUTGOING = [
    "1314322247#14_0",
    "1314322247#14_1",
    "28534416#7_0",
]


# ─────────────────────────────────────────────────────────────
# METRIC READING — pulled directly from whichever TraCI conn is active
# ─────────────────────────────────────────────────────────────

def read_metrics():
    veh_ids = traci.vehicle.getIDList()

    # Delay (1 − speed/allowed) averaged across all vehicles
    delays = []
    for vid in veh_ids:
        try:
            v = traci.vehicle.getSpeed(vid)
            allowed = traci.vehicle.getAllowedSpeed(vid)
            delays.append(1 - v / allowed if allowed > 0 else 0)
        except Exception:
            pass
    delay = float(np.mean(delays)) if delays else 0.0

    # Queue on TL_A incoming lanes
    queue = float(np.mean([traci.lane.getLastStepHaltingNumber(l) for l in INCOMING]))

    # Throughput: vehicles on TL_A outgoing lanes (instantaneous)
    throughput = float(sum(traci.lane.getLastStepVehicleNumber(l) for l in OUTGOING))

    # Stop ratio: fraction of all vehicles with speed < 0.1
    stops = sum(1 for vid in veh_ids if traci.vehicle.getSpeed(vid) < 0.1)
    stop_ratio = stops / len(veh_ids) if veh_ids else 0.0

    # Pressure on TL_A
    inc = sum(traci.lane.getLastStepVehicleNumber(l) for l in INCOMING)
    out = sum(traci.lane.getLastStepVehicleNumber(l) for l in OUTGOING)
    pressure = float(inc - out)

    return delay, queue, throughput, stop_ratio, pressure


# ─────────────────────────────────────────────────────────────
# ENV / MODEL SETUP
# ─────────────────────────────────────────────────────────────

print("Loading SPPO model + env...")
sys.path.insert(0, os.path.abspath("SPPO_model"))
from stable_baselines3 import PPO as SB3PPO
from env import SumoEnv as SppoEnv

sppo_model = SB3PPO.load(SPPO_MODEL_PATH)
sppo_env   = SppoEnv(sumo_cfg=SPPO_SUMOCFG, use_gui=True)
sppo_obs, _ = sppo_env.reset()
SPPO_LABEL = "default"   # SppoEnv calls traci.start() without label
sys.path.pop(0)

print("Loading MAPPO actor + env...")
import torch
import mappo_env as _menv
_menv.MAX_SIM_STEPS = 10**9
from mappo_env import MAPPOEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
from mappo_networks import Actor

mappo_actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
mappo_actor.load_state_dict(torch.load(MAPPO_MODEL_PATH, map_location="cpu", weights_only=True))
mappo_actor.eval()

_mappo_scen_idx = 0
def _next_scenario():
    """Cycle through MAPPO_SCENARIOS on each call."""
    global _mappo_scen_idx
    name = MAPPO_SCENARIOS[_mappo_scen_idx % len(MAPPO_SCENARIOS)]
    _mappo_scen_idx += 1
    return name

mappo_env = MAPPOEnv(use_gui=True, scenario_name=_next_scenario())
mappo_obs, _ = mappo_env.reset()
MAPPO_LABEL = mappo_env._label
current_mappo_scenario = MAPPO_SCENARIOS[(_mappo_scen_idx - 1) % len(MAPPO_SCENARIOS)]

# Histories
sppo_hist  = {k: [] for k in ("delay", "queue", "throughput", "stops", "pressure")}
mappo_hist = {k: [] for k in ("delay", "queue", "throughput", "stops", "pressure")}


# ─────────────────────────────────────────────────────────────
# PLOT SETUP
# ─────────────────────────────────────────────────────────────

plt.style.use("ggplot")
fig, axs = plt.subplots(5, 1, figsize=(13, 16))
ax_delay, ax_queue, ax_tp, ax_stops, ax_press = axs

PLOTS = [
    (ax_delay, "delay",      "Delay (0 = no delay, 1 = full stop)",  (0, 1)),
    (ax_queue, "queue",      "Queue (halting veh / lane)",            None),
    (ax_tp,    "throughput", "Throughput (veh on outgoing lanes)",    None),
    (ax_stops, "stops",      "Stop Ratio (fraction stopped)",         (0, 1)),
    (ax_press, "pressure",   "Pressure (incoming − outgoing)",        None),
]


# ─────────────────────────────────────────────────────────────
# ANIMATION UPDATE — step BOTH sims and redraw
# ─────────────────────────────────────────────────────────────

def update(frame):
    global sppo_obs, mappo_obs, mappo_env, MAPPO_LABEL, current_mappo_scenario

    # ── Step SPPO ────────────────────────────────────────────
    try:
        traci.switch(SPPO_LABEL)
        action, _ = sppo_model.predict(sppo_obs, deterministic=True)
        sppo_obs, _, term, trunc, _ = sppo_env.step(action)
        traci.switch(SPPO_LABEL)   # ensure still on SPPO conn for reading
        d, q, tp, s, p = read_metrics()
        sppo_hist["delay"].append(d)
        sppo_hist["queue"].append(q)
        sppo_hist["throughput"].append(tp)
        sppo_hist["stops"].append(s)
        sppo_hist["pressure"].append(p)
        if term or trunc:
            sppo_obs, _ = sppo_env.reset()
    except Exception as e:
        print(f"[SPPO] error: {e}")

    # ── Step MAPPO ───────────────────────────────────────────
    try:
        actions = {}
        for tl in TL_IDS:
            obs_t = torch.tensor(mappo_obs[tl], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a, _, _ = mappo_actor.get_action(obs_t, deterministic=True)
            actions[tl] = int(a.item())
        mappo_obs, _, _, done, _ = mappo_env.step(actions)
        traci.switch(MAPPO_LABEL)
        d, q, tp, s, p = read_metrics()
        mappo_hist["delay"].append(d)
        mappo_hist["queue"].append(q)
        mappo_hist["throughput"].append(tp)
        mappo_hist["stops"].append(s)
        mappo_hist["pressure"].append(p)
        if done:
            try:
                mappo_env.close()
            except Exception:
                pass
            current_mappo_scenario = _next_scenario()
            print(f"[MAPPO] episode ended — starting scenario '{current_mappo_scenario}'")
            mappo_env = MAPPOEnv(use_gui=True, scenario_name=current_mappo_scenario)
            mappo_obs, _ = mappo_env.reset()
            MAPPO_LABEL = mappo_env._label
    except Exception as e:
        print(f"[MAPPO] error: {e}")

    # ── Redraw all 5 subplots ─────────────────────────────────
    for ax, key, title, ylim in PLOTS:
        ax.clear()
        if sppo_hist[key]:
            ax.plot(sppo_hist[key],  color="darkorange", linewidth=1.8, label="SPPO")
        if mappo_hist[key]:
            ax.plot(mappo_hist[key], color="steelblue",  linewidth=1.8, label="MAPPO")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)

    fig.suptitle(
        f"TL_A (6073919354) — LIVE SPPO vs MAPPO  ·  MAPPO scenario: '{current_mappo_scenario}'  ·  close window to stop",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

print("\nStarting live animation — close the matplotlib window to stop.\n")
ani = FuncAnimation(fig, update, interval=80, repeat=False, cache_frame_data=False)

try:
    plt.show()
finally:
    print("\nShutting down simulations...")
    try:
        traci.switch(SPPO_LABEL); sppo_env.close()
    except Exception:
        pass
    try:
        mappo_env.close()
    except Exception:
        pass
    print("Done.")
