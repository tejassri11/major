"""
Multi-agent SUMO environment for the Palapye triple-intersection network.

This file is the BRIDGE between the real-world traffic simulation (SUMO) and the
reinforcement learning algorithm (MAPPO). Think of it as the "game" that the agents
learn to play.

HOW IT FITS INTO MAPPO:
  - Each RL "step" = agents pick phases → SUMO simulates a few seconds → we measure queues
  - reset() starts a fresh SUMO episode, returns initial observations
  - step() applies the agents' chosen phases, advances simulation, returns new obs + reward
  - The environment handles the messy details: yellow transitions, gridlock guards,
    vehicle injection, speed limits, curriculum scenario selection

KEY DISTINCTION FROM SINGLE-AGENT ENV (sumo_env.py):
  - sumo_env.py returned one flat observation for ALL traffic lights combined
  - MAPPOEnv returns TWO things every step:
      obs_dict     = {tl_id: np.array(22,)} — per-agent LOCAL observations
      global_state = np.array(66,)          — ALL observations concatenated
    The Actor only ever sees its own local obs (22 numbers).
    The Critic sees the global_state (66 numbers = all 3 × 22) during training.
    This split IS the CTDE principle in practice.
"""

import os
import random
import numpy as np
import sumolib
import traci

if "SUMO_HOME" not in os.environ:
    try:
        sumo_bin = sumolib.checkBinary("sumo")
        os.environ["SUMO_HOME"] = os.path.dirname(os.path.dirname(os.path.abspath(sumo_bin)))
    except Exception:
        pass

from traffic_scenario import TrafficScenario
import traffic_injector
import green_wave
from tl_programs import apply_tl_programs

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
# Path to the SUMO network configuration file (XML describing roads, junctions, etc.)
SUMO_CFG = "network/triple.sumocfg"

# The three traffic light IDs in the Delhi-Gurgaon NH48 network file.
TL_IDS   = ["cluster_266437400_266437558", "317145283", "cluster_4162190061_4162190645"]

# DELTA_T: how many SUMO simulation seconds pass per RL step.
DELTA_T       = 3

# Yellow is mandatory between any two green phases to prevent sudden changes (unsafe).
YELLOW_DUR    = 1

# MIN_GREEN: minimum number of RL steps before the agent can switch phases.
# = 5 steps × 3 seconds = 15 seconds minimum green. Prevents rapid flickering.
# Real-world traffic lights typically hold green for at least 10–15 seconds.
MIN_GREEN     = 5

# MAX_GREEN: after this many steps at the same green, the gridlock guard may force a rotation.
# = 40 steps × 3 seconds = 120 seconds. Prevents one direction monopolising the intersection.
MAX_GREEN     = 40

# MAX_SIM_STEPS: total RL steps per episode before the episode ends automatically.
# = 500 steps × 3 seconds = 1500 seconds = 25 simulated minutes.
# Shorter episodes mean more resets → more curriculum diversity → tighter credit assignment.
# 25 minutes of simulated traffic is enough for queues to build and dissipate several times.
MAX_SIM_STEPS = 500

# N_LANES_PER_TL: number of incoming lanes we track per traffic light.
# Each intersection has 8 controlled incoming lanes (2 per arm × 4 arms).
N_LANES_PER_TL     = 8

# N_OUT_LANES_PER_TL: outgoing lanes we track per traffic light.
# 4 outgoing lanes (1 per arm) — tells the agent "is there space on the other side?"
N_OUT_LANES_PER_TL = 4

# OBS_MAX_QUEUE: cap for queue length before log-scaling. Any queue ≥ 15 cars = "fully jammed".
OBS_MAX_QUEUE      = 15.0

# LOG_OBS_MAX: log1p(15) = precomputed divisor so queue values end up in [0, 1].
# We use log(1+x) instead of x directly because congestion has diminishing urgency:
# going from 0→1 car matters more than going from 14→15 cars.
LOG_OBS_MAX        = float(np.log1p(OBS_MAX_QUEUE))

# OBS_MAX_WAIT: cap for waiting time. Beyond 300 seconds (5 minutes) = "very late".
OBS_MAX_WAIT       = 300.0

# LOG_OBS_MAX_WAIT: precomputed log1p(300) for normalising waiting time to [0, 1].
LOG_OBS_MAX_WAIT   = float(np.log1p(OBS_MAX_WAIT))

# MAX_QUEUE: same cap used inside the reward calculation for the queue penalty term.
MAX_QUEUE          = 15.0

# ─── IRC PCU (Passenger Car Unit) Equivalents for Indian Traffic ──────────────
IRC_PCU_WEIGHTS = {
    "two_wheeler":     0.5,   # Motorcycles / Scooters
    "moto":            0.5,
    "auto_rickshaw":   1.0,   # Three-Wheelers
    "car_small":       1.0,   # Hatchbacks
    "car_normal":      1.0,   # Sedans
    "car_suv":         1.2,   # Crossovers / SUVs
    "car_sport":       1.0,
    "bus_city":        3.0,   # City transit buses
    "bus_articulated": 3.5,
    "truck_lcv":       1.5,   # Light commercial trucks
    "truck":           3.0,   # Medium / Heavy trucks
    "trailer":         4.0,   # Long articulated trucks
}


def _get_lane_pcu_halting(lane: str) -> float:
    """Calculate halting queue in Passenger Car Units (PCU) for Indian mixed traffic."""
    try:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane)
        if not veh_ids:
            return 0.0
        pcu = 0.0
        for vid in veh_ids:
            if traci.vehicle.getSpeed(vid) < 0.1:
                vtype = traci.vehicle.getTypeID(vid)
                pcu += IRC_PCU_WEIGHTS.get(vtype, 1.0)
        return pcu
    except Exception:
        try:
            return float(traci.lane.getLastStepHaltingNumber(lane))
        except Exception:
            return 0.0


def _get_lane_pcu_count(lane: str) -> float:
    """Calculate vehicle load in PCU on an outgoing lane."""
    try:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane)
        if not veh_ids:
            return 0.0
        pcu = 0.0
        for vid in veh_ids:
            vtype = traci.vehicle.getTypeID(vid)
            pcu += IRC_PCU_WEIGHTS.get(vtype, 1.0)
        return pcu
    except Exception:
        try:
            return float(traci.lane.getLastStepVehicleNumber(lane))
        except Exception:
            return 0.0


# LOCAL_OBS_DIM: how many numbers describe ONE traffic light's local situation.
#   8 queue lengths + 8 waiting times + 4 outgoing counts + 1 phase + 1 phase_state = 22
# This is the input size to the Actor network.
LOCAL_OBS_DIM  = N_LANES_PER_TL + N_LANES_PER_TL + N_OUT_LANES_PER_TL + 2   # 22

# GLOBAL_STATE_DIM: all three local observations concatenated into one big vector.
# = 3 × 22 = 66. This is the input size to the CentralizedCritic network.
GLOBAL_STATE_DIM = len(TL_IDS) * LOCAL_OBS_DIM                               # 66
N_ACTIONS      = 3   # Discrete(3) for every TL

# MAJOR_GREEN_PHASES: the SUMO phase indices (from the tls.add.xml program) that
# correspond to the agent's action choices 0, 1, 2.
# Phase 0 = N/S right-turns green, Phase 2 = E/W all green, Phase 4 = N/S all green.
# These are the "safe" green states — yellow transitions live at indices 1, 3, 5.
MAJOR_GREEN_PHASES = {
    "cluster_266437400_266437558": [0, 2, 4],
    "317145283":                 [0, 2, 4],
    "cluster_4162190061_4162190645": [0, 2, 4],
}

# YELLOW_AFTER: maps each major green phase index → its following yellow phase index.
YELLOW_AFTER = {
    "cluster_266437400_266437558": {0: 1, 2: 3, 4: 5},
    "317145283":                 {0: 1, 2: 3, 4: 5},
    "cluster_4162190061_4162190645": {0: 1, 2: 3, 4: 5},
}

# CURRICULUM: training difficulty schedule.
CURRICULUM = [
    (0,  ["low", "normal"]),
    (30, ["normal", "rush_hour_am", "rush_hour_pm", "holiday"]),
    (80, None),   # None = use TrafficScenario.random() = full random
]


class MAPPOEnv:
    """
    Multi-agent SUMO environment wrapper.

    This class manages one complete SUMO simulation instance. In training we create
    N_ENVS = 4 instances, each running independently in the same Python process.
    SUMO supports this via "labels" — each instance gets a unique ID so TraCI calls
    are routed to the correct simulation.

    Interface:
        reset()  → obs_dict {tl_id: array(22,)}, global_state array(66,)
        step(actions: {tl_id: int}) → obs_dict, global_state, reward, terminated, info

    Actions:  {tl_id: 0 | 1 | 2}  — one phase choice per traffic light per step
    Reward:   single float, shared by all agents (cooperative MARL)
    """

    # Class-level counter so each new instance gets a unique SUMO label.
    # Without unique labels, traci.start() would crash with a "label already exists" error.
    _instance_count = 0

    def __init__(self, use_gui=False, scenario_name=None, gui_delay=0):
        MAPPOEnv._instance_count += 1
        # e.g. "mappo_1", "mappo_2", ... — used as the TraCI connection label
        self._label        = f"mappo_{MAPPOEnv._instance_count}"
        self.use_gui       = use_gui          # True = open the SUMO visual GUI window
        self.gui_delay     = gui_delay        # SUMO GUI step delay in ms (e.g. 20-50 for smooth visual playback)
        self.scenario_name = scenario_name   # None = use curriculum; string = fixed scenario

        # Per-traffic-light state dictionaries, keyed by TL_ID.
        # All populated during _reset_inner(), empty at construction time.
        self._controlled_lanes = {}   # {tl: [lane_ids...]} — incoming lanes we control
        self._outgoing_lanes   = {}   # {tl: [lane_ids...]} — exit lanes (downstream space)
        self._phase_lanes      = {}   # {tl: {action_idx: [lane_ids...]}} — lanes served per phase
        self._phase_state      = {}   # {tl: "GREEN" | "YELLOW"} — current TL state
        self._state_timer      = {}   # {tl: int} — countdown steps remaining in yellow
        self._current_action   = {}   # {tl: int} — which green phase is currently active
        self._pending_action   = {}   # {tl: int} — which green phase to switch to after yellow
        self._green_time       = {}   # {tl: int} — steps held at current green (for MIN/MAX_GREEN)

        self._sim_step      = 0       # simulation step counter (resets each episode)
        self._sumo_running  = False   # True when a SUMO process is alive
        self._episode_count = 0       # cumulative across all resets — drives curriculum

    # ─────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────
    def reset(self):
        """
        Start a new episode. Launches (or restarts) SUMO, injects initial vehicles,
        and returns the first observation.

        The retry loop (up to 3 attempts) handles rare SUMO startup failures —
        e.g. port collisions or leftover processes from a previous crash. If all 3
        attempts fail, the exception propagates up and the caller decides what to do.
        """
        for attempt in range(3):
            try:
                return self._reset_inner()
            except Exception as e:
                print(f"[ENV {self._label}] reset attempt {attempt+1} failed: {e}")
                try:
                    traci.switch(self._label)
                    traci.close()
                except Exception:
                    pass
                self._sumo_running = False
                if attempt == 2:
                    raise
        # Fallback: return zero observations so training can continue even on failure
        dummy_obs   = {tl: np.zeros(LOCAL_OBS_DIM,    dtype=np.float32) for tl in TL_IDS}
        dummy_state = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)
        return dummy_obs, dummy_state

    def _reset_inner(self):
        """
        The actual reset logic. Called by reset() with retry wrapping.

        Steps performed:
          1. Close any existing SUMO process
          2. Pick a traffic scenario (curriculum or fixed)
          3. Launch SUMO via TraCI
          4. Apply traffic light programs (phase definitions)
          5. Cap all edge speeds to 60 km/h (Palapye urban limit)
          6. Initialise the vehicle injector for this scenario
          7. Compute green-wave offsets (coordinated signal timing)
          8. Build per-TL lane lists and phase-to-lane mappings
          9. Warm up by running 20 simulation steps before returning
        """
        # Close old SUMO if it's still running (e.g. after early episode termination)
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

        # ── Scenario selection ────────────────────────────────────────────────
        # If a specific scenario name was provided at construction, always use it.
        # Otherwise, walk the curriculum table and pick from the appropriate pool.
        if self.scenario_name:
            scenario = TrafficScenario(self.scenario_name)
        else:
            pool = None
            for threshold, scenarios in CURRICULUM:
                if self._episode_count >= threshold:
                    pool = scenarios
            # pool=None means we've passed all thresholds → fully random scenario
            scenario = (TrafficScenario.random()
                        if pool is None
                        else TrafficScenario(random.choice(pool)))
        self._episode_count += 1

        # ── Launch SUMO ───────────────────────────────────────────────────────
        _bin_name = "sumo-gui" if self.use_gui else "sumo"
        try:
            binary = sumolib.checkBinary(_bin_name)
        except Exception:
            _sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
            binary = os.path.join(_sumo_home, "bin", _bin_name + ".exe")
        sumo_cmd = [binary, "-c", SUMO_CFG, "--no-warnings", "--no-step-log"]
        if self.use_gui:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.gui_delay > 0:
                sumo_cmd.extend(["--delay", str(self.gui_delay)])
        traci.start(sumo_cmd, label=self._label)   # unique label → multiple sims can coexist
        traci.switch(self._label)        # tell TraCI which sim our commands go to
        self._sumo_running = True
        self._sim_step     = 0

        # ── Traffic light programs ────────────────────────────────────────────
        # Loads our custom phase definitions from tls.add.xml so the intersections
        # start with the 6-phase program (3 green + 3 yellow) we designed.
        apply_tl_programs()

        # ── Speed cap ─────────────────────────────────────────────────────────
        # 60 km/h → 16.67 m/s. Palapye is urban, 60 km/h is the road limit.
        # We cap all edges at startup to prevent vehicles speeding on highway edges.
        _max_ms = 60.0 / 3.6
        for eid in traci.edge.getIDList():
            try:
                if traci.edge.getMaxSpeed(eid) > _max_ms:
                    traci.edge.setMaxSpeed(eid, _max_ms)
            except Exception:
                pass

        # ── Vehicle injector ──────────────────────────────────────────────────
        # init() validates all OD (origin–destination) pairs against the live network
        # and builds the scenario-specific injection pool. MUST be called after SUMO starts.
        traffic_injector.init(scenario)

        # ── Green wave ────────────────────────────────────────────────────────
        # Computes signal offsets so vehicles traveling between intersections at
        # ~50 km/h encounter green lights without stopping. Improves baseline flow.
        gw = green_wave.create(TL_IDS)
        gw.bootstrap(free_flow_kmh=50.0)
        gw.enabled = True

        # ── Per-TL initialisation ─────────────────────────────────────────────
        for tl in TL_IDS:
            # Controlled lanes: the incoming lanes whose signals this TL manages.
            # dict.fromkeys() deduplicate while preserving order (sets don't preserve order).
            self._controlled_lanes[tl] = list(
                dict.fromkeys(traci.trafficlight.getControlledLanes(tl))
            )

            # Outgoing (downstream) lanes: where vehicles go AFTER crossing the junction.
            # Monitoring outgoing lanes tells the agent whether there's congestion downstream
            # (i.e., whether green here would just push cars into a jam on the other side).
            links_tl = traci.trafficlight.getControlledLinks(tl)
            out = []
            for link_group in links_tl:
                for (_, to_lane, _) in link_group:
                    if to_lane and to_lane not in out:
                        out.append(to_lane)
            self._outgoing_lanes[tl] = out

            # Phase-to-lane mapping: for each of the 3 agent actions, which incoming
            # lanes are served by that green phase?
            # This is used ONLY by the gridlock guard (not the observation) — it lets
            # the guard compare congestion between the current phase and alternatives.
            try:
                programs    = traci.trafficlight.getAllProgramLogics(tl)
                active_id   = traci.trafficlight.getProgram(tl)
                active_prog = next((p for p in programs if p.programID == active_id), programs[0])
                # phase_strs: {phase_index: "GGGrrr..."} — one char per signal link
                phase_strs  = {i: p.state for i, p in enumerate(active_prog.phases)}
            except Exception:
                phase_strs = {}

            self._phase_lanes[tl] = {}
            for act_idx, ph_idx in enumerate(MAJOR_GREEN_PHASES[tl]):
                state_str = phase_strs.get(ph_idx, "")
                served = []
                for li, ch in enumerate(state_str):
                    # 'G' = protected green (no conflict), 'g' = permissive green (yield applies)
                    # Both mean vehicles CAN move, so both count as "served".
                    if ch in ('G', 'g') and li < len(links_tl):
                        for (from_lane, _, _) in links_tl[li]:
                            if from_lane and from_lane not in served:
                                served.append(from_lane)
                self._phase_lanes[tl][act_idx] = served

            # Initialise phase state machine for this traffic light
            self._phase_state[tl]    = "GREEN"   # start green
            self._state_timer[tl]    = 0
            self._current_action[tl] = 0          # start on action 0 (phase index 0)
            self._pending_action[tl] = 0
            self._green_time[tl]     = 0
            traci.trafficlight.setPhase(tl, MAJOR_GREEN_PHASES[tl][0])

        # ── Warm-up steps ─────────────────────────────────────────────────────
        # Run 20 simulation steps before the episode "officially" begins.
        # This seeds the network with vehicles so the first observation isn't empty.
        # Starting from an empty network would give misleading zero-queue observations.
        for _ in range(20):
            traci.simulationStep()
            self._sim_step += 1
            traffic_injector.inject(self._sim_step)

        return self._get_obs()   # returns (obs_dict, global_state)

    # ─────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────
    def step(self, actions: dict):
        """
        Apply one set of agent actions and advance the simulation by DELTA_T seconds.

        actions: {tl_id: int}  — one integer in {0, 1, 2} per traffic light

        Returns:
          obs_dict     — {tl_id: array(22,)}  new local observations
          global_state — array(66,)           concatenated all-agent observations
          reward       — float                shared cooperative reward
          terminated   — bool                 True if episode has reached MAX_SIM_STEPS
          info         — {}                   empty dict (placeholder for extra diagnostics)

        PHASE STATE MACHINE (per traffic light):
          State "GREEN":
            - Every step: increment green_time counter
            - If agent requests a DIFFERENT phase AND green_time >= MIN_GREEN:
                → enter YELLOW, set pending_action to requested phase
            - If green_time >= MAX_GREEN and a more congested alternative exists:
                → force a rotation (gridlock guard)
            - Otherwise: extend current green indefinitely (setPhaseDuration=9999)
          State "YELLOW":
            - Countdown state_timer each step
            - When timer reaches 0: activate pending_action's green phase, back to GREEN
        """
        try:
            traci.switch(self._label)

            for tl in TL_IDS:
                req   = int(actions[tl])     # requested action: 0, 1, or 2
                state = self._phase_state[tl]

                if state == "GREEN":
                    self._green_time[tl] += 1
                    # Normal switch: different phase requested AND held green long enough
                    want_switch = (req != self._current_action[tl]
                                   and self._green_time[tl] >= MIN_GREEN)

                    # ── Gridlock guard ────────────────────────────────────────
                    # If the agent keeps requesting the current phase even after MAX_GREEN steps,
                    # check whether another phase has MORE waiting vehicles.
                    # If so, override the agent and force a rotation.
                    # This prevents a scenario where one direction starves completely while
                    # the agent incorrectly holds green on a less-congested direction.
                    if not want_switch and self._green_time[tl] >= MAX_GREEN:
                        cur       = self._current_action[tl]
                        cur_lanes = self._phase_lanes.get(tl, {}).get(cur, [])
                        cur_q     = (sum(traci.lane.getLastStepHaltingNumber(l) for l in cur_lanes)
                                     if cur_lanes else 0)
                        best_req, best_q = cur, -1
                        for alt in range(len(MAJOR_GREEN_PHASES[tl])):
                            if alt == cur:
                                continue
                            alt_lanes = self._phase_lanes.get(tl, {}).get(alt, [])
                            alt_q = (sum(traci.lane.getLastStepHaltingNumber(l) for l in alt_lanes)
                                     if alt_lanes else 0)
                            if alt_q > best_q:
                                best_q, best_req = alt_q, alt
                        # Only force rotation if the alternative has MORE queue than current
                        if best_req != cur and best_q > cur_q:
                            req, want_switch = best_req, True

                    if want_switch:
                        # Transition: green → yellow → new green
                        cur_green = MAJOR_GREEN_PHASES[tl][self._current_action[tl]]
                        yellow    = YELLOW_AFTER[tl][cur_green]
                        traci.trafficlight.setPhase(tl, yellow)
                        traci.trafficlight.setPhaseDuration(tl, 9999)   # hold yellow until timer expires
                        self._phase_state[tl]    = "YELLOW"
                        self._state_timer[tl]    = YELLOW_DUR            # steps to stay yellow
                        self._pending_action[tl] = req                   # remember where to go after
                    else:
                        # No switch: extend current green so SUMO doesn't auto-advance phases
                        traci.trafficlight.setPhaseDuration(tl, 9999)

                elif state == "YELLOW":
                    self._state_timer[tl] -= 1
                    if self._state_timer[tl] <= 0:
                        # Yellow expired: activate the new green phase
                        target = MAJOR_GREEN_PHASES[tl][self._pending_action[tl]]
                        traci.trafficlight.setPhase(tl, target)
                        traci.trafficlight.setPhaseDuration(tl, 9999)
                        self._phase_state[tl]    = "GREEN"
                        self._current_action[tl] = self._pending_action[tl]
                        self._green_time[tl]     = 0   # reset the green-time counter

            # ── Advance simulation ────────────────────────────────────────────
            # Run DELTA_T=3 SUMO steps, injecting new vehicles at each step.
            # Each SUMO step = 1 real second of simulation time.
            for _ in range(DELTA_T):
                traci.simulationStep()
                self._sim_step += 1
                traffic_injector.inject(self._sim_step)

            obs_dict, global_state = self._get_obs()
            reward     = self._compute_reward()
            terminated = self._sim_step >= MAX_SIM_STEPS

            if terminated:
                try:
                    traci.close()
                except Exception:
                    pass
                self._sumo_running = False

            return obs_dict, global_state, reward, terminated, {}

        except Exception as e:
            # If SUMO crashes mid-episode (rare but possible), end the episode cleanly
            # so the training loop can reset and continue rather than hanging.
            print(f"[ENV {self._label}] step() crashed: {e} — ending episode")
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False
            dummy_obs   = {tl: np.zeros(LOCAL_OBS_DIM,    dtype=np.float32) for tl in TL_IDS}
            dummy_state = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)
            return dummy_obs, dummy_state, 0.0, True, {}

    def close(self):
        """Cleanly shut down the SUMO process. Call this at the end of training."""
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

    # ─────────────────────────────────────────────────────────────
    # OBSERVATION  (local per-agent + global state)
    # ─────────────────────────────────────────────────────────────
    def _get_obs(self):
        """
        Build the observation vectors for all three traffic lights simultaneously.

        Returns:
          obs_dict    : {tl_id: np.array(22,)}  — one per agent, for the Actor
          global_state: np.array(66,)           — all concatenated, for the Critic

        Each 22-dim local observation is structured as:
          [0:8]   queue lengths per incoming lane (log-scaled, 0→1)
          [8:16]  waiting times per incoming lane (log-scaled, 0→1)
          [16:20] vehicle counts on outgoing lanes (log-scaled, 0→1)
          [20]    current phase normalised: action_idx / (n_phases - 1)  → 0.0, 0.5, or 1.0
          [21]    phase state flag: 0.0 = currently GREEN, 1.0 = currently YELLOW

        WHY LOG SCALING?
          Queue length is read as a raw integer (0, 1, 2, ..., 15+).
          log(1+x) compresses large values: log(1)=0, log(2)=0.69, log(6)=1.79, log(16)=2.77.
          Dividing by log(16) maps [0, 15] → roughly [0, 1].
          The key benefit: the difference between 0 and 1 car (urgent!) looks larger than
          the difference between 14 and 15 cars (already jammed anyway).
          Neural networks learn better when inputs are in [0, 1] and their scale matches importance.

        WHY PADDING TO FIXED SIZE?
          SUMO sometimes returns fewer lanes than expected for a TL (e.g. if a lane
          is filtered out). We always pad with 0.0 up to N_LANES_PER_TL so the
          observation vector has a fixed length every step. Fixed-size input is required
          by the neural network.
        """
        obs_dict  = {}
        all_parts = []   # will become global_state by concatenation

        for tl in TL_IDS:
            parts = []

            # ── Queue lengths in PCU (8 values) ───────────────────────────────────────
            # Measured in Passenger Car Units (PCU) for Indian heterogeneous traffic
            for lane in self._controlled_lanes[tl]:
                try:
                    h = _get_lane_pcu_halting(lane)
                    parts.append(min(float(np.log1p(h) / LOG_OBS_MAX), 1.0))
                except Exception:
                    parts.append(0.0)
            while len(parts) < N_LANES_PER_TL:
                parts.append(0.0)
            parts = parts[:N_LANES_PER_TL]   # trim if somehow more than expected

            # ── Waiting times (8 values) ───────────────────────────────────────
            # getWaitingTime(lane) = sum of time-since-last-movement for all vehicles on lane.
            # This is different from queue: a lane with 3 cars stopped for 100s each
            # has waiting time = 300, but queue = 3.
            # Both signals together help the agent distinguish "small queue, long wait"
            # (people have been stuck a while) from "big queue, short wait" (just arrived).
            wait_parts = []
            for lane in self._controlled_lanes[tl]:
                try:
                    w = traci.lane.getWaitingTime(lane)
                    wait_parts.append(min(float(np.log1p(w) / LOG_OBS_MAX_WAIT), 1.0))
                except Exception:
                    wait_parts.append(0.0)
            while len(wait_parts) < N_LANES_PER_TL:
                wait_parts.append(0.0)
            parts.extend(wait_parts[:N_LANES_PER_TL])

            # ── Outgoing lane vehicle counts in PCU (4 values) ────────────────────────
            # Downstream congestion in PCU: informs agent of physical road occupancy
            out_parts = []
            for lane in self._outgoing_lanes[tl]:
                try:
                    v = _get_lane_pcu_count(lane)
                    out_parts.append(min(float(np.log1p(v) / LOG_OBS_MAX), 1.0))
                except Exception:
                    out_parts.append(0.0)
            while len(out_parts) < N_OUT_LANES_PER_TL:
                out_parts.append(0.0)
            parts.extend(out_parts[:N_OUT_LANES_PER_TL])

            # ── Phase context (2 values) ────────────────────────────────────────
            # The agent needs to know what it's currently doing so it can reason about
            # whether to keep going or switch. Without this, it has no memory of its own action.
            n_phases = len(MAJOR_GREEN_PHASES[tl])
            parts.append(self._current_action[tl] / max(n_phases - 1, 1))   # 0.0, 0.5, or 1.0
            parts.append(0.0 if self._phase_state[tl] == "GREEN" else 1.0)  # 0=green, 1=yellow

            obs_arr          = np.array(parts, dtype=np.float32)   # (22,) local obs
            obs_dict[tl]     = obs_arr
            all_parts.extend(parts)   # accumulate for global state

        # global_state = [TL_A obs (22) | TL_B obs (22) | TL_C obs (22)] → shape (66,)
        global_state = np.array(all_parts, dtype=np.float32)
        return obs_dict, global_state

    # ─────────────────────────────────────────────────────────────
    # REWARD  (shared across all agents)
    # ─────────────────────────────────────────────────────────────
    def _compute_reward(self):
        """
        Compute the cooperative reward signal for all three agents.

        All agents receive the SAME reward every step — this is the cooperative MARL setup.
        If they all benefit together when traffic flows well, they'll learn to coordinate.
        If each agent had its own selfish reward, they might compete (e.g. TL_A dumps cars
        onto TL_B to clear its own queue, penalising TL_B).

        REWARD FORMULA:
          r = -tanh(total_wait / (n_lanes × 60s))    ← primary: penalise waiting
              - 0.1 × (total_queue / (n_lanes × 15))  ← secondary: penalise congestion
              - 0.5 × collisions                       ← safety penalty
              - 0.3 × teleports                        ← jam penalty

        clipped to [-2.0, 0.5]

        WHY tanh?
          tanh maps any value to (-1, 0]. It's bounded so extreme situations don't explode
          the gradient. The sensitivity scale (n_lanes × 60s) means:
            - average wait ≈ 60s → reward ≈ -0.76  (bad)
            - average wait ≈ 0s  → reward ≈  0.0   (ideal)
            - average wait ≈ 180s → reward ≈ -0.99  (near worst)
          The 0.5 max allows a small positive reward when conditions are excellent (very
          short queues, fast flow), which gives the agent a tangible goal to aim for.

        WHY TELEPORTS AS A PENALTY?
          SUMO "teleports" a vehicle when it's been stuck for too long and is blocking
          other traffic. Teleports indicate serious gridlock and are not realistic —
          penalising them encourages the agent to prevent the network from seizing up.
        """
        try:
            n_lanes    = sum(len(self._controlled_lanes[tl]) for tl in TL_IDS)

            # Total cumulative waiting time across all incoming lanes at all TLs
            total_wait = sum(
                traci.lane.getWaitingTime(lane)
                for tl in TL_IDS
                for lane in self._controlled_lanes[tl]
            )

            # Total halting PCUs across all controlled lanes (Indian mixed traffic)
            total_q = sum(
                _get_lane_pcu_halting(lane)
                for tl in TL_IDS
                for lane in self._controlled_lanes[tl]
            )

            # Primary: tanh penalty on normalised average waiting time
            reward  = -float(np.tanh(total_wait / max(n_lanes * 60.0, 1.0)))

            # Secondary: small additive queue penalty to discourage sustained high queues
            reward -= 0.1 * total_q / max(n_lanes * MAX_QUEUE, 1)

            # Safety: collisions are expensive (real-world equivalent: accidents)
            reward -= traci.simulation.getCollidingVehiclesNumber() * 0.5

            # Jam proxy: teleports mean a vehicle was stuck so long SUMO had to move it
            reward -= traci.simulation.getStartingTeleportNumber()  * 0.3

            return float(np.clip(reward, -2.0, 0.5))

        except Exception:
            # If TraCI call fails (e.g. SUMO crashed), return neutral reward rather than crashing
            return 0.0
