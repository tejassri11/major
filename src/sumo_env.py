"""
Single-agent SUMO traffic environment for the Palapye intersection system.

Presentation summary:
  - This file connects Python reinforcement learning code to the SUMO simulator.
  - The agent chooses traffic-light phases, SUMO simulates the vehicles, and the
    environment returns observations and rewards.
  - MAPPO uses mappo_env.py for multi-agent training, but this file is useful for
    understanding the core simulation logic: reset, step, observation, and reward.
"""

import os
import random
import numpy as np
import gymnasium as gym
import traci

from traffic_scenario import TrafficScenario
import traffic_injector
import green_wave
from tl_programs import apply_tl_programs

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
# SUMO_CFG points to the SUMO configuration file that loads the road network,
# routes, traffic-light program, and simulation settings. If you change this path,
# the environment will run a different road network or fail to start if the file
# does not exist.
SUMO_CFG = "network/triple.sumocfg"

# TL_IDS are the three traffic-light IDs controlled by the agent. These IDs must
# exactly match the junction/traffic-light IDs inside the SUMO network. If one ID
# is wrong, TraCI will not be able to read or set that traffic light.
TL_IDS = ["cluster_266437400_266437558", "317145283", "cluster_4162190061_4162190645"]

# DELTA_T is the number of SUMO simulation seconds that pass after every RL
# action. With DELTA_T = 3, the agent makes a decision every 3 simulated seconds.
# Increasing this makes training faster because there are fewer decisions, but it
# also makes the controller less responsive. Decreasing it lets the agent react
# more often, but training takes longer and the policy may become noisy.
DELTA_T = 3

# YELLOW_DUR is measured in RL steps, not raw seconds. Because DELTA_T = 3,
# YELLOW_DUR = 1 means yellow lasts about 3 simulated seconds. Raising this makes
# transitions safer and more conservative, but reduces available green time.
# Lowering it can make the signal behavior unrealistic.
YELLOW_DUR = 1

# MIN_GREEN prevents traffic-light flickering. The agent cannot switch away from
# a green phase until it has been held for this many RL steps. With MIN_GREEN = 5
# and DELTA_T = 3, minimum green is about 15 seconds. Increasing it creates
# smoother signals but slower reaction. Decreasing it creates faster reaction but
# may cause unstable stop-start traffic.
MIN_GREEN = 5

# MAX_GREEN is a gridlock guard. If one phase has stayed green too long, the code
# checks whether another phase is more congested and may force a rotation. With
# MAX_GREEN = 40 and DELTA_T = 3, the limit is about 120 seconds. Increasing it
# lets one road dominate for longer. Decreasing it forces fairness more often.
MAX_GREEN = 40

# MAX_SIM_STEPS is the length of one episode in RL decisions. With 1200 steps and
# DELTA_T = 3, one episode represents about 3600 simulated seconds, or 1 hour.
# Longer episodes show long-term congestion effects but cost more time. Shorter
# episodes train faster but may miss full queue build-up and recovery.
MAX_SIM_STEPS = 1200

# These define how many lanes are represented in the observation vector. Each
# traffic light contributes incoming queue data, incoming waiting-time data,
# outgoing-lane occupancy data, and two signal-state values. If the network file
# changes and intersections have more lanes, these values and the observation
# construction must be updated together.
N_LANES_PER_TL = 8
N_OUT_LANES_PER_TL = 4

# OBS_MAX_QUEUE controls queue normalisation. Queue values are log-scaled and
# capped so the neural network receives stable inputs between 0 and 1. Increasing
# this cap lets the model distinguish very large queues, but small queues become
# less important. Decreasing it makes the model react strongly to small queues
# but saturates quickly during heavy congestion.
OBS_MAX_QUEUE = 15.0
LOG_OBS_MAX = float(np.log1p(OBS_MAX_QUEUE))  # precomputed for log scaling

# OBS_MAX_WAIT controls waiting-time normalisation. 300 seconds means that a lane
# with 5 minutes of accumulated waiting is treated as severe. Increasing it gives
# more detail for very long delays. Decreasing it makes the model more sensitive
# to shorter delays.
OBS_MAX_WAIT = 300.0                           # seconds cap for per-lane waiting time
LOG_OBS_MAX_WAIT = float(np.log1p(OBS_MAX_WAIT))

# MAX_QUEUE is used in the reward penalty. If this is lowered, queue penalties
# become stronger sooner. If it is raised, the reward becomes more tolerant of
# queues before strongly punishing the agent.
MAX_QUEUE = 15.0

# Curriculum:
#
# Training starts with easier scenarios, then introduces harder traffic patterns
# as the episode count increases. This is useful because a completely random hard
# traffic pattern from episode 1 can make learning unstable. The agent first
# learns basic phase control, then faces rush-hour and holiday traffic.
#
# If hard scenarios appear earlier, training becomes more realistic but harder.
# If they appear later, the agent learns smoothly but may need extra time to
# generalise to congestion.
CURRICULUM = [
    (0,  ["low", "normal"]),
    (30, ["normal", "rush_hour_am", "rush_hour_pm", "holiday"]),
    (80, None),  # None = full random (all scenarios)
]

# Major green phases:
#
# The agent chooses action 0, 1, or 2. This dictionary maps those action numbers
# to actual SUMO phase indices. For example, action 0 maps to SUMO phase 0.
#
# Change this only if you also understand the traffic-light program in
# network/tls.add.xml. If the phase numbers are wrong, the agent may control the
# wrong movement. If you add more actions, you must also update the action space,
# neural network output size, and yellow-phase mapping.
MAJOR_GREEN_PHASES = {
    "cluster_266437400_266437558": [0, 2, 4],
    "317145283":                 [0, 2, 4],
    "cluster_4162190061_4162190645": [0, 2, 4],
}

YELLOW_AFTER = {
    "cluster_266437400_266437558": {0: 1, 2: 3, 4: 5},
    "317145283":                 {0: 1, 2: 3, 4: 5},
    "cluster_4162190061_4162190645": {0: 1, 2: 3, 4: 5},
}


class SumoEnv(gym.Env):
    """
    Gym-style environment wrapper around SUMO.

    Main responsibilities:
      - start and reset the SUMO traffic simulation,
      - receive traffic-light actions from the RL agent,
      - advance the simulation by a few seconds,
      - measure congestion, waiting time, and safety events,
      - return a reward that tells the agent whether traffic improved.

    How to read this class:
      __init__ defines what the agent can see and do.
      reset starts a new traffic episode.
      step applies the agent's action and advances the simulator.
      _get_obs converts SUMO traffic data into neural-network inputs.
      _compute_reward converts traffic performance into a learning signal.

    For the final presentation, this class is the easiest way to explain the
    system cycle: observe traffic, choose signal phases, simulate movement,
    measure congestion, then reward or penalise the controller.
    """

    _instance_count = 0  # unique label per instance so train/eval don't collide

    def __init__(self, use_gui=False, scenario_name=None):
        super().__init__()

        SumoEnv._instance_count += 1
        self._label = f"sumo_{SumoEnv._instance_count}"

        self.use_gui = use_gui
        self.scenario_name = scenario_name

        # Observation space:
        #
        # This tells Gym/RL code the size and numeric range of the state vector.
        # Every traffic light contributes:
        #   8 incoming queue values
        #   8 incoming waiting-time values
        #   4 outgoing lane vehicle counts
        #   1 current phase value
        #   1 green/yellow state value
        #
        # For 3 traffic lights, that becomes 3 * 22 = 66 numbers. If you add more
        # intersections or more lane features, obs_size must increase and the
        # neural network input size must match it.
        obs_size = len(TL_IDS) * (N_LANES_PER_TL + N_LANES_PER_TL + N_OUT_LANES_PER_TL + 2)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Action space:
        #
        # MultiDiscrete([3, 3, 3]) means the controller chooses one action for
        # each of the 3 traffic lights. Each action can be 0, 1, or 2, matching
        # the three entries in MAJOR_GREEN_PHASES.
        #
        # If you add a fourth possible green phase, this must become [4, 4, 4],
        # and the model output layer must also produce 4 action probabilities.
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3])

        # Runtime state:
        #
        # These dictionaries are filled during reset because we need SUMO running
        # before we can ask it which lanes belong to each traffic light. They
        # track the current phase, pending phase, yellow timers, and green time.
        # In simple terms, this is the memory that makes the traffic lights obey
        # realistic timing rules instead of switching instantly.
        self._controlled_lanes = {}
        self._outgoing_lanes = {}
        self._phase_lanes = {}    # action_idx → list of incoming lanes served by that phase
        self._phase_state = {}    # "GREEN" or "YELLOW"
        self._state_timer = {}    # steps remaining in yellow
        self._current_action = {}
        self._pending_action = {}
        self._green_time = {}     # steps held at current green

        self._sim_step = 0
        self._sumo_running = False
        self._episode_count = 0   # curriculum progression

    # ─────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        """
        Start a fresh episode and return the first observation.

        In reinforcement learning, an episode is one complete simulation run.
        Reset closes any old SUMO process, chooses a traffic scenario, starts
        SUMO again, initialises lanes and signals, warms up the network with
        vehicles, and finally returns the initial traffic state.

        Changing reset behavior affects the starting conditions of training. For
        example, a longer warm-up produces more realistic initial traffic, but it
        also makes every episode slower to start.
        """
        super().reset(seed=seed)

        # Retry loop: if SUMO fails to start, try up to 3 times before giving up
        for _attempt in range(3):
            try:
                return self._reset_inner()
            except Exception as e:
                print(f"[ENV {self._label}] reset() attempt {_attempt+1} failed: {e}")
                try:
                    traci.switch(self._label)
                    traci.close()
                except Exception:
                    pass
                self._sumo_running = False
                if _attempt == 2:
                    raise

        # unreachable, but satisfies type checkers
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def _reset_inner(self):
        # Close any previous SUMO instance before starting a clean episode.
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

        # Scenario selection:
        #
        # If scenario_name is supplied, every episode uses that fixed scenario.
        # This is useful for evaluation because it keeps tests consistent.
        #
        # If scenario_name is None, training uses the curriculum above. That
        # means the environment gradually introduces harder traffic patterns.
        if self.scenario_name:
            scenario = TrafficScenario(self.scenario_name)
        else:
            # Curriculum: pick from an increasingly complex pool as episodes accumulate
            pool = None
            for threshold, scenarios in CURRICULUM:
                if self._episode_count >= threshold:
                    pool = scenarios
            scenario = TrafficScenario.random() if pool is None else TrafficScenario(random.choice(pool))
        self._episode_count += 1

        # Start SUMO in either GUI mode for demos or command-line mode for fast training.
        _sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
        _bin_name  = "sumo-gui.exe" if self.use_gui else "sumo.exe"
        binary     = os.path.join(_sumo_home, "bin", _bin_name)
        traci.start([binary, "-c", SUMO_CFG, "--no-warnings", "--no-step-log"],
                    label=self._label)
        traci.switch(self._label)

        self._sumo_running = True
        self._sim_step = 0

        # Load the traffic-light phase program used by the RL actions.
        apply_tl_programs()

        # Cap road speeds to a realistic urban range so the simulation matches
        # the project setting more closely.
        _max_ms = 60.0 / 3.6
        for _eid in traci.edge.getIDList():
            try:
                if traci.edge.getMaxSpeed(_eid) > _max_ms:
                    traci.edge.setMaxSpeed(_eid, _max_ms)
            except Exception:
                pass

        # Prepare the selected traffic scenario, then enable green-wave support.
        traffic_injector.init(scenario)

        gw = green_wave.create(TL_IDS)
        gw.bootstrap(free_flow_kmh=50.0)
        gw.enabled = True

        # Lane discovery:
        #
        # SUMO knows which lanes are controlled by each traffic light. The code
        # asks TraCI for those lanes and stores them so observation and reward can
        # later measure queues, waiting time, and outgoing-lane occupancy.
        for tl in TL_IDS:
            self._controlled_lanes[tl] = list(
                dict.fromkeys(traci.trafficlight.getControlledLanes(tl))
            )
            links_tl = traci.trafficlight.getControlledLinks(tl)
            out = []
            for link_group in links_tl:
                for (_, to_lane, _) in link_group:
                    if to_lane and to_lane not in out:
                        out.append(to_lane)
            self._outgoing_lanes[tl] = out

            # Action-to-lane mapping:
            #
            # For the gridlock guard, the environment needs to know which lanes
            # each green phase serves. It reads the SUMO traffic-light phase
            # strings and records the lanes that receive a green signal for each
            # action. Later, if one phase has been green too long, the environment
            # can compare queues on alternative phases and rotate intelligently.
            try:
                programs = traci.trafficlight.getAllProgramLogics(tl)
                active_id = traci.trafficlight.getProgram(tl)
                active_prog = next((p for p in programs if p.programID == active_id), programs[0])
                phase_strs = {i: p.state for i, p in enumerate(active_prog.phases)}
            except Exception:
                phase_strs = {}
            self._phase_lanes[tl] = {}
            for act_idx, ph_idx in enumerate(MAJOR_GREEN_PHASES[tl]):
                state_str = phase_strs.get(ph_idx, "")
                served = []
                for li, ch in enumerate(state_str):
                    if ch in ('G', 'g') and li < len(links_tl):
                        for (from_lane, _, _) in links_tl[li]:
                            if from_lane and from_lane not in served:
                                served.append(from_lane)
                self._phase_lanes[tl][act_idx] = served

            # Start every light at action 0. Changing this would change the
            # initial traffic pattern at the beginning of every episode.
            self._phase_state[tl] = "GREEN"
            self._state_timer[tl] = 0
            self._current_action[tl] = 0
            self._pending_action[tl] = 0
            self._green_time[tl] = 0
            traci.trafficlight.setPhase(tl, MAJOR_GREEN_PHASES[tl][0])

        # Warm-up period: add initial vehicles before the agent starts controlling.
        for _ in range(20):
            traci.simulationStep()
            self._sim_step += 1
            traffic_injector.inject(self._sim_step)

        return self._get_obs(), {}

    # ─────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Apply one RL action and advance the SUMO simulation.

        The agent proposes one green phase per traffic light. This method enforces
        realistic signal behavior: minimum green time, yellow transitions, and a
        maximum-green guard to reduce gridlock.

        What changing values affects here:
          MIN_GREEN changes how soon a requested switch is allowed.
          YELLOW_DUR changes how long the signal stays yellow before new green.
          MAX_GREEN changes when the environment may force a phase rotation.
          DELTA_T changes how much traffic movement happens after each action.
        """
        try:
            traci.switch(self._label)
            for i, tl in enumerate(TL_IDS):
                req = int(action[i])
                state = self._phase_state[tl]

                if state == "GREEN":
                    self._green_time[tl] += 1

                    # The agent may request a different green phase, but the
                    # environment only allows the switch after MIN_GREEN has
                    # passed. This prevents unrealistic flickering where signals
                    # change every few seconds.
                    want_switch = req != self._current_action[tl] and self._green_time[tl] >= MIN_GREEN

                    # Gridlock guard:
                    #
                    # If the current green has lasted too long, the environment
                    # checks the queue on the current phase and compares it to
                    # queues on the other phases. If another phase has a larger
                    # queue, the environment overrides the requested action and
                    # rotates to that more congested movement.
                    #
                    # This is not the main learning algorithm; it is a safety
                    # rule that stops the simulation from being trapped in one
                    # green direction forever.
                    if not want_switch and self._green_time[tl] >= MAX_GREEN:
                        cur = self._current_action[tl]
                        cur_lanes = self._phase_lanes.get(tl, {}).get(cur, [])
                        cur_q = sum(traci.lane.getLastStepHaltingNumber(l) for l in cur_lanes) if cur_lanes else 0
                        best_req, best_q = cur, -1
                        for alt in range(len(MAJOR_GREEN_PHASES[tl])):
                            if alt == cur:
                                continue
                            alt_lanes = self._phase_lanes.get(tl, {}).get(alt, [])
                            alt_q = (
                                sum(traci.lane.getLastStepHaltingNumber(l) for l in alt_lanes)
                                if alt_lanes else 0
                            )
                            if alt_q > best_q:
                                best_q, best_req = alt_q, alt
                        if best_req != cur and best_q > cur_q:
                            req = best_req
                            want_switch = True

                    if want_switch:
                        # A valid phase switch always goes through yellow first.
                        # We store the requested green phase in _pending_action,
                        # enter the yellow phase now, and complete the switch
                        # later when the yellow timer reaches zero.
                        cur_green = MAJOR_GREEN_PHASES[tl][self._current_action[tl]]
                        yellow = YELLOW_AFTER[tl][cur_green]
                        traci.trafficlight.setPhase(tl, yellow)
                        traci.trafficlight.setPhaseDuration(tl, 9999)
                        self._phase_state[tl] = "YELLOW"
                        self._state_timer[tl] = YELLOW_DUR
                        self._pending_action[tl] = req
                    else:
                        # Keeping duration high prevents SUMO's built-in timer
                        # from changing the phase automatically. We want the RL
                        # environment, not SUMO's default schedule, to control
                        # when phase changes happen.
                        traci.trafficlight.setPhaseDuration(tl, 9999)

                elif state == "YELLOW":
                    # During yellow, the agent cannot choose another green yet.
                    # The environment counts down the yellow timer, then commits
                    # to the pending green phase.
                    self._state_timer[tl] -= 1
                    if self._state_timer[tl] <= 0:
                        target = MAJOR_GREEN_PHASES[tl][self._pending_action[tl]]
                        traci.trafficlight.setPhase(tl, target)
                        traci.trafficlight.setPhaseDuration(tl, 9999)
                        self._phase_state[tl] = "GREEN"
                        self._current_action[tl] = self._pending_action[tl]
                        self._green_time[tl] = 0

            # Advance the microscopic traffic simulation after applying signals.
            #
            # During these DELTA_T seconds, vehicles accelerate, stop, queue, and
            # pass through intersections according to the selected phases.
            for _ in range(DELTA_T):
                traci.simulationStep()
                self._sim_step += 1
                traffic_injector.inject(self._sim_step)

            # New observation and reward are measured after the traffic has moved.
            # This is important: the reward reflects the result of the action,
            # not the state before the action was applied.
            obs = self._get_obs()
            reward = self._compute_reward()
            terminated = self._sim_step >= MAX_SIM_STEPS

            if terminated:
                try:
                    traci.close()
                except Exception:
                    pass
                self._sumo_running = False

            return obs, reward, terminated, False, {}

        except Exception as e:
            print(f"[ENV {self._label}] step() crashed: {e} — ending episode early")
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False
            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return dummy_obs, 0.0, True, False, {}

    def close(self):
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

    # ─────────────────────────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────────────────────────
    def _get_obs(self):
        """
        Build the numeric state seen by the RL agent.

        Values are normalised into roughly [0, 1] so the neural network receives
        stable inputs instead of raw queue lengths or waiting times with large scale.

        For each traffic light, the observation contains:
          1. queue lengths on incoming lanes,
          2. waiting times on incoming lanes,
          3. vehicle counts on outgoing lanes,
          4. the currently active green phase,
          5. whether the signal is green or yellow.

        Why outgoing lanes matter:
          A traffic light should not only know where vehicles are waiting. It
          should also know whether there is space after the intersection. If the
          outgoing road is full, giving green may push vehicles into a blocked
          area and worsen congestion.

        What changing this section would do:
          Adding features can give the agent more information, but the neural
          network input size must be changed to match. Removing features makes
          training simpler but may hide important traffic conditions.
        """
        parts = []
        for tl in TL_IDS:
            # Incoming: halting vehicles per controlled lane (log-scaled for better low-queue resolution)
            vals = []
            for lane in self._controlled_lanes[tl]:
                try:
                    h = traci.lane.getLastStepHaltingNumber(lane)
                    vals.append(min(float(np.log1p(h) / LOG_OBS_MAX), 1.0))
                except Exception:
                    vals.append(0.0)
            while len(vals) < N_LANES_PER_TL:
                vals.append(0.0)
            parts.extend(vals[:N_LANES_PER_TL])

            # Waiting time: cumulative per-lane waiting time (log-scaled)
            wait_vals = []
            for lane in self._controlled_lanes[tl]:
                try:
                    w = traci.lane.getWaitingTime(lane)
                    wait_vals.append(min(float(np.log1p(w) / LOG_OBS_MAX_WAIT), 1.0))
                except Exception:
                    wait_vals.append(0.0)
            while len(wait_vals) < N_LANES_PER_TL:
                wait_vals.append(0.0)
            parts.extend(wait_vals[:N_LANES_PER_TL])

            # Outgoing: vehicle count per exit lane (log-scaled)
            out_vals = []
            for lane in self._outgoing_lanes[tl]:
                try:
                    v = traci.lane.getLastStepVehicleNumber(lane)
                    out_vals.append(min(float(np.log1p(v) / LOG_OBS_MAX), 1.0))
                except Exception:
                    out_vals.append(0.0)
            while len(out_vals) < N_OUT_LANES_PER_TL:
                out_vals.append(0.0)
            parts.extend(out_vals[:N_OUT_LANES_PER_TL])

            n_phases = len(MAJOR_GREEN_PHASES[tl])
            parts.append(self._current_action[tl] / max(n_phases - 1, 1))
            parts.append(0.0 if self._phase_state[tl] == "GREEN" else 1.0)
        return np.array(parts, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────
    # REWARD
    # ─────────────────────────────────────────────────────────────
    def _compute_reward(self):
        """
        Reward function used for learning.

        Higher reward means better traffic flow. The reward penalises long waiting
        times, large queues, collisions, and teleports. This guides the agent toward
        reducing congestion while keeping the simulation safe.

        The reward is mostly negative because congestion is a cost. A value closer
        to zero means better performance. A very negative value means the network
        caused heavy waiting, large queues, or safety problems.

        Main terms:
          total_wait penalty: teaches the agent to reduce vehicle delay.
          total_q penalty: prevents the agent from ignoring queue length.
          collision penalty: strongly discourages unsafe behavior.
          teleport penalty: discourages severe gridlock where SUMO removes stuck cars.

        What changing this section would do:
          Increasing the waiting-time penalty makes the agent focus more on delay.
          Increasing the queue penalty makes it focus more on clearing visible queues.
          Increasing safety penalties makes the controller more conservative.
          Poor reward weights can make the agent learn the wrong behavior, so this
          section is one of the most important parts of the project.
        """
        try:
            n_lanes = sum(len(self._controlled_lanes[tl]) for tl in TL_IDS)

            # Cumulative waiting time across all controlled lanes (primary metric)
            total_wait = sum(
                traci.lane.getWaitingTime(lane)
                for tl in TL_IDS
                for lane in self._controlled_lanes[tl]
            )

            # Total halting vehicles (absolute gridlock guard)
            total_q = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for tl in TL_IDS
                for lane in self._controlled_lanes[tl]
            )

            # tanh on average waiting time per lane keeps reward in (-1, 0]; 60 s is the sensitivity scale
            reward = -float(np.tanh(total_wait / max(n_lanes * 60.0, 1.0)))

            # Absolute queue penalty (prevents coasting at sustained high queue)
            reward -= 0.1 * total_q / max(n_lanes * MAX_QUEUE, 1)

            # Safety penalties
            reward -= traci.simulation.getCollidingVehiclesNumber() * 0.5
            reward -= traci.simulation.getStartingTeleportNumber() * 0.3

            return float(np.clip(reward, -2.0, 0.5))

        except Exception:
            return 0.0
