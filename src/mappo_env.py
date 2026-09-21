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

SUMO_CFG = "network/triple.sumocfg"

TL_IDS   = ["cluster_266437400_266437558", "317145283", "cluster_4162190061_4162190645"]

DELTA_T       = 3

YELLOW_DUR    = 1

MIN_GREEN     = 5

MAX_GREEN     = 40

MAX_SIM_STEPS = 500

N_LANES_PER_TL     = 8

N_OUT_LANES_PER_TL = 4

OBS_MAX_QUEUE      = 15.0

LOG_OBS_MAX        = float(np.log1p(OBS_MAX_QUEUE))

OBS_MAX_WAIT       = 300.0

LOG_OBS_MAX_WAIT   = float(np.log1p(OBS_MAX_WAIT))

MAX_QUEUE          = 15.0

IRC_PCU_WEIGHTS = {
    "two_wheeler":     0.5,
    "moto":            0.5,
    "auto_rickshaw":   1.0,
    "car_small":       1.0,
    "car_normal":      1.0,
    "car_suv":         1.2,
    "car_sport":       1.0,
    "bus_city":        3.0,
    "bus_articulated": 3.5,
    "truck_lcv":       1.5,
    "truck":           3.0,
    "trailer":         4.0,
}


def _get_lane_pcu_halting(lane: str) -> float:
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


LOCAL_OBS_DIM  = N_LANES_PER_TL + N_LANES_PER_TL + N_OUT_LANES_PER_TL + 2

GLOBAL_STATE_DIM = len(TL_IDS) * LOCAL_OBS_DIM
N_ACTIONS      = 3

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

CURRICULUM = [
    (0,  ["low", "normal"]),
    (30, ["normal", "rush_hour_am", "rush_hour_pm", "holiday"]),
    (80, None),
]


class MAPPOEnv:

    _instance_count = 0

    def __init__(self, use_gui=False, scenario_name=None, gui_delay=0):
        MAPPOEnv._instance_count += 1
        self._label        = f"mappo_{MAPPOEnv._instance_count}"
        self.use_gui       = use_gui
        self.gui_delay     = gui_delay
        self.scenario_name = scenario_name

        self._controlled_lanes = {}
        self._outgoing_lanes   = {}
        self._phase_lanes      = {}
        self._phase_state      = {}
        self._state_timer      = {}
        self._current_action   = {}
        self._pending_action   = {}
        self._green_time       = {}

        self._sim_step      = 0
        self._sumo_running  = False
        self._episode_count = 0

    def reset(self):
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
        dummy_obs   = {tl: np.zeros(LOCAL_OBS_DIM,    dtype=np.float32) for tl in TL_IDS}
        dummy_state = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)
        return dummy_obs, dummy_state

    def _reset_inner(self):
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

        if self.scenario_name:
            scenario = TrafficScenario(self.scenario_name)
        else:
            pool = None
            for threshold, scenarios in CURRICULUM:
                if self._episode_count >= threshold:
                    pool = scenarios
            scenario = (TrafficScenario.random()
                        if pool is None
                        else TrafficScenario(random.choice(pool)))
        self._episode_count += 1

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
        traci.start(sumo_cmd, label=self._label)
        traci.switch(self._label)
        self._sumo_running = True
        self._sim_step     = 0

        apply_tl_programs()

        _max_ms = 60.0 / 3.6
        for eid in traci.edge.getIDList():
            try:
                if traci.edge.getMaxSpeed(eid) > _max_ms:
                    traci.edge.setMaxSpeed(eid, _max_ms)
            except Exception:
                pass

        traffic_injector.init(scenario)

        gw = green_wave.create(TL_IDS)
        gw.bootstrap(free_flow_kmh=50.0)
        gw.enabled = True

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

            try:
                programs    = traci.trafficlight.getAllProgramLogics(tl)
                active_id   = traci.trafficlight.getProgram(tl)
                active_prog = next((p for p in programs if p.programID == active_id), programs[0])
                phase_strs  = {i: p.state for i, p in enumerate(active_prog.phases)}
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

            self._phase_state[tl]    = "GREEN"
            self._state_timer[tl]    = 0
            self._current_action[tl] = 0
            self._pending_action[tl] = 0
            self._green_time[tl]     = 0
            traci.trafficlight.setPhase(tl, MAJOR_GREEN_PHASES[tl][0])

        for _ in range(20):
            traci.simulationStep()
            self._sim_step += 1
            traffic_injector.inject(self._sim_step)

        return self._get_obs()

    def step(self, actions: dict):
        try:
            traci.switch(self._label)

            for tl in TL_IDS:
                req   = int(actions[tl])
                state = self._phase_state[tl]

                if state == "GREEN":
                    self._green_time[tl] += 1
                    want_switch = (req != self._current_action[tl]
                                   and self._green_time[tl] >= MIN_GREEN)

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
                        if best_req != cur and best_q > cur_q:
                            req, want_switch = best_req, True

                    if want_switch:
                        cur_green = MAJOR_GREEN_PHASES[tl][self._current_action[tl]]
                        yellow    = YELLOW_AFTER[tl][cur_green]
                        traci.trafficlight.setPhase(tl, yellow)
                        traci.trafficlight.setPhaseDuration(tl, 9999)
                        self._phase_state[tl]    = "YELLOW"
                        self._state_timer[tl]    = YELLOW_DUR
                        self._pending_action[tl] = req
                    else:
                        traci.trafficlight.setPhaseDuration(tl, 9999)

                elif state == "YELLOW":
                    self._state_timer[tl] -= 1
                    if self._state_timer[tl] <= 0:
                        target = MAJOR_GREEN_PHASES[tl][self._pending_action[tl]]
                        traci.trafficlight.setPhase(tl, target)
                        traci.trafficlight.setPhaseDuration(tl, 9999)
                        self._phase_state[tl]    = "GREEN"
                        self._current_action[tl] = self._pending_action[tl]
                        self._green_time[tl]     = 0

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
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

    def _get_obs(self):
        obs_dict  = {}
        all_parts = []

        for tl in TL_IDS:
            parts = []

            for lane in self._controlled_lanes[tl]:
                try:
                    h = _get_lane_pcu_halting(lane)
                    parts.append(min(float(np.log1p(h) / LOG_OBS_MAX), 1.0))
                except Exception:
                    parts.append(0.0)
            while len(parts) < N_LANES_PER_TL:
                parts.append(0.0)
            parts = parts[:N_LANES_PER_TL]

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

            n_phases = len(MAJOR_GREEN_PHASES[tl])
            parts.append(self._current_action[tl] / max(n_phases - 1, 1))
            parts.append(0.0 if self._phase_state[tl] == "GREEN" else 1.0)

            obs_arr          = np.array(parts, dtype=np.float32)
            obs_dict[tl]     = obs_arr
            all_parts.extend(parts)

        global_state = np.array(all_parts, dtype=np.float32)
        return obs_dict, global_state

    def _compute_reward(self):
        try:
            n_lanes    = sum(len(self._controlled_lanes[tl]) for tl in TL_IDS)

            total_wait = sum(
                traci.lane.getWaitingTime(lane)
                for tl in TL_IDS
                for lane in self._controlled_lanes[tl]
            )

            total_q = sum(
                _get_lane_pcu_halting(lane)
                for tl in TL_IDS
                for lane in self._controlled_lanes[tl]
            )

            reward  = -float(np.tanh(total_wait / max(n_lanes * 60.0, 1.0)))

            reward -= 0.1 * total_q / max(n_lanes * MAX_QUEUE, 1)

            reward -= traci.simulation.getCollidingVehiclesNumber() * 0.5

            reward -= traci.simulation.getStartingTeleportNumber()  * 0.3

            return float(np.clip(reward, -2.0, 0.5))

        except Exception:
            return 0.0
