import math
import traci

_instance: "GreenWave | None" = None


def create(tl_ids: list[str]) -> "GreenWave":
    global _instance
    _instance = GreenWave(tl_ids)
    return _instance


def get() -> "GreenWave | None":
    return _instance


class GreenWave:
    def __init__(self, tl_ids: list[str]):
        self.tl_ids = tl_ids
        self.enabled = False

        self._offsets: dict[str, float] = {}
        self._cycle_length: float = 0.0
        self._junction_pos: dict[str, tuple] = {}
        self._incoming_lanes: dict[str, list[str]] = {}
        self._last_cycle: int = -1


    def bootstrap(self, free_flow_kmh: float = 50.0):
        speed_ms = free_flow_kmh / 3.6

        for tl_id in self.tl_ids:
            try:
                self._junction_pos[tl_id] = traci.junction.getPosition(tl_id)
            except Exception:
                self._junction_pos[tl_id] = (0.0, 0.0)

            try:
                self._incoming_lanes[tl_id] = list(
                    dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id))
                )
            except Exception:
                self._incoming_lanes[tl_id] = []

        try:
            programs = traci.trafficlight.getAllProgramLogics(self.tl_ids[0])
            self._cycle_length = float(sum(p.duration for p in programs[0].phases))
        except Exception:
            self._cycle_length = 90.0

        self._offsets[self.tl_ids[0]] = 0.0
        cumulative = 0.0
        for i in range(1, len(self.tl_ids)):
            pa = self._junction_pos[self.tl_ids[i - 1]]
            pb = self._junction_pos[self.tl_ids[i]]
            dist = math.dist(pa, pb)
            cumulative += dist / speed_ms
            self._offsets[self.tl_ids[i]] = cumulative % self._cycle_length

        print(
            f"[GREEN WAVE] Bootstrap | speed={free_flow_kmh:.0f} km/h | "
            f"cycle={self._cycle_length:.1f}s | offsets={self._fmt()}"
        )

        if self.enabled:
            self._apply_all()

    def update(self, step: int):
        if not self.enabled or self._cycle_length == 0:
            return

        cycle_idx = int(step) // max(1, int(self._cycle_length))
        if cycle_idx == self._last_cycle:
            return
        self._last_cycle = cycle_idx

        any_updated = False
        cumulative = 0.0

        for i in range(1, len(self.tl_ids)):
            tl_a = self.tl_ids[i - 1]
            tl_b = self.tl_ids[i]

            travel_time = self._measure_travel_time(tl_a, tl_b)
            if travel_time is None:
                continue

            cumulative += travel_time
            new_offset = cumulative % self._cycle_length
            old_offset = self._offsets.get(tl_b, new_offset)

            blended = 0.7 * old_offset + 0.3 * new_offset

            if abs(blended - old_offset) > 0.5:
                self._offsets[tl_b] = blended
                any_updated = True
                print(
                    f"[GREEN WAVE] Cycle {cycle_idx} | {tl_a}→{tl_b} | "
                    f"travel={travel_time:.1f}s → offset={blended:.1f}s"
                )

        if any_updated:
            self._apply_downstream()

    def set_speed(self, kmh: float):
        self.bootstrap(kmh)


    def _measure_travel_time(self, tl_a: str, tl_b: str) -> "float | None":
        lanes = self._incoming_lanes.get(tl_b, [])
        speeds, lengths = [], []

        for lane_id in lanes:
            try:
                spd = traci.lane.getLastStepMeanSpeed(lane_id)
                ln  = traci.lane.getLength(lane_id)
                if spd > 0.5:
                    speeds.append(spd)
                    lengths.append(ln)
            except Exception:
                pass

        if not speeds:
            return None

        total_len = sum(lengths)
        avg_speed = sum(s * l for s, l in zip(speeds, lengths)) / total_len

        pa = self._junction_pos[tl_a]
        pb = self._junction_pos[tl_b]
        dist = math.dist(pa, pb)

        return dist / avg_speed

    def _apply_all(self):
        for tl_id in self.tl_ids:
            self._apply_offset(tl_id, self._offsets.get(tl_id, 0.0))

    def _apply_downstream(self):
        for tl_id in self.tl_ids[1:]:
            self._apply_offset(tl_id, self._offsets.get(tl_id, 0.0))

    def _apply_offset(self, tl_id: str, offset_s: float):
        if offset_s <= 0:
            return
        try:
            programs = traci.trafficlight.getAllProgramLogics(tl_id)
            if not programs:
                return
            phases = programs[0].phases
            target = offset_s % self._cycle_length
            cumulative = 0.0
            for idx, phase in enumerate(phases):
                if cumulative + phase.duration >= target:
                    traci.trafficlight.setPhase(tl_id, idx)
                    time_into = target - cumulative
                    remaining = max(1.0, phase.duration - time_into)
                    traci.trafficlight.setPhaseDuration(tl_id, remaining)
                    return
                cumulative += phase.duration
        except Exception as e:
            print(f"[GREEN WAVE] Offset apply failed for {tl_id}: {e}")

    def _fmt(self) -> str:
        return " | ".join(f"{k}: {v:.1f}s" for k, v in self._offsets.items())


    @property
    def offsets(self) -> dict[str, float]:
        return dict(self._offsets)

    @property
    def cycle_length(self) -> float:
        return self._cycle_length
