"""
Green-wave (progressive signal coordination) for a corridor of traffic lights.

How it works
------------
1. Bootstrap (geometry-based):
   At simulation start, compute each downstream TL's offset from the inter-junction
   distance and an assumed free-flow speed.  Apply immediately so the corridor is
   already coordinated before the first vehicle arrives.

2. Dynamic correction (measurement-based):
   Once per signal cycle, measure actual mean speed on each TL's incoming lanes.
   Those are the lanes vehicles travel on between the two signals — exactly what a
   real stop-line or advance detector measures.  Blend old offset 70 % / new 30 %
   to avoid oscillation, then re-apply only if the change exceeds 0.5 s.

Usage (from run_sumo)
---------------------
    gw = green_wave.create(TL_IDS)
    gw.bootstrap(free_flow_kmh=50.0)   # call AFTER traci.start()
    gw.enabled = True

    # inside simulation loop:
    gw.update(step)
"""

import math
import traci

# ── module-level singleton so the dashboard thread can reach it ───────────────
_instance: "GreenWave | None" = None


def create(tl_ids: list[str]) -> "GreenWave":
    global _instance
    _instance = GreenWave(tl_ids)
    return _instance


def get() -> "GreenWave | None":
    return _instance


# ─────────────────────────────────────────────────────────────────────────────

class GreenWave:
    def __init__(self, tl_ids: list[str]):
        self.tl_ids = tl_ids
        self.enabled = False

        self._offsets: dict[str, float] = {}      # {tl_id: seconds into cycle}
        self._cycle_length: float = 0.0
        self._junction_pos: dict[str, tuple] = {} # {tl_id: (x, y)}
        self._incoming_lanes: dict[str, list[str]] = {}  # {tl_id: [lane_id, …]}
        self._last_cycle: int = -1

    # ── public API ────────────────────────────────────────────────────────────

    def bootstrap(self, free_flow_kmh: float = 50.0):
        """
        Compute geometry-based offsets and optionally apply them.
        Safe to call again with a new speed (re-calibrates).
        Must be called after traci.start().
        """
        speed_ms = free_flow_kmh / 3.6

        # Cache positions and incoming lanes for every TL
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

        # Read cycle length from the first TL's active program
        try:
            programs = traci.trafficlight.getAllProgramLogics(self.tl_ids[0])
            self._cycle_length = float(sum(p.duration for p in programs[0].phases))
        except Exception:
            self._cycle_length = 90.0  # reasonable urban default

        # Reference TL has offset 0; each downstream TL gets cumulative travel time
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
        """
        Call every simulation step.
        Once per cycle, re-measures actual travel times and nudges offsets.
        No-op if disabled or no vehicles are moving yet.
        """
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
                continue  # no traffic yet — keep bootstrap value

            cumulative += travel_time
            new_offset = cumulative % self._cycle_length
            old_offset = self._offsets.get(tl_b, new_offset)

            # Smooth blend to suppress cycle-to-cycle jitter
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
        """Re-calibrate using a new free-flow speed assumption."""
        self.bootstrap(kmh)

    # ── internal ──────────────────────────────────────────────────────────────

    def _measure_travel_time(self, tl_a: str, tl_b: str) -> "float | None":
        """
        Estimate travel time A→B.

        Uses mean speed on TL_B's incoming lanes (the lanes vehicles are on
        between the two signals), weighted by lane length.  Falls back to
        junction-to-junction distance / speed if no traffic is detected.
        """
        lanes = self._incoming_lanes.get(tl_b, [])
        speeds, lengths = [], []

        for lane_id in lanes:
            try:
                spd = traci.lane.getLastStepMeanSpeed(lane_id)
                ln  = traci.lane.getLength(lane_id)
                if spd > 0.5:          # only lanes with moving vehicles
                    speeds.append(spd)
                    lengths.append(ln)
            except Exception:
                pass

        if not speeds:
            return None  # no data — keep existing offset

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
        for tl_id in self.tl_ids[1:]:   # reference TL (index 0) is never shifted
            self._apply_offset(tl_id, self._offsets.get(tl_id, 0.0))

    def _apply_offset(self, tl_id: str, offset_s: float):
        """
        Shift a TL into the phase that corresponds to 'offset_s' seconds
        into its cycle.  This implements the green-wave timing offset without
        touching the underlying program logic.
        """
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

    # ── properties for the dashboard ─────────────────────────────────────────

    @property
    def offsets(self) -> dict[str, float]:
        return dict(self._offsets)

    @property
    def cycle_length(self) -> float:
        return self._cycle_length
