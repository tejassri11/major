"""
Vehicle injection module — adds traffic to the SUMO simulation each step.

WHY A SEPARATE INJECTOR?
  SUMO can load vehicles from a pre-built .rou.xml route file, but that gives fixed traffic.
  We need DYNAMIC traffic that changes with the scenario (rush hour, low traffic, holiday).
  This module reads a reference route file to learn valid OD (origin-destination) pairs,
  then injects individual vehicles step-by-step at a rate controlled by the scenario.

HOW IT WORKS PER EPISODE:
  1. init()    — called once after SUMO starts; validates OD pairs against the live network
  2. inject()  — called every simulation step; decides whether to spawn a vehicle this step
  3. The scenario's demand_rate() controls the probability of injection each step

OD PAIRS (Origin → Destination):
  An OD pair is (edge_A, edge_B) meaning a vehicle starts on edge_A and drives to edge_B.
  SUMO's findRoute() computes the actual path. We validate all pairs at episode start so
  we never try to route a vehicle on a path that doesn't exist.
"""

import xml.etree.ElementTree as ET
import random
import traci
from traffic_scenario import (
    TrafficScenario,
    HIGHWAY_ENTRIES, LOCAL_DESTS,
    LOCAL_ORIGINS, HIGHWAY_EXITS,
    TRANSIT_ENTRIES, TRANSIT_EXITS,
)

# Path to the reference flow file. We parse this XML to get valid OD pairs —
# the actual flows defined here don't run; we use the edges as a source of truth.
FLOW_FILE = "network/triple_routes_flows.rou.xml"

# Vehicle types available in the simulation (Indian mixed traffic corridor).
TYPES = [
    "two_wheeler",    # Indian motorcycles/scooters (50%)
    "moto",           # Alternate motorcycle model (10%)
    "auto_rickshaw",  # 3-Wheeler auto-rickshaw (15%)
    "car_small",      # Hatchbacks (12%)
    "car_normal",     # Sedans (5%)
    "car_suv",        # SUVs (3%)
    "bus_city",       # City transit buses (3%)
    "truck_lcv",      # Light commercial delivery trucks (2%)
]

# Indian arterial modal split weights
INDIAN_MODAL_WEIGHTS = [0.50, 0.10, 0.15, 0.12, 0.05, 0.03, 0.03, 0.02]

# Per-scenario origin spawn reduction. When a spawn is drawn from a listed origin
# edge under the named scenario, it is suppressed with the given probability —
# e.g. 0.40 means 40% of vehicles from that origin are dropped. Used to relieve
# localized gridlock on a specific approach without changing overall demand.
_ORIGIN_REDUCTION: dict[str, dict[str, float]] = {
    "holiday": {"1080437656#0": 0.40},
}

# ── Module-level state ─────────────────────────────────────────────────────────
# These are module globals — effectively singleton state for the injector.
# All are reset in init() at the start of each episode.

# Full list of validated OD pairs (confirmed to exist in the live SUMO network).
# Built once at init(), never changes mid-episode.
_all_od_pairs: list[tuple[str, str]] = []

# Directional subsets — built from _all_od_pairs when the scenario changes.
# They're pre-filtered so inject() can quickly pick a random pair for the active direction.
_inbound_pairs: list[tuple[str, str]] = []   # highway entry  → local destination
_outbound_pairs: list[tuple[str, str]] = []  # local origin   → highway exit
_transit_pairs: list[tuple[str, str]] = []   # through-traffic (enter and leave the study area)

# The active pool used by inject() each step. Which subset depends on the scenario's direction_mode.
_active_pairs: list[tuple[str, str]] = []

_scenario: TrafficScenario | None = None

# Counter for unique vehicle IDs. CRITICAL: this must always increment, even on failed injections.
# The original bug was incrementing only on success, which caused the same ID to be reused,
# making SUMO reject the vehicle as "already exists" and creating an infinite retry loop.
_veh_count = 0

# Step counter local to each scenario (resets when scenario changes).
# Passed to scenario.demand_rate() — some scenarios have time-varying demand curves
# (e.g., rush hour peaks, then tapers off).
_scenario_step = 0

# Log of every successfully injected vehicle this episode. Useful for post-episode analysis.
injection_log: list[dict] = []


def init(scenario: TrafficScenario | None = None):
    """
    Initialise the injector for a new episode. Must be called after traci.start().

    Steps:
      1. Reset vehicle counter, step counter, and injection log
      2. Parse the reference flow file to get candidate OD pairs
      3. Deduplicate the pairs (the flow file may repeat some)
      4. Validate each pair against the LIVE SUMO network (edges must exist, route must be findable)
      5. Build the directional subsets and apply the scenario filter

    Args:
      scenario: which traffic scenario to use. If None, defaults to "normal".
                Scenarios control demand rate and the directional mix of traffic.
    """
    global _all_od_pairs, _veh_count, _scenario_step, injection_log

    _veh_count = 0
    _scenario_step = 0
    injection_log.clear()

    if scenario is None:
        scenario = TrafficScenario("normal")

    # ── Parse reference OD pairs ──────────────────────────────────────────────
    # The flow file has <flow from="edgeA" to="edgeB" .../> entries.
    # We extract just the (from, to) edges — the actual flow counts are ignored.
    raw: list[tuple[str, str]] = []
    try:
        root = ET.parse(FLOW_FILE).getroot()
        for flow in root.findall("flow"):
            frm, to = flow.get("from"), flow.get("to")
            if frm and to:
                raw.append((frm, to))
    except Exception as e:
        print(f"[INJECTOR] Could not parse {FLOW_FILE}: {e}")
        return

    # ── Deduplicate while preserving order ────────────────────────────────────
    # A set would deduplicate but lose the original order. This pattern uses a set
    # as a "seen" tracker while the list comprehension preserves order.
    seen: set[tuple[str, str]] = set()
    unique_raw = [p for p in raw if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]

    # ── Validate against live network ─────────────────────────────────────────
    # The network file is authoritative. We check two things per OD pair:
    #   a) Both edges actually exist in the SUMO network (some edges may be trimmed at build time)
    #   b) SUMO can actually find a route between them (not all edge pairs are connected)
    live_edges = set(traci.edge.getIDList())
    valid: list[tuple[str, str]] = []
    skipped_edge = skipped_route = 0

    for frm, to in unique_raw:
        if frm not in live_edges or to not in live_edges:
            skipped_edge += 1
            continue
        r = traci.simulation.findRoute(frm, to)
        if r and r.edges:
            valid.append((frm, to))
        else:
            skipped_route += 1

    _all_od_pairs = valid
    print(
        f"[INJECTOR] {len(_all_od_pairs)} valid OD pairs "
        f"(dropped {skipped_edge} bad-edge, {skipped_route} no-route)"
    )

    _apply_scenario(scenario)


def set_scenario(scenario: TrafficScenario):
    """
    Hot-swap the traffic scenario without restarting SUMO.
    Resets the scenario step counter so demand curves restart from t=0 for the new scenario.
    """
    global _scenario_step
    _scenario_step = 0
    _apply_scenario(scenario)


def _apply_scenario(scenario: TrafficScenario):
    """
    Build the directional subsets and select the active injection pool for the scenario.

    DIRECTION MODES (from TrafficScenario.direction_mode):
      "inbound"  — mostly people driving INTO town (morning commute inward)
      "outbound" — mostly people driving OUT of town (evening commute outward)
      "transit"  — through-traffic passing through without stopping
      "mixed"    — any direction (all valid OD pairs)

    blocked_origins: some scenarios block certain origin edges (e.g. road works, events).
    """
    global _inbound_pairs, _outbound_pairs, _transit_pairs, _active_pairs, _scenario
    _scenario = scenario

    # Remove blocked origins from the available pool
    blocked   = scenario.blocked_origins
    available = [(o, d) for o, d in _all_od_pairs if o not in blocked]

    # Build directional subsets using the geographic constants from traffic_scenario.py:
    #   HIGHWAY_ENTRIES = edges where vehicles come in from the main highway (A1 road)
    #   LOCAL_DESTS     = edges inside town (local destinations)
    #   LOCAL_ORIGINS   = edges inside town (local trip origins)
    #   HIGHWAY_EXITS   = edges going out to the main highway
    #   TRANSIT_ENTRIES/EXITS = edges for through-traffic flow
    _inbound_pairs  = [(o, d) for o, d in available if o in HIGHWAY_ENTRIES and d in LOCAL_DESTS]
    _outbound_pairs = [(o, d) for o, d in available if o in LOCAL_ORIGINS   and d in HIGHWAY_EXITS]
    _transit_pairs  = [(o, d) for o, d in available
                       if (o in TRANSIT_ENTRIES and d in TRANSIT_EXITS)
                       or (o in TRANSIT_EXITS   and d in HIGHWAY_EXITS)]

    mode = scenario.direction_mode
    if mode == "inbound":
        pool = _inbound_pairs
    elif mode == "outbound":
        pool = _outbound_pairs
    elif mode == "transit":
        pool = _transit_pairs
    else:
        pool = available   # "mixed" — any direction

    # Fallback: if the directional pool happens to be empty (e.g. no valid inbound pairs),
    # fall back to ALL available pairs so injection never stalls.
    _active_pairs = pool if pool else available

    print(
        f"[INJECTOR] '{scenario.name}' | mode={mode} | "
        f"pool={len(_active_pairs)} pairs "
        f"(inbound={len(_inbound_pairs)}, outbound={len(_outbound_pairs)}, "
        f"transit={len(_transit_pairs)}, blocked={len(_all_od_pairs)-len(available)})"
    )


def inject(step: int):
    """
    Inject vehicles into the simulation at the current step.

    The scenario's demand_rate() returns the EXPECTED number of vehicles for this
    step (not a capped probability). We spawn the whole part deterministically and
    one extra with probability equal to the fractional part. This means a rate of
    2.4 spawns 2 vehicles every step plus a third 40% of the time — so high-demand
    scenarios (rate > 1.0) actually produce dense traffic instead of being silently
    capped at one vehicle per step.
    """
    global _scenario_step

    if not _active_pairs or _scenario is None:
        return

    # Expected vehicles this step, then advance the per-scenario step counter.
    rate = _scenario.demand_rate(_scenario_step)
    _scenario_step += 1

    # whole part = guaranteed spawns; fractional part = one extra spawn by chance
    n = int(rate) + (1 if random.random() < (rate - int(rate)) else 0)
    for _ in range(n):
        _spawn_one(step)


def _spawn_one(step: int):
    """
    Spawn a single vehicle on a random OD pair from the active pool.

    _veh_count is incremented BEFORE the try block so every attempt uses a unique ID.
    A failed injection (e.g. SUMO rejects the vehicle for being too close to another)
    would otherwise retry with the same route_id and raise "route already exists"
    errors forever — an infinite retry loop. Always incrementing guarantees unique IDs.
    """
    global _veh_count

    # Pick a random valid OD pair from the active pool
    origin, dest = random.choice(_active_pairs)

    # Per-scenario origin throttle: drop this spawn with the configured probability
    # to reduce demand from a specific approach (e.g. -465932558#2_C in holiday).
    reduction = _ORIGIN_REDUCTION.get(_scenario.name, {}).get(origin, 0.0)
    if reduction and random.random() < reduction:
        return   # suppressed — keeps this origin from overloading the junction

    # Re-find the route dynamically each step (vehicles may have changed network conditions)
    route_obj = traci.simulation.findRoute(origin, dest)
    if not route_obj or not route_obj.edges:
        return   # no route found this step — skip silently

    veh_id   = f"veh_{_veh_count}"
    route_id = f"route_{veh_id}"
    vtype    = random.choices(TYPES, weights=INDIAN_MODAL_WEIGHTS, k=1)[0]

    # INCREMENT FIRST — always, regardless of whether the TraCI calls succeed below.
    _veh_count += 1

    try:
        # Add the route (list of edge IDs) to SUMO's route registry
        traci.route.add(route_id, route_obj.edges)
        # Add the vehicle to the simulation using that route
        traci.vehicle.add(
            vehID=veh_id,
            routeID=route_id,
            typeID=vtype,
            depart="now",   # spawn immediately at the current simulation time
        )
        injection_log.append({
            "step":       step,
            "veh_id":     veh_id,
            "origin":     origin,
            "destination": dest,
            "route_len":  len(route_obj.edges),
            "type":       vtype,
            "scenario":   _scenario.name,
        })
    except Exception as e:
        # Log the failure but continue — a single failed injection is not fatal.
        # The counter was already incremented, so the next inject() call uses a new ID.
        print(f"[INJECT FAIL] {veh_id}: {e}")


def current_scenario() -> TrafficScenario | None:
    """Return the currently active scenario (useful for logging in train_mappo.py)."""
    return _scenario


def vehicle_count() -> int:
    """Return the total number of vehicle injection attempts this episode."""
    return _veh_count
