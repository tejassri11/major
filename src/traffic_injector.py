import xml.etree.ElementTree as ET
import random
import traci
from traffic_scenario import (
    TrafficScenario,
    HIGHWAY_ENTRIES, LOCAL_DESTS,
    LOCAL_ORIGINS, HIGHWAY_EXITS,
    TRANSIT_ENTRIES, TRANSIT_EXITS,
)

FLOW_FILE = "network/triple_routes_flows.rou.xml"

TYPES = [
    "two_wheeler",
    "moto",
    "auto_rickshaw",
    "car_small",
    "car_normal",
    "car_suv",
    "bus_city",
    "truck_lcv",
]

INDIAN_MODAL_WEIGHTS = [0.50, 0.10, 0.15, 0.12, 0.05, 0.03, 0.03, 0.02]

_ORIGIN_REDUCTION: dict[str, dict[str, float]] = {
    "holiday": {"1080437656#0": 0.40},
}


_all_od_pairs: list[tuple[str, str]] = []

_inbound_pairs: list[tuple[str, str]] = []
_outbound_pairs: list[tuple[str, str]] = []
_transit_pairs: list[tuple[str, str]] = []

_active_pairs: list[tuple[str, str]] = []

_scenario: TrafficScenario | None = None

_veh_count = 0

_scenario_step = 0

injection_log: list[dict] = []


def init(scenario: TrafficScenario | None = None):
    global _all_od_pairs, _veh_count, _scenario_step, injection_log

    _veh_count = 0
    _scenario_step = 0
    injection_log.clear()

    if scenario is None:
        scenario = TrafficScenario("normal")

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

    seen: set[tuple[str, str]] = set()
    unique_raw = [p for p in raw if not (p in seen or seen.add(p))]

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
    global _scenario_step
    _scenario_step = 0
    _apply_scenario(scenario)


def _apply_scenario(scenario: TrafficScenario):
    global _inbound_pairs, _outbound_pairs, _transit_pairs, _active_pairs, _scenario
    _scenario = scenario

    blocked   = scenario.blocked_origins
    available = [(o, d) for o, d in _all_od_pairs if o not in blocked]

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
        pool = available

    _active_pairs = pool if pool else available

    print(
        f"[INJECTOR] '{scenario.name}' | mode={mode} | "
        f"pool={len(_active_pairs)} pairs "
        f"(inbound={len(_inbound_pairs)}, outbound={len(_outbound_pairs)}, "
        f"transit={len(_transit_pairs)}, blocked={len(_all_od_pairs)-len(available)})"
    )


def inject(step: int):
    global _scenario_step

    if not _active_pairs or _scenario is None:
        return

    rate = _scenario.demand_rate(_scenario_step)
    _scenario_step += 1

    n = int(rate) + (1 if random.random() < (rate - int(rate)) else 0)
    for _ in range(n):
        _spawn_one(step)


def _spawn_one(step: int):
    global _veh_count

    origin, dest = random.choice(_active_pairs)

    reduction = _ORIGIN_REDUCTION.get(_scenario.name, {}).get(origin, 0.0)
    if reduction and random.random() < reduction:
        return

    route_obj = traci.simulation.findRoute(origin, dest)
    if not route_obj or not route_obj.edges:
        return

    veh_id   = f"veh_{_veh_count}"
    route_id = f"route_{veh_id}"
    vtype    = random.choices(TYPES, weights=INDIAN_MODAL_WEIGHTS, k=1)[0]

    _veh_count += 1

    try:
        traci.route.add(route_id, route_obj.edges)
        traci.vehicle.add(
            vehID=veh_id,
            routeID=route_id,
            typeID=vtype,
            depart="now",
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
        print(f"[INJECT FAIL] {veh_id}: {e}")


def current_scenario() -> TrafficScenario | None:
    return _scenario


def vehicle_count() -> int:
    return _veh_count
