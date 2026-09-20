import random

# ── Edge classification for Delhi-Gurgaon NH48 Corridor ───────────────────────
# "Inbound"  = coming from the main highway into local arterial
HIGHWAY_ENTRIES: set[str] = {"1080437656#0", "1314296767", "23197680#0", "24501730#0", "28534416#0", "415417690#0"}
LOCAL_DESTS: set[str]     = {"1080437656#5", "1314307403", "1314322247#20", "28534416#20"}

# "Outbound" = leaving local arterial onto highway
LOCAL_ORIGINS: set[str]   = {"35538104", "38133453#0", "414318559", "540215197"}
HIGHWAY_EXITS: set[str]   = {"142873095#1", "24496274#3", "24501725#14", "24501787#3"}

# "Transit"  = through-traffic along the primary NH48 trunk
TRANSIT_ENTRIES: set[str] = {"1080437656#0", "28534416#0"}
TRANSIT_EXITS: set[str]   = {"1080437656#5", "28534416#20"}

# ── Scenario profiles ─────────────────────────────────────────────────────────
# direction_mode: which OD subset inject() draws from
#   "any"      → all active pairs (background traffic)
#   "inbound"  → highway entry → local dest  (morning rush, people going to work)
#   "outbound" → local origin → highway exit (evening rush, people going home)
#   "transit"  → E0 / 465932558#2_C corridor (holiday through-traffic)
#
# phase_mults: per-1000-step multipliers [0-1k, 1-2k, 2-3k, 3-4k, 4-5k]
# base_rate × phase_mult = EXPECTED vehicles spawned per simulation step (not capped;
# values > 1.0 spawn multiple vehicles per step — see traffic_injector.inject()).

PROFILES: dict[str, dict] = {
    "low": {
        # ~0.06 veh/step peak — quiet side streets, Sunday morning
        "base_rate": 0.05,
        "phase_mults": [0.8, 1.0, 1.2, 1.0, 0.8],
        "blocked_origins": [],
        "direction_mode": "any",
        "description": "Light off-peak traffic, dispersed routes",
    },
    "normal": {
        # ~0.18 veh/step peak — steady weekday mid-morning
        "base_rate": 0.10,
        "phase_mults": [0.6, 0.8, 1.2, 1.5, 1.8],
        "blocked_origins": [],
        "direction_mode": "any",
        "description": "Typical weekday mixed demand",
    },
    "rush_hour_am": {
        # ~0.24 veh/step peak — A1 inbound surge 07:00-08:30
        "base_rate": 0.12,
        "phase_mults": [2.0, 1.8, 1.4, 0.8, 0.4],
        "blocked_origins": [],
        "direction_mode": "inbound",
        "description": "Morning rush — high inbound flow, people going to work",
    },
    "rush_hour_pm": {
        # ~0.24 veh/step peak — outbound surge 16:30-18:00
        "base_rate": 0.12,
        "phase_mults": [2.0, 1.8, 1.4, 1.0, 0.8],
        "blocked_origins": [],
        "direction_mode": "outbound",
        "description": "Evening rush — high outbound flow, people going home",
    },
    "holiday": {
        # ~1.05 veh/step peak — A1 long-distance holiday traffic
        "base_rate": 0.35,
        "phase_mults": [2.0, 2.5, 3.0, 2.5, 2.0],
        "blocked_origins": [],
        "direction_mode": "transit",
        "description": "Holiday — sustained transit traffic through town (E0 ↔ 465932558#2_C)",
    },
    "incident": {
        # Same as normal but two highway entries blocked
        "base_rate": 0.10,
        "phase_mults": [0.6, 0.8, 1.2, 1.5, 1.8],
        "blocked_origins": ["-E5_B", "-E5"],
        "direction_mode": "any",
        "description": "Normal demand but two highway entries closed",
    },
}

PROFILE_NAMES: list[str] = list(PROFILES.keys())


class TrafficScenario:
    def __init__(self, name: str | None = None):
        if name is None:
            name = random.choice(PROFILE_NAMES)
        if name not in PROFILES:
            raise ValueError(f"Unknown scenario '{name}'. Valid: {PROFILE_NAMES}")
        self.name = name
        self._p = PROFILES[name]

    def demand_rate(self, step: int) -> float:
        """Expected number of vehicles to spawn this step (may exceed 1.0)."""
        phase_idx = min(step // 1000, len(self._p["phase_mults"]) - 1)
        return self._p["base_rate"] * self._p["phase_mults"][phase_idx]

    @property
    def direction_mode(self) -> str:
        return self._p["direction_mode"]

    @property
    def blocked_origins(self) -> set[str]:
        return set(self._p["blocked_origins"])

    @property
    def description(self) -> str:
        return self._p["description"]

    @staticmethod
    def random() -> "TrafficScenario":
        return TrafficScenario(random.choice(PROFILE_NAMES))

    def __repr__(self) -> str:
        return f"TrafficScenario('{self.name}')"
