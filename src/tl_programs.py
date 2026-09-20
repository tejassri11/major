"""
Traffic light program definitions.

All three TL programs are defined in network/tls.add.xml and loaded by SUMO
at startup — no runtime setProgramLogic calls are needed or made.

  TL_A / TL_B : 4-phase program  (phases 0,1,2,3)
    phase 0 → arms {-465932558#1.34, 465932558#0} green
    phase 2 → arms {-E5, 470773638#1} green
    agent actions: 0→phase 0,  1→phase 2

  TL_C        : 6-phase program  (phases 0..5)
    phase 0 → approach -465932558#2
    phase 2 → approach 470773638#1
    phase 4 → approach E0
    agent actions: 0→phase 0,  1→phase 2,  2→phase 4
"""


def apply_tl_programs():
    """No-op — programs loaded from network/tls.add.xml at SUMO startup."""
    print("[TL CONFIG] Programs loaded via tls.add.xml (no TraCI override needed)")
