"""Split main.tex into two standalone, compilable halves for Turnitin length limits.

Part 1 (PPO / Phase 1):  preamble + title page + TOC + Chapters 1-5 (Intro -> Discussion)
Part 2 (MAPPO / Phase 2): preamble + short title + Scope Extension -> Recommendations + Appendices
Both halves include the bibliography so \\cite resolves.
"""
from pathlib import Path

here = Path(__file__).parent
lines = (here / "main.tex").read_text(encoding="utf-8").splitlines(keepends=True)

# 1-indexed line numbers from inspection of main.tex
PREAMBLE        = lines[0:69]      # 1..69   packages + formatting
DOC_SETUP       = lines[69:80]     # 70..80  \begin{document} + page geometry + \pagestyle{empty}
FRONT_MATTER    = lines[80:141]    # 81..141 title page, TOC, lists, \pagenumbering{arabic}
PHASE1          = lines[141:1700]  # 142..1700  Ch1 Introduction -> end of Ch5 Discussion
PHASE2          = lines[1700:2996] # 1701..2996 Scope Extension -> Appendix code listings
BIB             = lines[2996:2998] # 2997..2998 \bibliographystyle + \bibliography
END             = "\n\\end{document}\n"

# ---- Part 1 -------------------------------------------------------------
part1 = PREAMBLE + DOC_SETUP + FRONT_MATTER + PHASE1 + ["\n"] + BIB + [END]
(here / "main_part1_ppo.tex").write_text("".join(part1), encoding="utf-8")

# ---- Part 2 -------------------------------------------------------------
part2_title = [
    "\n",
    "\\begin{center}\n",
    "{\\Large\\bfseries Adaptive Multi-Agent Reinforcement Learning for Traffic Signal\n",
    "Optimisation along the Palapye A1 Urban Corridor}\\\\[0.4cm]\n",
    "{\\large Part 2 of 2 --- Multi-Agent (MAPPO) Phase}\\\\[0.2cm]\n",
    "Tsotlhe Nayang Seiphepi (21001137)\n",
    "\\end{center}\n",
    "\n",
    "\\newpage\n",
    "\\pagenumbering{arabic}\n",
    "\\onehalfspacing\n",
    "\n",
]
part2 = PREAMBLE + DOC_SETUP + part2_title + PHASE2 + ["\n"] + BIB + [END]
(here / "main_part2_mappo.tex").write_text("".join(part2), encoding="utf-8")

print("wrote main_part1_ppo.tex  (%d lines)" % len(part1))
print("wrote main_part2_mappo.tex (%d lines)" % len(part2))
