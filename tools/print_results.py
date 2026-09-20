import pandas as pd
import numpy as np

df = pd.read_csv("eval_summary.csv")
SCENARIOS = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]

for metric, label, lower_better in [
    ("mean_wait",  "Mean Wait (s/lane)",          True),
    ("mean_queue", "Mean Queue (veh/lane)",        True),
    ("throughput", "Total Throughput (vehicles)",  False),
]:
    print(f"\n  {label}")
    print(f"  {'Scenario':<18} {'MAPPO':>9} {'Fixed':>9} {'Random':>9}  {'vs Fixed':>10}")
    print("  " + "-" * 60)
    for s in SCENARIOS:
        m = df[(df.controller == "mappo") & (df.scenario == s)][metric].values[0]
        f = df[(df.controller == "fixed") & (df.scenario == s)][metric].values[0]
        r = df[(df.controller == "random") & (df.scenario == s)][metric].values[0]
        pct = (f - m) / max(abs(f), 1e-6) * 100 if lower_better else (m - f) / max(abs(f), 1e-6) * 100
        sign = "+" if pct > 0 else ""
        print(f"  {s:<18} {m:>9.2f} {f:>9.2f} {r:>9.2f}  {sign}{pct:>8.1f}%")
