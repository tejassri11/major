"""
PPO Dashboard (Fixed-System Style, Blue Color-Blind Friendly)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime


def find_latest_csv():
    files = glob.glob("comparison_*.csv")
    if not files:
        print("❌ No comparison CSV files found.")
        return None
    return max(files, key=lambda f: f)


def plot_ppo_dashboard(csv_file=None):

    if csv_file is None:
        csv_file = find_latest_csv()
    if csv_file is None:
        return

    print(f"📥 Loading PPO metrics from: {csv_file}")
    df = pd.read_csv(csv_file)

    metrics = {
        "Average Delay": df["ppo_delay"].dropna().values,
        "Average Queue Length": df["ppo_queue"].dropna().values,
        "Throughput": df["ppo_throughput"].dropna().values,
        "Stop Ratio": df["ppo_stops"].dropna().values,
        "Pressure": df["ppo_pressure"].dropna().values,
    }

    # --- Fixed-style settings ---
    line_color = "#1f77b4"      # Blue (color-blind friendly)
    fill_alpha = 0.28
    bg_color = "#f0f0f0"        # Light gray background

    fig, axs = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "PPO Traffic Control – Performance Metrics (Blue Theme)",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    for ax, (label, data) in zip(axs, metrics.items()):

        ax.set_facecolor(bg_color)

        # Main blue line
        ax.plot(
            data,
            color=line_color,
            linewidth=1.8,
        )

        # Fill with blue
        ax.fill_between(
            range(len(data)),
            data,
            alpha=fill_alpha,
            color=line_color
        )

        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.28)

        # Stats box
        ax.text(
            0.02, 0.90,
            f"Mean: {np.mean(data):.3f}\nStd:  {np.std(data):.3f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.55)
        )

    axs[-1].set_xlabel("Simulation Step", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fname = f"ppo_dashboard_blue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fname, dpi=160, bbox_inches="tight")

    print(f"✅ Saved as: {fname}")
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    plot_ppo_dashboard(args.file)


if __name__ == "__main__":
    main()
