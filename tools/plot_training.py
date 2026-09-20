import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

df = pd.read_csv("mappo_logs/mappo_train.csv")

def smooth(series, window=30):
    return series.rolling(window=window, min_periods=1, center=True).mean()

steps = df["timestep"] / 1_000_000  # show in millions

fig = plt.figure(figsize=(14, 10))
fig.suptitle("MAPPO Training — Palapye A1 Corridor", fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# --- Mean reward ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(steps, df["mean_reward"], alpha=0.2, color="steelblue", linewidth=0.8, label="Raw")
ax1.plot(steps, smooth(df["mean_reward"]), color="steelblue", linewidth=2, label="Smoothed (30-update window)")
ax1.set_title("Mean Episode Reward", fontweight="bold")
ax1.set_xlabel("Timesteps (millions)")
ax1.set_ylabel("Mean Reward")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# annotate best reward
best_idx = df["mean_reward"].idxmax()
best_val = df["mean_reward"].max()
best_step = steps[best_idx]
ax1.annotate(f"Best: {best_val:.3f} @ {best_step:.2f}M",
             xy=(best_step, best_val),
             xytext=(best_step + 0.05, best_val + 0.5),
             fontsize=8, color="darkgreen",
             arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2))

# --- Actor loss ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(steps, df["actor_loss"], alpha=0.2, color="tomato", linewidth=0.8)
ax2.plot(steps, smooth(df["actor_loss"]), color="tomato", linewidth=2)
ax2.set_title("Actor Loss", fontweight="bold")
ax2.set_xlabel("Timesteps (millions)")
ax2.set_ylabel("Loss")
ax2.grid(True, alpha=0.3)

# --- Critic loss ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(steps, df["critic_loss"], alpha=0.2, color="darkorange", linewidth=0.8)
ax3.plot(steps, smooth(df["critic_loss"]), color="darkorange", linewidth=2)
ax3.set_title("Critic Loss", fontweight="bold")
ax3.set_xlabel("Timesteps (millions)")
ax3.set_ylabel("Loss")
ax3.grid(True, alpha=0.3)

# --- Entropy (overlaid on reward axes as inset or separate) ---
ax4_twin = ax1.twinx()
ax4_twin.plot(steps, smooth(df["entropy"]), color="grey", linewidth=1.5,
              linestyle="--", alpha=0.7, label="Entropy (smoothed)")
ax4_twin.set_ylabel("Entropy", color="grey")
ax4_twin.tick_params(axis="y", labelcolor="grey")
ax4_twin.legend(loc="upper right", fontsize=9)

plt.savefig("mappo_training_curve.png", dpi=150, bbox_inches="tight")
print("Saved: mappo_training_curve.png")

# --- Print summary stats ---
print("\n=== Training Summary ===")
print(f"Total updates:      {len(df)}")
print(f"Final timestep:     {df['timestep'].iloc[-1]:,}")
print(f"Start reward:       {df['mean_reward'].iloc[:10].mean():.3f}")
print(f"End reward:         {df['mean_reward'].iloc[-50:].mean():.3f}")
print(f"Best reward:        {df['mean_reward'].max():.3f}  (update {df['mean_reward'].idxmax()})")
print(f"Start entropy:      {df['entropy'].iloc[:10].mean():.3f}")
print(f"Final entropy:      {df['entropy'].iloc[-50:].mean():.3f}")
print(f"Final actor loss:   {df['actor_loss'].iloc[-50:].mean():.5f}")
print(f"Final critic loss:  {df['critic_loss'].iloc[-50:].mean():.4f}")
