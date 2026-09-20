"""Convert mappo_train.csv to TensorBoard event files."""
import csv
from torch.utils.tensorboard import SummaryWriter

CSV_PATH = "mappo_logs/mappo_train.csv"
OUT_DIR  = "mappo_logs/tb"

writer = SummaryWriter(OUT_DIR)

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        step = int(row["timestep"])
        writer.add_scalar("train/mean_reward",  float(row["mean_reward"]),  step)
        writer.add_scalar("train/actor_loss",   float(row["actor_loss"]),   step)
        writer.add_scalar("train/critic_loss",  float(row["critic_loss"]),  step)
        writer.add_scalar("train/entropy",      float(row["entropy"]),      step)
        if row["eval_reward"] and float(row["eval_reward"]) != 0.0:
            writer.add_scalar("eval/mean_reward", float(row["eval_reward"]), step)

writer.close()
print(f"Done — TensorBoard events written to: {OUT_DIR}")
