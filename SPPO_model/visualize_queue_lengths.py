import traci
import matplotlib.pyplot as plt
import numpy as np

SUMO_CFG = "map.sumocfg"
STEPS = 5000

# === Incoming lanes (7 lanes) ===
incoming_lanes = [
    "-E5_0",
    "-465932558#1.34_0",
    "-465932558#1.34_1",
    "-465932558#1.34_2",
    "470773638#1_0",
    "465932558#0_0",
    "465932558#0_1",
]

# Storage for queue length data
queue_history = {lane: [] for lane in incoming_lanes}

print("Starting SUMO for visualization...")
traci.start(["sumo", "-c", SUMO_CFG])

for step in range(STEPS):
    traci.simulationStep()

    # Record queue length for each lane
    for lane in incoming_lanes:
        q = traci.lane.getLastStepHaltingNumber(lane)
        queue_history[lane].append(q)

traci.close()
print("Simulation finished. Generating plots...")

# =======================================
# PLOTTING
# =======================================
plt.figure(figsize=(14, 8))

for lane in incoming_lanes:
    plt.plot(queue_history[lane], label=lane)

plt.title("Queue Length Over Time (Each Lane)")
plt.xlabel("Simulation Step")
plt.ylabel("Queue Length (# Vehicles)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("queue_lengths.png", dpi=300)
plt.show()

print("Saved plot as queue_lengths.png")
