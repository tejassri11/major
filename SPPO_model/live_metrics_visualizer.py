import traci
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SUMO_CFG = "map.sumocfg"

# 7 incoming lanes
incoming_lanes = [
    "-E5_0",
    "-465932558#1.34_0",
    "-465932558#1.34_1",
    "-465932558#1.34_2",
    "470773638#1_0",
    "465932558#0_0",
    "465932558#0_1",
]

# outgoing lanes
outgoing_lanes = [
    "E5_0",
    "-465932558#0_0",
    "-465932558#0_1",
    "-470773638#1_0",
    "465932558#1_0",
]

# Metric histories
history_delay = []
history_queue = []
history_throughput = []
history_stops = []
history_pressure = []

# --------------------------
# START SUMO
# --------------------------
traci.start(["sumo-gui", "-c", SUMO_CFG])

print("Running simulation with LIVE VISUALIZATION...")

# --------------------------
# PLOTTING SETUP
# --------------------------
plt.style.use("ggplot")   # SAFE STYLE, always available

fig, axs = plt.subplots(5, 1, figsize=(12, 16))
ax_delay, ax_queue, ax_tp, ax_stops, ax_press = axs


def update(frame):

    # STEP SUMO
    traci.simulationStep()

    veh_ids = traci.vehicle.getIDList()

    # -------- DELAY --------
    delays = []
    for vid in veh_ids:
        v = traci.vehicle.getSpeed(vid)
        allowed = traci.vehicle.getAllowedSpeed(vid)
        d = 1 - (v / allowed) if allowed > 0 else 0
        delays.append(d)

    avg_delay = np.mean(delays) if delays else 0
    history_delay.append(avg_delay)

    # -------- QUEUE LENGTH --------
    q = [traci.lane.getLastStepHaltingNumber(l) for l in incoming_lanes]
    avg_queue = np.mean(q)
    history_queue.append(avg_queue)

    # -------- THROUGHPUT --------
    throughput = sum(traci.lane.getLastStepVehicleNumber(l) for l in outgoing_lanes)
    history_throughput.append(throughput)

    # -------- STOPS --------
    stops = sum(1 for vid in veh_ids if traci.vehicle.getSpeed(vid) < 0.1)
    stop_ratio = stops / len(veh_ids) if veh_ids else 0
    history_stops.append(stop_ratio)

    # -------- PRESSURE --------
    incoming_sum = sum(traci.lane.getLastStepVehicleNumber(l) for l in incoming_lanes)
    outgoing_sum = sum(traci.lane.getLastStepVehicleNumber(l) for l in outgoing_lanes)
    pressure = incoming_sum - outgoing_sum
    history_pressure.append(pressure)

    # -------------------------------------------
    # UPDATE PLOTS
    # -------------------------------------------

    # Delay
    ax_delay.clear()
    ax_delay.plot(history_delay, label="Avg Delay")
    ax_delay.set_title("Average Delay (0 = no delay, 1 = full stop)")
    ax_delay.set_ylim(0, 1)
    ax_delay.grid(True)

    # Queue
    ax_queue.clear()
    ax_queue.plot(history_queue, label="Avg Queue Length")
    ax_queue.set_title("Average Queue Length (incoming lanes)")
    ax_queue.grid(True)

    # Throughput
    ax_tp.clear()
    ax_tp.plot(history_throughput, label="Throughput")
    ax_tp.set_title("Vehicles Passing (Throughput)")
    ax_tp.grid(True)

    # Stops
    ax_stops.clear()
    ax_stops.plot(history_stops, label="Stop Ratio")
    ax_stops.set_title("Stop Ratio")
    ax_stops.grid(True)

    # Pressure
    ax_press.clear()
    ax_press.plot(history_pressure, label="Pressure")
    ax_press.set_title("Pressure (incoming - outgoing)")
    ax_press.grid(True)

    plt.tight_layout()


# ANIMATION — continuous (no frames limit)
ani = FuncAnimation(fig, update, interval=50, repeat=False)

plt.show()

# CLOSE SUMO SAFELY AFTER PLOT WINDOW CLOSE
traci.close()
