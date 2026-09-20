import traci
import numpy as np
import time

SUMO_CFG = "map.sumocfg"
STEPS = 5000   # or full simulation length

# Incoming lanes for your intersection
incoming_lanes = [
    "-E5_0",
    "-465932558#1.34_0",
    "-465932558#1.34_1",
    "-465932558#1.34_2",
    "470773638#1_0",
    "465932558#0_0",
    "465932558#0_1",
]

# Outgoing lanes
outgoing_lanes = [
    "E5_0",
    "-465932558#0_0",
    "-465932558#0_1",
    "-470773638#1_0",
    "465932558#1_0",
]

# ----------------------------------------
# Metric Storage
# ----------------------------------------
delays = []
queues = []
stops = []
travel_times = []
throughput_list = []
pressure_list = []

vehicle_entry_time = {}  # For exact travel time tracking

# ----------------------------------------
# START SUMO
# ----------------------------------------
traci.start(["sumo", "-c", SUMO_CFG])

print("Evaluating FIXED TRAFFIC LIGHT...")

for step in range(STEPS):

    traci.simulationStep()

    veh_ids = traci.vehicle.getIDList()

    # ---------------------------
    # Delay
    # ---------------------------
    step_delay = []
    for vid in veh_ids:
        speed = traci.vehicle.getSpeed(vid)
        allowed = traci.vehicle.getAllowedSpeed(vid)
        d = 1 - (speed / allowed) if allowed > 0 else 0
        step_delay.append(d)
    delays.append(np.mean(step_delay) if len(step_delay) else 0)

    # ---------------------------
    # Queue Length
    # ---------------------------
    q = [traci.lane.getLastStepHaltingNumber(l) for l in incoming_lanes]
    queues.append(np.mean(q))

    # ---------------------------
    # Stops
    # ---------------------------
    s = 0
    for vid in veh_ids:
        if traci.vehicle.getSpeed(vid) < 0.1:
            s += 1
    stop_ratio = s / len(veh_ids) if veh_ids else 0
    stops.append(stop_ratio)

    # ---------------------------
    # Travel Time Tracking
    # ---------------------------
    for vid in veh_ids:
        if vid not in vehicle_entry_time:
            vehicle_entry_time[vid] = step

    for vid in list(vehicle_entry_time.keys()):
        if vid not in veh_ids:  # vehicle left network
            tt = step - vehicle_entry_time[vid]
            travel_times.append(tt)
            del vehicle_entry_time[vid]

    # ---------------------------
    # Throughput
    # ---------------------------
    T = sum(traci.lane.getLastStepVehicleNumber(l) for l in outgoing_lanes)
    throughput_list.append(T)

    # ---------------------------
    # Pressure
    # ---------------------------
    inc = sum(traci.lane.getLastStepVehicleNumber(l) for l in incoming_lanes)
    out = sum(traci.lane.getLastStepVehicleNumber(l) for l in outgoing_lanes)
    pressure_list.append(inc - out)

traci.close()

# ----------------------------------------
# FINAL RESULTS
# ----------------------------------------

results = {
    "Average Delay": np.mean(delays),
    "Average Queue Length": np.mean(queues),
    "Stop Ratio": np.mean(stops),
    "Average Travel Time": np.mean(travel_times) if travel_times else 0,
    "Average Throughput": np.mean(throughput_list),
    "Average Pressure": np.mean(pressure_list)
}

print("\n============== FIXED LIGHT PERFORMANCE ==============")
for k, v in results.items():
    print(f"{k}: {v:.3f}")

print("=====================================================")
