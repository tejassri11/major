import traci

print("Starting SUMO clean test...")
traci.start(["sumo-gui", "-c", "map.sumocfg"])

for i in range(200):
    traci.simulationStep()

traci.close()
print("SUMO works.")
