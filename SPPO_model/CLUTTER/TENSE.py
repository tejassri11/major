import traci
import csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------
SUMO_CFG = "map.sumocfg"
TLS_ID = "6073919354"
OUTPUT_CSV = "cycle_log_grouped.csv"
TENSORBOARD_LOG_DIR = "./runs/traffic_light_analysis"

# ---------------------------------------------------------
# PHASE GROUPING CONFIGURATION
# ---------------------------------------------------------
PHASE_GROUPS = {
    0: {"group": "Phase Group 1", "lanes": "Lanes 0,1,2", "type": "GREEN", "expected_duration": 30},
    1: {"group": "Phase Group 1", "lanes": "Lanes 0,1,2", "type": "AMBER", "expected_duration": 3},
    2: {"group": "Phase Group 1", "lanes": "Lanes 0,1,2", "type": "ALL-RED", "expected_duration": 2},
    
    3: {"group": "Phase Group 2", "lanes": "Lane 3", "type": "GREEN", "expected_duration": 25},
    4: {"group": "Phase Group 2", "lanes": "Lane 3", "type": "AMBER", "expected_duration": 3},
    5: {"group": "Phase Group 2", "lanes": "Lane 3", "type": "ALL-RED", "expected_duration": 2},
    
    6: {"group": "Phase Group 3", "lanes": "Lanes 5,9", "type": "GREEN", "expected_duration": 30},
    7: {"group": "Phase Group 3", "lanes": "Lanes 5,9", "type": "AMBER", "expected_duration": 3},
    8: {"group": "Phase Group 3", "lanes": "Lanes 5,9", "type": "ALL-RED", "expected_duration": 2},
    
    9: {"group": "Phase Group 4", "lanes": "Lanes 4,8", "type": "GREEN", "expected_duration": 25},
    10: {"group": "Phase Group 4", "lanes": "Lanes 4,8", "type": "AMBER", "expected_duration": 3},
    11: {"group": "Phase Group 4", "lanes": "Lanes 4,8", "type": "ALL-RED", "expected_duration": 2},
    
    12: {"group": "Phase Group 5", "lanes": "Lanes 6,7", "type": "GREEN", "expected_duration": 30},
    13: {"group": "Phase Group 5", "lanes": "Lanes 6,7", "type": "AMBER", "expected_duration": 3},
    14: {"group": "Phase Group 5", "lanes": "Lanes 6,7", "type": "ALL-RED", "expected_duration": 2},
}

# Create group name mapping
GROUP_NAMES = {
    "Phase Group 1": "Lanes 0,1,2",
    "Phase Group 2": "Lane 3",
    "Phase Group 3": "Lanes 5,9",
    "Phase Group 4": "Lanes 4,8",
    "Phase Group 5": "Lanes 6,7"
}

# ---------------------------------------------------------
# INITIALIZE TENSORBOARD
# ---------------------------------------------------------
writer = SummaryWriter(TENSORBOARD_LOG_DIR)

# ---------------------------------------------------------
# INITIALIZE CSV
# ---------------------------------------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow([
        "timestamp",
        "phase_index",
        "group",
        "lanes",
        "phase_type",
        "duration",
        "expected_duration",
        "deviation"
    ])

print(f"TensorBoard logging to: {TENSORBOARD_LOG_DIR}")
print(f"CSV output to: {OUTPUT_CSV}")
print(f"Tracking traffic light: {TLS_ID}\n")

# ---------------------------------------------------------
# START SUMO
# ---------------------------------------------------------
traci.start(["sumo-gui", "-c", SUMO_CFG])

# ---------------------------------------------------------
# INTERNAL STATE
# ---------------------------------------------------------
prev_phase = traci.trafficlight.getPhase(TLS_ID)
phase_start_time = traci.simulation.getTime()
cycle_count = 0
green_cycles = {}  # Track green cycles per group

# Initialize tracking for each group
for group in GROUP_NAMES.keys():
    green_cycles[group] = []

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
try:
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        current_phase = traci.trafficlight.getPhase(TLS_ID)
        
        # Detect PHASE CHANGE
        if current_phase != prev_phase:
            duration = sim_time - phase_start_time
            cycle_count += 1
            
            # Get phase info from grouping config
            if prev_phase in PHASE_GROUPS:
                phase_info = PHASE_GROUPS[prev_phase]
                group = phase_info["group"]
                lanes = phase_info["lanes"]
                phase_type = phase_info["type"]
                expected = phase_info["expected_duration"]
                deviation = duration - expected
                
                # Save to CSV
                with open(OUTPUT_CSV, "a", newline="") as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow([
                        sim_time,
                        prev_phase,
                        group,
                        lanes,
                        phase_type,
                        f"{duration:.2f}",
                        expected,
                        f"{deviation:.2f}"
                    ])
                
                # Log to TensorBoard
                # 1. Duration per phase
                writer.add_scalar(f"Phase/{group}/duration", duration, cycle_count)
                
                # 2. Deviation from expected
                writer.add_scalar(f"Phase/{group}/deviation", deviation, cycle_count)
                
                # 3. Track only GREEN phases
                if phase_type == "GREEN":
                    green_cycles[group].append(duration)
                    writer.add_scalar(f"GreenPhase/{group}/duration", duration, len(green_cycles[group]))
                    writer.add_scalar(f"GreenPhase/{group}/deviation", deviation, len(green_cycles[group]))
                
                # Print to terminal
                status = "✓" if abs(deviation) < 0.5 else "⚠"
                print(
                    f"[{sim_time:7.2f}s] {group} ({lanes}) | {phase_type:8s} | "
                    f"Duration: {duration:6.2f}s (expected: {expected:2d}s, "
                    f"deviation: {deviation:+6.2f}s) {status}"
                )
            
            # Reset trackers
            prev_phase = current_phase
            phase_start_time = sim_time

finally:
    traci.close()
    
    # Calculate and log statistics
    print("\n" + "="*70)
    print("PHASE ANALYSIS SUMMARY")
    print("="*70 + "\n")
    
    for group, durations in green_cycles.items():
        if len(durations) > 0:
            mean = np.mean(durations)
            std = np.std(durations)
            cv = (std / mean) * 100 if mean > 0 else 0
            min_d = np.min(durations)
            max_d = np.max(durations)
            
            lanes = GROUP_NAMES[group]
            
            # Log summary statistics to TensorBoard
            writer.add_scalar(f"Summary/{group}/mean_duration", mean, 1)
            writer.add_scalar(f"Summary/{group}/std_dev", std, 1)
            writer.add_scalar(f"Summary/{group}/cv_percent", cv, 1)
            writer.add_scalar(f"Summary/{group}/min_duration", min_d, 1)
            writer.add_scalar(f"Summary/{group}/max_duration", max_d, 1)
            
            # Log as histograms for distribution
            writer.add_histogram(f"Distribution/{group}/durations", np.array(durations), 1)
            
            print(f"{group} ({lanes}):")
            print(f"  Green cycles: {len(durations)}")
            print(f"  Mean duration: {mean:.2f}s")
            print(f"  Std deviation: {std:.2f}s")
            print(f"  Coefficient of Variation: {cv:.1f}%")
            print(f"  Min: {min_d:.2f}s, Max: {max_d:.2f}s")
            print()
    
    # Overall statistics
    all_green_durations = [d for durations in green_cycles.values() for d in durations]
    if all_green_durations:
        overall_cv = (np.std(all_green_durations) / np.mean(all_green_durations)) * 100
        writer.add_scalar("Overall/average_cv_percent", overall_cv, 1)
        writer.add_text("Overall/control_type", 
                       "FIXED-TIME" if overall_cv < 15 else "ADAPTIVE", 1)
        
        print("="*70)
        print(f"OVERALL AVERAGE CV: {overall_cv:.1f}%")
        print(f"CONTROL TYPE: {'FIXED-TIME' if overall_cv < 15 else 'ADAPTIVE'}")
        print("="*70)
    
    # Close TensorBoard writer
    writer.flush()
    writer.close()
    
    print(f"\n✓ CSV saved to: {OUTPUT_CSV}")
    print(f"✓ TensorBoard logs saved to: {TENSORBOARD_LOG_DIR}")
    print(f"\nTo view TensorBoard dashboard, run:")
    print(f"  tensorboard --logdir {TENSORBOARD_LOG_DIR}")