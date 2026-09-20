"""
calculate_steps.py - Figure out how many steps = 900 seconds in SUMO

Run this to see the conversion ratio, then we'll update compare.py
"""

import subprocess
import time
import traci

def measure_step_to_time_ratio(port=8815, test_steps=100):
    """Run SUMO for N steps and measure how much simulation time passes"""
    print("\n" + "="*70)
    print("MEASURING STEP-TO-TIME RATIO")
    print("="*70)
    
    # Start SUMO
    cmd = [
        "sumo",
        "-n", "lol.xml",
        "-r", "routes.rou.xml",
        "--additional-files", "vtypes.add.xml,tls.add.xml",
        "--remote-port", str(port),
        "--step-length", "0.1",
    ]
    print(f"\n[MEASURE] Starting SUMO with step-length=0.1...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)
    
    try:
        tr = traci.connect(port=port, numRetries=10)
        print(f"[MEASURE] ✓ Connected")
        
        start_time = tr.simulation.getTime()
        print(f"[MEASURE] Starting simulation time: {start_time}s\n")
        
        for step in range(test_steps):
            tr.simulationStep()
        
        end_time = tr.simulation.getTime()
        elapsed_sim_time = end_time - start_time
        
        print(f"[MEASURE] After {test_steps} steps:")
        print(f"  Simulation time elapsed: {elapsed_sim_time:.1f}s")
        print(f"  Steps per second: {test_steps / elapsed_sim_time:.2f}")
        
        # Calculate for 900 seconds
        steps_needed_for_900s = int((900 / elapsed_sim_time) * test_steps)
        
        print(f"\n" + "="*70)
        print(f"RESULT FOR 900 SECONDS:")
        print(f"="*70)
        print(f"  Need {steps_needed_for_900s} steps to get ~900 seconds")
        print(f"\nUpdate compare.py with:")
        print(f"  MAX_STEPS = {steps_needed_for_900s}")
        print("="*70)
        
        tr.close()
    
    except Exception as e:
        print(f"[MEASURE] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        proc.terminate()

if __name__ == "__main__":
    measure_step_to_time_ratio(test_steps=100)