"""
compare_fixed_ppo.py - SEQUENTIAL VERSION
Runs PPO simulation FIRST, then Fixed timing simulation SECOND
Avoids all port and TraCI conflicts by running them one after another

Why sequential?
- Your env.py manages SUMO startup and port allocation
- Running both parallel causes TraCI socket conflicts
- Sequential: collect PPO data → close cleanly → collect Fixed data
"""

import os
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traci
from stable_baselines3 import PPO
from datetime import datetime

# ==================== CONFIG ====================
MODEL_PATH = "ppo_traffic_final.zip"
SUMO_BINARY = "sumo"
SUMO_GUI_BINARY = "sumo-gui"

PLOT_UPDATE_INTERVAL = 0.6
MAX_STEPS = 9000  # 9000 steps = 900 seconds simulation time

# Traffic flow monitoring lanes
INCOMING_LANES = [
    "-E5_0", "-465932558#1.34_0", "-465932558#1.34_1",
    "-465932558#1.34_2", "470773638#1_0", "465932558#0_0", "465932558#0_1",
]
OUTGOING_LANES = [
    "E5_0", "-465932558#0_0", "-465932558#0_1",
    "-470773638#1_0", "465932558#1_0",
]

# ==================== METRIC COLLECTION ====================
ppo_metrics = {k: [] for k in ["delay", "queue", "throughput", "stops", "pressure"]}
fixed_metrics = {k: [] for k in ["delay", "queue", "throughput", "stops", "pressure"]}

def collect_metrics_from_env(info, which_dict):
    """Collect metrics from SumoEnv info dict"""
    m = info.get("metrics", {})
    
    which_dict["delay"].append(m.get("avg_delay", 0))
    which_dict["queue"].append(m.get("avg_queue", 0))
    which_dict["throughput"].append(m.get("throughput_total", 0))
    which_dict["stops"].append(m.get("stop_ratio", 0))
    
    inc = sum(m.get("queues", [])) if m.get("queues") else 0
    which_dict["pressure"].append(inc)

def collect_metrics_from_traci(tr, which_dict):
    """Collect metrics using direct TraCI"""
    veh_ids = tr.vehicle.getIDList()
    
    # Delay
    delays = []
    for vid in veh_ids:
        try:
            v = tr.vehicle.getSpeed(vid)
            allowed = tr.vehicle.getAllowedSpeed(vid)
            d = 1 - (v / allowed) if allowed > 0 else 0
            delays.append(np.clip(d, 0, 1))
        except:
            pass
    which_dict["delay"].append(np.mean(delays) if delays else 0)
    
    # Queue
    q = [tr.lane.getLastStepHaltingNumber(l) for l in INCOMING_LANES]
    which_dict["queue"].append(float(np.mean(q)))
    
    # Throughput
    tp = sum(tr.lane.getLastStepVehicleNumber(l) for l in OUTGOING_LANES)
    which_dict["throughput"].append(float(tp))
    
    # Stops
    stops = 0
    total = 0
    for lane in INCOMING_LANES:
        vids = tr.lane.getLastStepVehicleIDs(lane)
        for vid in vids:
            total += 1
            if tr.vehicle.getSpeed(vid) < 0.1:
                stops += 1
    which_dict["stops"].append((stops / total) if total > 0 else 0)
    
    # Pressure
    inc = sum(tr.lane.getLastStepVehicleNumber(l) for l in INCOMING_LANES)
    out = sum(tr.lane.getLastStepVehicleNumber(l) for l in OUTGOING_LANES)
    which_dict["pressure"].append(float(inc - out))

def export_metrics(filename=None):
    """Export metrics to CSV"""
    if filename is None:
        filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    max_len = max(len(ppo_metrics["delay"]), len(fixed_metrics["delay"]))
    
    # Pad with NaN
    for key in ppo_metrics:
        if len(ppo_metrics[key]) < max_len:
            ppo_metrics[key].extend([np.nan] * (max_len - len(ppo_metrics[key])))
    
    for key in fixed_metrics:
        if len(fixed_metrics[key]) < max_len:
            fixed_metrics[key].extend([np.nan] * (max_len - len(fixed_metrics[key])))
    
    data = {}
    for key in ppo_metrics:
        data[f"ppo_{key}"] = ppo_metrics[key]
        data[f"fixed_{key}"] = fixed_metrics[key]
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n✓ Metrics exported to {filename}")

def print_comparison_report():
    """Print detailed statistical comparison - one metric per section"""
    print("\n" + "="*100)
    print(" "*35 + "DETAILED METRICS REPORT")
    print("="*100)
    
    # DELAY
    print("\n" + "="*100)
    print(" "*35 + "1. DELAY METRIC")
    print("="*100)
    print("(Lower = Better | 0=Full Speed, 1=Stopped)\n")
    
    delay_fixed = np.array(fixed_metrics["delay"])
    delay_ppo = np.array(ppo_metrics["delay"])
    
    print(f"{'Statistic':<20} {'FIXED TIMING':<35} {'PPO MODEL':<35}")
    print("-"*100)
    print(f"{'Mean':<20} {np.mean(delay_fixed):>20.4f}               {np.mean(delay_ppo):>20.4f}")
    print(f"{'Median':<20} {np.median(delay_fixed):>20.4f}               {np.median(delay_ppo):>20.4f}")
    print(f"{'Std Deviation':<20} {np.std(delay_fixed):>20.4f}               {np.std(delay_ppo):>20.4f}")
    print(f"{'Min':<20} {np.min(delay_fixed):>20.4f}               {np.min(delay_ppo):>20.4f}")
    print(f"{'Max':<20} {np.max(delay_fixed):>20.4f}               {np.max(delay_ppo):>20.4f}")
    delay_imp = ((np.mean(delay_fixed) - np.mean(delay_ppo)) / np.mean(delay_fixed) * 100) if np.mean(delay_fixed) > 0 else 0
    print("-"*100)
    print(f"{'✅ PPO Better By':<20} {delay_imp:>45.2f}%")
    print("="*100)
    
    # QUEUE
    print("\n" + "="*100)
    print(" "*35 + "2. QUEUE METRIC")
    print("="*100)
    print("(Lower = Better | Halting vehicles on incoming lanes)\n")
    
    queue_fixed = np.array(fixed_metrics["queue"])
    queue_ppo = np.array(ppo_metrics["queue"])
    
    print(f"{'Statistic':<20} {'FIXED TIMING':<35} {'PPO MODEL':<35}")
    print("-"*100)
    print(f"{'Mean':<20} {np.mean(queue_fixed):>20.4f}               {np.mean(queue_ppo):>20.4f}")
    print(f"{'Median':<20} {np.median(queue_fixed):>20.4f}               {np.median(queue_ppo):>20.4f}")
    print(f"{'Std Deviation':<20} {np.std(queue_fixed):>20.4f}               {np.std(queue_ppo):>20.4f}")
    print(f"{'Min':<20} {np.min(queue_fixed):>20.4f}               {np.min(queue_ppo):>20.4f}")
    print(f"{'Max':<20} {np.max(queue_fixed):>20.4f}               {np.max(queue_ppo):>20.4f}")
    queue_imp = ((np.mean(queue_fixed) - np.mean(queue_ppo)) / np.mean(queue_fixed) * 100) if np.mean(queue_fixed) > 0 else 0
    print("-"*100)
    print(f"{'✅ PPO Better By':<20} {queue_imp:>45.2f}%")
    print("="*100)
    
    # THROUGHPUT
    print("\n" + "="*100)
    print(" "*35 + "3. THROUGHPUT METRIC")
    print("="*100)
    print("(Higher = Better | Vehicles exiting per step)\n")
    
    tp_fixed = np.array(fixed_metrics["throughput"])
    tp_ppo = np.array(ppo_metrics["throughput"])
    
    print(f"{'Statistic':<20} {'FIXED TIMING':<35} {'PPO MODEL':<35}")
    print("-"*100)
    print(f"{'Mean':<20} {np.mean(tp_fixed):>20.4f}               {np.mean(tp_ppo):>20.4f}")
    print(f"{'Median':<20} {np.median(tp_fixed):>20.4f}               {np.median(tp_ppo):>20.4f}")
    print(f"{'Std Deviation':<20} {np.std(tp_fixed):>20.4f}               {np.std(tp_ppo):>20.4f}")
    print(f"{'Min':<20} {np.min(tp_fixed):>20.4f}               {np.min(tp_ppo):>20.4f}")
    print(f"{'Max':<20} {np.max(tp_fixed):>20.4f}               {np.max(tp_ppo):>20.4f}")
    tp_imp = ((np.mean(tp_ppo) - np.mean(tp_fixed)) / np.mean(tp_fixed) * 100) if np.mean(tp_fixed) > 0 else 0
    print("-"*100)
    print(f"{'✅ PPO Better By':<20} {tp_imp:>45.2f}%")
    print("="*100)
    
    # STOPS
    print("\n" + "="*100)
    print(" "*35 + "4. STOPS METRIC")
    print("="*100)
    print("(Lower = Better | Ratio of stopped vehicles, 0=None 1=All)\n")
    
    stops_fixed = np.array(fixed_metrics["stops"])
    stops_ppo = np.array(ppo_metrics["stops"])
    
    print(f"{'Statistic':<20} {'FIXED TIMING':<35} {'PPO MODEL':<35}")
    print("-"*100)
    print(f"{'Mean':<20} {np.mean(stops_fixed):>20.4f}               {np.mean(stops_ppo):>20.4f}")
    print(f"{'Median':<20} {np.median(stops_fixed):>20.4f}               {np.median(stops_ppo):>20.4f}")
    print(f"{'Std Deviation':<20} {np.std(stops_fixed):>20.4f}               {np.std(stops_ppo):>20.4f}")
    print(f"{'Min':<20} {np.min(stops_fixed):>20.4f}               {np.min(stops_ppo):>20.4f}")
    print(f"{'Max':<20} {np.max(stops_fixed):>20.4f}               {np.max(stops_ppo):>20.4f}")
    stops_imp = ((np.mean(stops_fixed) - np.mean(stops_ppo)) / np.mean(stops_fixed) * 100) if np.mean(stops_fixed) > 0 else 0
    print("-"*100)
    print(f"{'✅ PPO Better By':<20} {stops_imp:>45.2f}%")
    print("="*100)
    
    # PRESSURE
    print("\n" + "="*100)
    print(" "*35 + "5. PRESSURE METRIC")
    print("="*100)
    print("(Lower = Better | Congestion accumulation: incoming - outgoing)\n")
    
    pressure_fixed = np.array(fixed_metrics["pressure"])
    pressure_ppo = np.array(ppo_metrics["pressure"])
    
    print(f"{'Statistic':<20} {'FIXED TIMING':<35} {'PPO MODEL':<35}")
    print("-"*100)
    print(f"{'Mean':<20} {np.mean(pressure_fixed):>20.4f}               {np.mean(pressure_ppo):>20.4f}")
    print(f"{'Median':<20} {np.median(pressure_fixed):>20.4f}               {np.median(pressure_ppo):>20.4f}")
    print(f"{'Std Deviation':<20} {np.std(pressure_fixed):>20.4f}               {np.std(pressure_ppo):>20.4f}")
    print(f"{'Min':<20} {np.min(pressure_fixed):>20.4f}               {np.min(pressure_ppo):>20.4f}")
    print(f"{'Max':<20} {np.max(pressure_fixed):>20.4f}               {np.max(pressure_ppo):>20.4f}")
    pressure_imp = ((np.mean(pressure_fixed) - np.mean(pressure_ppo)) / np.mean(pressure_fixed) * 100) if np.mean(pressure_fixed) > 0 else 0
    print("-"*100)
    print(f"{'✅ PPO Better By':<20} {pressure_imp:>45.2f}%")
    print("="*100)
    
    # SUMMARY
    print("\n" + "="*100)
    print(" "*30 + "OVERALL PERFORMANCE SUMMARY")
    print("="*100)
    
    improvements = [delay_imp, queue_imp, tp_imp, stops_imp, pressure_imp]
    avg_imp = np.mean(improvements)
    
    print(f"\n{'Metric':<20} {'PPO Better By':<20} {'Status':<20}")
    print("-"*100)
    print(f"{'Delay':<20} {delay_imp:>18.2f}% {'✅' if delay_imp > 0 else '❌':<20}")
    print(f"{'Queue':<20} {queue_imp:>18.2f}% {'✅' if queue_imp > 0 else '❌':<20}")
    print(f"{'Throughput':<20} {tp_imp:>18.2f}% {'✅' if tp_imp > 0 else '❌':<20}")
    print(f"{'Stops':<20} {stops_imp:>18.2f}% {'✅' if stops_imp > 0 else '❌':<20}")
    print(f"{'Pressure':<20} {pressure_imp:>18.2f}% {'✅' if pressure_imp > 0 else '❌':<20}")
    print("-"*100)
    print(f"{'AVERAGE':<20} {avg_imp:>18.2f}%")
    print("="*100)
    
    if avg_imp > 20:
        print("\n🚀 VERDICT: PPO SIGNIFICANTLY OUTPERFORMS FIXED TIMING!\n")
    elif avg_imp > 5:
        print("\n✅ VERDICT: PPO CLEARLY BETTER THAN FIXED TIMING!\n")
    else:
        print("\n⚠️  VERDICT: RESULTS ARE COMPARABLE\n")

# ==================== PPO SIMULATION ====================
def run_ppo_simulation():
    """Run PPO-controlled traffic simulation"""
    print("\n" + "="*70)
    print("PHASE 1: Running PPO-Controlled Simulation")
    print("="*70)
    
    try:
        from env import SumoEnv
        print("[PPO] ✓ SumoEnv imported")
    except ImportError as e:
        print(f"[PPO] ✗ Failed to import SumoEnv: {e}")
        return False
    
    env = None
    try:
        # Initialize environment
        env = SumoEnv(sumo_cfg="map.sumocfg", use_gui=False)
        print("[PPO] ✓ Environment initialized")
        
        # Load model
        try:
            model = PPO.load(MODEL_PATH)
            print(f"[PPO] ✓ Model loaded from {MODEL_PATH}")
            use_model = True
        except Exception as e:
            print(f"[PPO] ✗ Model load failed: {e}")
            use_model = False
        
        # Reset and run
        obs, info = env.reset()
        print("[PPO] Starting episode...")
        
        step = 0
        done = False
        
        print("[PPO] Running for 9000 steps (900 seconds)...")
        
        while step < MAX_STEPS:
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Collect metrics even if episode is done (reset and continue)
            collect_metrics_from_env(info, ppo_metrics)
            
            step += 1
            if step % 500 == 0:
                print(f"[PPO] Step {step}/{MAX_STEPS}")
            
            # If episode ends, reset and continue for remaining steps
            if (terminated or truncated) and step < MAX_STEPS:
                print(f"[PPO] Episode ended at step {step}, resetting...")
                obs, info = env.reset()
        
        print(f"[PPO] ✓ Episode finished at step {step}")
        return True
    
    except Exception as e:
        print(f"[PPO] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if env:
            try:
                env.close()
                print("[PPO] ✓ Environment closed")
            except:
                pass
        
        # Clean up any lingering SUMO processes
        try:
            os.system("taskkill /IM sumo.exe /F 2>nul")
        except:
            pass
        
        time.sleep(2)

# ==================== FIXED TIMING SIMULATION ====================
def run_fixed_timing_simulation(port=8814):
    """Run fixed timing traffic simulation"""
    print("\n" + "="*70)
    print("PHASE 2: Running Fixed Timing Simulation")
    print("="*70)
    
    # Start SUMO process
    cmd = [
        SUMO_GUI_BINARY,
        "-n", "lol.xml",
        "-r", "routes.rou.xml",
        "--additional-files", "vtypes.add.xml,tls.add.xml",
        "--remote-port", str(port),
        "--step-length", "0.1",
    ]
    print(f"[FIXED] Starting SUMO: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Wait for SUMO to start
    
    tr = None
    try:
        # Connect to SUMO
        for attempt in range(20):
            try:
                tr = traci.connect(port=port, numRetries=3)
                print(f"[FIXED] ✓ Connected on port {port}")
                break
            except Exception as e:
                print(f"[FIXED] Connection attempt {attempt+1}/20 failed: {str(e)[:60]}")
                time.sleep(0.5)
        else:
            print("[FIXED] ✗ Failed to connect")
            return False
        
        # Run simulation
        step = 0
        print("[FIXED] Starting simulation...")
        
        while step < MAX_STEPS:
            tr.simulationStep()
            collect_metrics_from_traci(tr, fixed_metrics)
            
            step += 1
            if step % 100 == 0:
                print(f"[FIXED] Step {step}/{MAX_STEPS}")
        
        print(f"[FIXED] ✓ Simulation finished at step {step}")
        return True
    
    except Exception as e:
        print(f"[FIXED] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if tr:
            try:
                tr.close()
                print("[FIXED] ✓ TraCI closed")
            except:
                pass
        
        if proc.poll() is None:
            proc.terminate()
            time.sleep(0.5)
            print("[FIXED] ✓ SUMO process terminated")

# ==================== PLOTTING ====================
def plot_comparison():
    """Plot each metric in its own figure"""
    print("\n" + "="*70)
    print("PHASE 3: Creating 5 Metric Figures")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_info = [
        ("delay", "Delay", "Normalized Delay (0=Full Speed, 1=Stopped)", "orange"),
        ("queue", "Queue Length", "Average Halting Vehicles", "green"),
        ("throughput", "Throughput", "Vehicles Exiting Per Step", "purple"),
        ("stops", "Stop Ratio", "Ratio of Stopped Vehicles (0=None, 1=All)", "red"),
        ("pressure", "Pressure", "Congestion Accumulation (Incoming - Outgoing)", "brown"),
    ]
    
    for metric_key, title, ylabel, color in metrics_info:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        f = np.array(fixed_metrics[metric_key])
        p = np.array(ppo_metrics[metric_key])
        
        if f.size > 0:
            ax.plot(f, label="Fixed Timing", linewidth=2.5, color='orange', alpha=0.8)
        if p.size > 0:
            ax.plot(p, label="PPO Model", linewidth=2.5, color='blue', alpha=0.8)
        
        ax.set_title(f"{title}: PPO vs Fixed Timing", fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Simulation Step", fontsize=12)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"metric_{metric_key}_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        print(f"✓ Saved: {filename}")
        plt.show()

# ==================== MAIN ====================
def main():
    print("\n" + "="*70)
    print("SUMO PPO vs FIXED TIMING COMPARISON (SEQUENTIAL)")
    print("="*70)
    
    # Phase 1: PPO
    ppo_ok = run_ppo_simulation()
    
    if not ppo_ok:
        print("\n✗ PPO simulation failed")
        return
    
    # Phase 2: Fixed
    fixed_ok = run_fixed_timing_simulation()
    
    if not fixed_ok:
        print("\n✗ Fixed timing simulation failed")
        return
    
    # Phase 3: Analysis
    export_metrics()
    print_comparison_report()
    plot_comparison()
    
    print("\n" + "="*70)
    print("=== COMPARISON COMPLETE ===")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()