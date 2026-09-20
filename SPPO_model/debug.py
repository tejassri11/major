"""
debug_ppo_metrics.py - Inspect what metrics your env is actually returning

This script runs PPO for a few steps and shows you exactly what info dict
contains so we can see where the metrics are being lost.
"""

import numpy as np
from env import SumoEnv
from stable_baselines3 import PPO
import json

# ==================== CONFIG ====================
MODEL_PATH = "ppo_traffic_final.zip"

def debug_env_info():
    """Check what info dict your environment returns"""
    print("\n" + "="*70)
    print("DEBUGGING ENVIRONMENT INFO DICT")
    print("="*70)
    
    env = SumoEnv(sumo_cfg="map.sumocfg", use_gui=False)
    obs, info = env.reset()
    
    print("\n[DEBUG] Initial reset info keys:")
    print(f"  Keys: {list(info.keys())}")
    print(f"  Full info: {info}\n")
    
    print("[DEBUG] Running 10 steps and printing info each time:\n")
    print(f"{'Step':<6} {'Info Keys':<40} {'Metrics':<50}")
    print("-" * 100)
    
    try:
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            info_keys = list(info.keys())
            metrics = info.get("metrics", {})
            metric_keys = list(metrics.keys()) if metrics else []
            
            print(f"{step:<6} {str(info_keys):<40} {str(metric_keys):<50}")
            
            if step == 0:
                print(f"\n[DEBUG] Full info dict at step 0:")
                print(json.dumps(str(info), indent=2))
                print(f"\n[DEBUG] Full metrics dict:")
                print(json.dumps(str(metrics), indent=2))
    
    finally:
        env.close()

def debug_metric_collection():
    """See what metrics we're collecting from env"""
    print("\n" + "="*70)
    print("DEBUGGING METRIC COLLECTION")
    print("="*70)
    
    env = SumoEnv(sumo_cfg="map.sumocfg", use_gui=False)
    model = PPO.load(MODEL_PATH)
    
    obs, info = env.reset()
    
    ppo_metrics = {k: [] for k in ["delay", "queue", "throughput", "stops", "pressure"]}
    
    print("\n[DEBUG] Running 50 steps with metric collection:\n")
    print(f"{'Step':<6} {'Delay':<12} {'Queue':<12} {'Throughput':<12} {'Stops':<12} {'Pressure':<12}")
    print("-" * 78)
    
    try:
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            m = info.get("metrics", {})
            
            delay = m.get("avg_delay", np.nan)
            queue = m.get("avg_queue", np.nan)
            throughput = m.get("throughput_total", np.nan)
            stops = m.get("stop_ratio", np.nan)
            pressure = sum(m.get("queues", [])) if m.get("queues") else np.nan
            
            ppo_metrics["delay"].append(delay)
            ppo_metrics["queue"].append(queue)
            ppo_metrics["throughput"].append(throughput)
            ppo_metrics["stops"].append(stops)
            ppo_metrics["pressure"].append(pressure)
            
            print(f"{step:<6} {delay:<12.4f} {queue:<12.4f} {throughput:<12.4f} {stops:<12.4f} {pressure:<12.4f}")
            
            if terminated or truncated:
                print(f"\n[DEBUG] Episode ended at step {step}")
                break
    
    finally:
        env.close()
    
    # Check for NaN values
    print(f"\n[DEBUG] Metric collection summary:")
    for metric_name, values in ppo_metrics.items():
        values_array = np.array(values)
        nan_count = np.sum(np.isnan(values_array))
        valid_count = len(values_array) - nan_count
        
        if valid_count > 0:
            print(f"  {metric_name}: {valid_count} valid, {nan_count} NaN, mean={np.nanmean(values_array):.4f}")
        else:
            print(f"  {metric_name}: ALL NaN! ❌")

def test_direct_traci_metrics():
    """Test if we can collect metrics directly from TraCI"""
    print("\n" + "="*70)
    print("TESTING DIRECT TRACI METRICS")
    print("="*70)
    
    import traci
    
    env = SumoEnv(sumo_cfg="map.sumocfg", use_gui=False)
    obs, info = env.reset()
    
    print("\n[DEBUG] Checking if traci is loaded...")
    try:
        if traci.isLoaded():
            print("[DEBUG] ✓ TraCI is loaded")
            
            # Try getting vehicle info
            veh_ids = traci.vehicle.getIDList()
            print(f"[DEBUG] Vehicles on network: {len(veh_ids)}")
            
            if veh_ids:
                vid = veh_ids[0]
                print(f"[DEBUG] First vehicle: {vid}")
                print(f"  Speed: {traci.vehicle.getSpeed(vid):.2f}")
                print(f"  Allowed Speed: {traci.vehicle.getAllowedSpeed(vid):.2f}")
        else:
            print("[DEBUG] ✗ TraCI is not loaded")
    except Exception as e:
        print(f"[DEBUG] ✗ TraCI error: {e}")
    
    finally:
        env.close()

def show_env_code():
    """Show the relevant parts of env.py"""
    print("\n" + "="*70)
    print("WHAT TO CHECK IN YOUR env.py")
    print("="*70)
    print("""
In your env.py step() method, make sure you're returning metrics like this:

    def step(self, action):
        # ... do simulation step ...
        
        info = {
            "metrics": {
                "avg_delay": ...,
                "avg_queue": ...,
                "throughput_total": ...,
                "stop_ratio": ...,
                "queues": [...]  # List of queues for pressure calculation
            }
        }
        
        return obs, reward, terminated, truncated, info

If metrics are missing, the comparison script will get NaN values.
    """)

# ==================== MAIN ====================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug PPO metrics collection")
    parser.add_argument("--env-info", action="store_true", help="Check env info dict")
    parser.add_argument("--metric-collection", action="store_true", help="Test metric collection")
    parser.add_argument("--traci", action="store_true", help="Test direct TraCI access")
    parser.add_argument("--show-code", action="store_true", help="Show what to check in env.py")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.all or (not any([args.env_info, args.metric_collection, args.traci, args.show_code])):
        print("\nRunning all debug tests...\n")
        try:
            debug_env_info()
        except Exception as e:
            print(f"[ERROR] env_info failed: {e}")
        
        try:
            debug_metric_collection()
        except Exception as e:
            print(f"[ERROR] metric_collection failed: {e}")
        
        try:
            test_direct_traci_metrics()
        except Exception as e:
            print(f"[ERROR] traci test failed: {e}")
        
        show_env_code()
    else:
        if args.env_info:
            debug_env_info()
        if args.metric_collection:
            debug_metric_collection()
        if args.traci:
            test_direct_traci_metrics()
        if args.show_code:
            show_env_code()

if __name__ == "__main__":
    main()