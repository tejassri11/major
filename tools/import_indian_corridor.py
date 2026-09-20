import os
import sys
import argparse
import subprocess
import urllib.request
import sumolib

PRESETS = {
    "bangalore_orr": {
        "name": "Bengaluru Outer Ring Road (Silk Board to HSR)",
        "bbox": (12.9150, 77.6200, 12.9250, 77.6350),
    },
    "delhi_ring_road": {
        "name": "Delhi Ring Road (AIIMS to Moolchand)",
        "bbox": (28.5650, 77.2100, 28.5750, 77.2350),
    },
    "pune_nagar_road": {
        "name": "Pune Nagar Road (Viman Nagar to Yerwada)",
        "bbox": (18.5500, 73.8900, 18.5650, 73.9150),
    },
}

def download_osm_bbox(bbox, output_osm):
    min_lat, min_lon, max_lat, max_lon = bbox
    overpass_url = f"https://overpass-api.de/api/map?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
    print(f"[OSM DOWNLOAD] Fetching corridor bbox {bbox} from Overpass API...")
    req = urllib.request.Request(overpass_url, headers={"User-Agent": "SUMO-Indian-Corridor-Builder/1.0"})
    with urllib.request.urlopen(req) as resp, open(output_osm, "wb") as f:
        f.write(resp.read())
    print(f"[OSM DOWNLOAD] Saved OSM raw data to {output_osm}")

def convert_osm_to_sumo(osm_file, output_net):
    netconvert_bin = sumolib.checkBinary("netconvert")
    print(f"[NETCONVERT] Running {netconvert_bin} for Left-Hand Drive Indian network...")
    cmd = [
        netconvert_bin,
        "--osm-files", osm_file,
        "-o", output_net,
        "--lefthand",
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals", "true",
        "--tls.discard-simple", "true",
        "--no-warnings", "true",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] netconvert failed:\n{res.stderr}")
        return False
    print(f"[NETCONVERT SUCCESS] Generated Indian corridor network: {output_net}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Import and build Indian corridor SUMO network")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Select pre-configured Indian corridor")
    parser.add_argument("--osm", help="Path to existing local .osm file")
    parser.add_argument("-o", "--output", default="network/indian_corridor.net.xml", help="Target .net.xml output path")
    args = parser.parse_args()

    os.makedirs("network", exist_ok=True)
    osm_path = args.osm

    if args.preset:
        preset_data = PRESETS[args.preset]
        print(f"[PRESET] Selected: {preset_data['name']}")
        osm_path = f"network/{args.preset}.osm"
        download_osm_bbox(preset_data["bbox"], osm_path)

    if not osm_path or not os.path.isfile(osm_path):
        print("[ERROR] Please provide a valid --preset or --osm file path.")
        sys.exit(1)

    success = convert_osm_to_sumo(osm_path, args.output)
    if success:
        print(f"\n[DONE] You can inspect the new corridor in Netedit:")
        print(f"  netedit {args.output}")

if __name__ == "__main__":
    main()
