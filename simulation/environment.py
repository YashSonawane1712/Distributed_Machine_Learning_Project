# simulation/environment.py — Edge environment simulation settings

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import NUM_CLIENTS


# Simulated device profiles for heterogeneous edge environment
DEVICE_PROFILES = {
    0: {"name": "Mobile Phone",    "compute_factor": 0.5,  "bandwidth_mbps": 5},
    1: {"name": "Raspberry Pi 4",  "compute_factor": 0.3,  "bandwidth_mbps": 2},
    2: {"name": "Laptop (i5)",     "compute_factor": 1.0,  "bandwidth_mbps": 50},
    3: {"name": "IoT Sensor Hub",  "compute_factor": 0.2,  "bandwidth_mbps": 1},
    4: {"name": "Edge Server",     "compute_factor": 2.0,  "bandwidth_mbps": 100},
}


def get_device_profile(client_id: int) -> dict:
    return DEVICE_PROFILES.get(client_id % len(DEVICE_PROFILES), DEVICE_PROFILES[0])


def print_environment():
    print("\n[Edge Environment] Simulated device types:")
    for cid in range(NUM_CLIENTS):
        p = get_device_profile(cid)
        print(f"  Client {cid}: {p['name']:<20} "
              f"compute={p['compute_factor']}×  "
              f"bandwidth={p['bandwidth_mbps']} Mbps")
    print()
