# simulation/edge_simulator.py — Simulates bandwidth delays and device heterogeneity

import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from simulation.environment import get_device_profile


def simulate_transmission_delay(client_id: int, payload_bytes: float,
                                  verbose=False) -> float:
    """
    Estimate transmission time in seconds given device bandwidth profile.
    In a real deployment this would be actual network I/O.
    """
    profile      = get_device_profile(client_id)
    bandwidth_bps = profile["bandwidth_mbps"] * 1e6 / 8   # bytes/sec
    delay_sec    = payload_bytes / bandwidth_bps

    if verbose:
        print(f"  [EdgeSim] Client {client_id} ({profile['name']}): "
              f"{payload_bytes/1024:.1f} KB @ "
              f"{profile['bandwidth_mbps']} Mbps → "
              f"~{delay_sec*1000:.1f} ms simulated delay")
    return delay_sec


def simulate_compute_delay(client_id: int, base_time_sec: float) -> float:
    """Scale training time by device compute factor."""
    factor = get_device_profile(client_id)["compute_factor"]
    return base_time_sec / factor


class EdgeSimulator:
    """Tracks simulated transmission stats across all rounds."""

    def __init__(self):
        self.total_bytes = 0.0
        self.total_delay = 0.0
        self.round_bytes = []

    def record_transmission(self, client_id: int, payload_bytes: float):
        delay = simulate_transmission_delay(client_id, payload_bytes, verbose=True)
        self.total_bytes += payload_bytes
        self.total_delay += delay

    def end_round(self):
        self.round_bytes.append(self.total_bytes)

    def summary(self):
        print(f"\n[EdgeSimulator Summary]")
        print(f"  Total bytes transmitted : {self.total_bytes/1e6:.2f} MB")
        print(f"  Total simulated delay   : {self.total_delay:.2f} s")
        print(f"  Rounds tracked          : {len(self.round_bytes)}\n")
