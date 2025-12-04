#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import time
import io
import os
import json
import argparse
import matplotlib.pyplot as plt

# ======================
# CLI ARGUMENTS
# ======================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--json_file",
    required=False,
    help="Optional JSON config file with plotting parameters"
)
parser.add_argument(
    "--csv_file",
    required=True,
    help="Path to step_log.csv (e.g. runs/<run_name>/step_log.csv)"
)

args = parser.parse_args()

# ======================
# CONFIG
# ======================

PARAMS = {}
if args.json_file is not None and os.path.isfile(args.json_file):
    with open(args.json_file, "r") as f:
        PARAMS = json.load(f)

CSV_FILE = args.csv_file
OUTPUT_PNG = PARAMS.get("output_png", "./figs/step_log_live.png")
PLOT_INTERVAL_SEC = float(PARAMS.get("plot_interval_sec", 10.0))  # time between file reads
PAUSE_SEC = float(PARAMS.get("pause_sec", 5.0))                  # how long plot stays visible
SMOOTH_WINDOW = int(PARAMS.get("smooth_window", 50))             # moving average window for reward

# Make sure figs dir exists if we are saving there
fig_dir = os.path.dirname(OUTPUT_PNG)
if fig_dir:
    os.makedirs(fig_dir, exist_ok=True)


# ======================
# SAFE CSV READER
# ======================

def read_csv_safely(path):
    """
    Safely read step_log.csv while another process may be writing it.

    Expected header:
        global_step,update,step_in_update,reward,action_noisy,action_mean,value

    Returns:
        dict of lists or None if file missing/empty/invalid.
    """
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r") as f:
            lines = f.readlines()
            if not lines:
                return None

        # Parse with csv.reader using in-memory buffer
        buf = io.StringIO("".join(lines))
        reader = csv.reader(buf)

        # Read header
        header = next(reader, None)
        if header is None:
            return None

        # Normalize header (strip spaces)
        header = [h.strip() for h in header]

        # Map expected columns to indices if present
        col_indices = {}
        for name in [
            "global_step",
            "update",
            "step_in_update",
            "reward",
            "action_noisy",
            "action_mean",
            "value",
        ]:
            if name in header:
                col_indices[name] = header.index(name)

        required = ["global_step", "reward", "action_noisy", "action_mean", "value"]
        if not all(name in col_indices for name in required):
            print(f"[WARN] CSV missing required cols, found header: {header}")
            return None

        data = {
            "global_step": [],
            "update": [],
            "step_in_update": [],
            "reward": [],
            "action_noisy": [],
            "action_mean": [],
            "value": [],
        }

        for row in reader:
            if not row:
                continue
            # Skip malformed rows
            if len(row) < len(header):
                continue
            try:
                # Required
                data["global_step"].append(float(row[col_indices["global_step"]]))
                data["reward"].append(float(row[col_indices["reward"]]))
                data["action_noisy"].append(float(row[col_indices["action_noisy"]]))
                data["action_mean"].append(float(row[col_indices["action_mean"]]))
                data["value"].append(float(row[col_indices["value"]]))
                # Optional
                if "update" in col_indices:
                    data["update"].append(int(row[col_indices["update"]]))
                if "step_in_update" in col_indices:
                    data["step_in_update"].append(int(row[col_indices["step_in_update"]]))
            except ValueError:
                # Skip lines that can't be parsed cleanly
                continue

        if len(data["global_step"]) == 0:
            return None

        # If optional lists are shorter, pad with zeros (not really used for plotting)
        if len(data["update"]) < len(data["global_step"]):
            data["update"] += [0] * (len(data["global_step"]) - len(data["update"]))
        if len(data["step_in_update"]) < len(data["global_step"]):
            data["step_in_update"] += [0] * (len(data["global_step"]) - len(data["step_in_update"]))

        return data

    except Exception as e:
        print(f"[read_csv_safely] Warning: {e}")
        return None


def moving_average(values, window):
    """
    Pure-Python moving average.
    Returns a list of same length as 'values'.
    """
    n = len(values)
    if n == 0 or window <= 1:
        return values[:]

    out = []
    cumsum = 0.0
    for i, v in enumerate(values):
        cumsum += v
        if i >= window:
            cumsum -= values[i - window]
            out.append(cumsum / window)
        else:
            out.append(cumsum / (i + 1))
    return out


# ======================
# MAIN LOOP
# ======================

print(f"Watching: {CSV_FILE}")
print(f"Plot refresh every {PLOT_INTERVAL_SEC} s, pause {PAUSE_SEC} s")
print("Press Ctrl+C to stop.")

while True:
    try:
        data = read_csv_safely(CSV_FILE)

        if data is not None:
            steps = data["global_step"]
            rewards = data["reward"]
            a_noisy = data["action_noisy"]
            a_mean = data["action_mean"]
            values = data["value"]

            fig, axs = plt.subplots(4, 1, figsize=(12, 10))

            # --- 1) Reward vs Step ---
            axs[0].plot(steps, rewards, linewidth=1.0, color="black", alpha=0.7, label="Reward")
            if len(rewards) > SMOOTH_WINDOW:
                rew_smooth = moving_average(rewards, SMOOTH_WINDOW)
                axs[0].plot(steps, rew_smooth, "--", linewidth=1.0, color="red", alpha=0.8,
                            label=f"Moving avg ({SMOOTH_WINDOW})")
            axs[0].set_xlabel("Global Step")
            axs[0].set_ylabel("Reward")
            axs[0].set_title("Reward vs Global Step")
            axs[0].grid(True)
            axs[0].legend()

            # --- 2) Reward vs Action (noisy / executed) ---
            axs[1].plot(a_noisy, rewards, "o", markersize=3, alpha=0.2, color="blue",
                        label="Reward vs action_noisy")
            axs[1].set_xlabel("Action (noisy / executed)")
            axs[1].set_ylabel("Reward")
            axs[1].set_title("Reward vs Action (noisy)")
            axs[1].grid(True)
            axs[1].legend()

            # --- 3) Value vs Step ---
            axs[2].plot(steps, values, linewidth=1.0, color="green", alpha=0.8, label="V(s)")
            axs[2].set_xlabel("Global Step")
            axs[2].set_ylabel("Value")
            axs[2].set_title("Value Estimate vs Global Step")
            axs[2].grid(True)
            axs[2].legend()

            # --- 4) Action (mean & noisy) vs Step ---
            axs[3].plot(steps, a_noisy, linewidth=1.0, color="blue", alpha=0.7, label="Action noisy")
            axs[3].plot(steps, a_mean, "--", linewidth=1.0, color="orange", alpha=0.9, label="Action mean")
            axs[3].set_xlabel("Global Step")
            axs[3].set_ylabel("Action")
            axs[3].set_title("Action vs Global Step")
            axs[3].grid(True)
            axs[3].legend()

            plt.tight_layout()
            plt.savefig(OUTPUT_PNG, dpi=200)
            plt.show(block=False)
            plt.pause(PAUSE_SEC)
            plt.close()

        time.sleep(PLOT_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("Stopped.")
        break
