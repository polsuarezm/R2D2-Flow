#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import time
import matplotlib.pyplot as plt

# ==== CONFIG ====
CSV_PATH = "runs/HalfCheetah-v4__SHARP_rpo_continuous_action_v3__1__1764766794/step_log.csv"   # <-- CHANGE THIS
REFRESH_SEC = 10.0                             # refresh interval (seconds)


def load_step_log(path):
    """
    Reads step_log.csv without pandas.
    Returns dict of lists containing each column.
    """
    data = {
        "global_step": [],
        "update": [],
        "step_in_update": [],
        "reward": [],
        "action_noisy": [],
        "action_mean": [],
        "value": [],
    }

    try:
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header row, if present

            for row in reader:
                # Skip empty / malformed rows
                if not row or len(row) < 7:
                    continue
                try:
                    data["global_step"].append(float(row[0]))
                    data["update"].append(int(row[1]))
                    data["step_in_update"].append(int(row[2]))
                    data["reward"].append(float(row[3]))
                    data["action_noisy"].append(float(row[4]))
                    data["action_mean"].append(float(row[5]))
                    data["value"].append(float(row[6]))
                except ValueError:
                    # skip bad lines (e.g., partially written)
                    continue
    except FileNotFoundError:
        # If file does not exist yet, just return empty data
        pass

    return data


def moving_average(values, window):
    """
    Simple moving average without numpy/pandas.
    Returns list of the same length as values (prefix is left un-averaged).
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


if __name__ == "__main__":
    print("Live plotting from:", CSV_PATH)
    print("Refresh interval:", REFRESH_SEC, "seconds")
    print("Press Ctrl+C to stop.")

    # Interactive mode
    plt.ion()

    # Create figure with 4 rows, 1 column
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=False)
    ax_reward_step   = axs[0]
    ax_reward_action = axs[1]
    ax_value_step    = axs[2]
    ax_action_step   = axs[3]

    # Show once (non-blocking)
    plt.show()

    try:
        while True:
            data = load_step_log(CSV_PATH)

            steps   = data["global_step"]
            rewards = data["reward"]
            values  = data["value"]
            a_noisy = data["action_noisy"]
            a_mean  = data["action_mean"]

            # Clear all axes
            for ax in axs:
                ax.clear()

            if len(steps) > 0:
                # --- 1) Reward vs Step ---
                ax_reward_step.plot(steps, rewards, linewidth=1)
                # Optional: smooth version
                if len(rewards) > 20:
                    rew_smooth = moving_average(rewards, window=50)
                    ax_reward_step.plot(steps, rew_smooth, linestyle="--", linewidth=1)
                ax_reward_step.set_ylabel("Reward")
                ax_reward_step.set_title("Reward vs Global Step")
                ax_reward_step.grid(True)

                # --- 2) Reward vs Action (noisy / executed) ---
                ax_reward_action.scatter(a_noisy, rewards, s=5, alpha=0.6, edgecolors="none")
                ax_reward_action.set_xlabel("Action (noisy/executed)")
                ax_reward_action.set_ylabel("Reward")
                ax_reward_action.set_title("Reward vs Action (noisy)")
                ax_reward_action.grid(True)

                # --- 3) Value vs Step ---
                ax_value_step.plot(steps, values, linewidth=1)
                ax_value_step.set_ylabel("V(s)")
                ax_value_step.set_title("Value Estimate vs Global Step")
                ax_value_step.grid(True)

                # --- 4) Action (mean & noisy) vs Step ---
                ax_action_step.plot(steps, a_noisy, linewidth=1, label="Action noisy")
                ax_action_step.plot(steps, a_mean, linewidth=1, linestyle="--", label="Action mean")
                ax_action_step.set_xlabel("Global Step")
                ax_action_step.set_ylabel("Action")
                ax_action_step.set_title("Action vs Global Step")
                ax_action_step.legend(loc="best")
                ax_action_step.grid(True)

            else:
                ax_reward_step.set_title("No data yet...")
                for ax in axs:
                    ax.grid(True)

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Small pause so the GUI can update
            time.sleep(REFRESH_SEC)

    except KeyboardInterrupt:
        print("\nStopped live plotting.")

    # Final block so the window doesn't instantly disappear if run from double-click
    plt.ioff()
    plt.show()
