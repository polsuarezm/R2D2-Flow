import matplotlib.pyplot as plt
import pandas as pd
import time
import io
import os

# === Configuration ===
CSV_FILE = "/home/guardiola-pcaux/Documentos/AFC-DRL-experiment/09-test-CTA-DRL/logs_v1/PPO_V1noUDP_20250703-1424/live_rewards.csv"
OUTPUT_PNG = "last_reward_plot.png"
PLOT_INTERVAL_SEC = 1.0

# === Safe CSV reader for parallel writes ===
def read_csv_safely(path):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
            if not lines:
                return None
        df = pd.read_csv(io.StringIO("".join(lines)), header=None, names=["step", "obs0", "obs1", "obs2", "obs3", "reward", "action", "timestamp"])
        return df
    except Exception as e:
        print(f"[read_csv_safely] Warning: {e}")
        return None

# === Main loop ===
print(f"Watching: {CSV_FILE}")
while True:
    try:
        df = read_csv_safely(CSV_FILE)
        if df is not None and not df.empty:
            fig, axs = plt.subplots(4, 1, figsize=(13, 10))  # 1 row, 2 columns

            # Plot 1: Reward vs Step
            axs[0].plot(df["step"], df["reward"], label="Reward", linewidth=1.5, color='black', marker='o', markersize=1, alpha=0.2)
            axs[0].set_xlabel("Step")
            axs[0].set_ylabel("Reward")
            axs[0].set_title("Reward vs Step")
            axs[0].set_ylim(-1.5, 1.5)
            axs[0].grid(True)
            axs[0].legend()
            axs[1].plot(df["step"], df["action"], 'o', label="action vs step", markersize=2, alpha=0.8, color='blue')
            axs[1].set_xlabel("step")
            axs[1].set_ylabel("action")
            axs[1].set_title("action vs step")
            axs[1].grid(True)
            axs[1].legend()
            axs[2].plot(df["obs2"], df["obs3"], 'o', label="obs3 vs obs2", markersize=3, alpha=0.05, color='red')
            axs[2].set_xlabel("obs[2]")
            axs[2].set_ylabel("obs[3]")
            axs[2].set_title("obs[3] vs obs[2]")
            axs[2].grid(True)
            axs[2].legend()
            axs[3].plot(df["action"], df["reward"], 'o', label="action vs obs2", markersize=2, alpha=0.3, color='green')
            axs[3].set_xlabel("action")
            axs[3].set_ylabel("reward")
            axs[3].set_title("action vs reward")
            axs[3].set_xlim(-1.1, 1.1)
            axs[3].set_ylim(-0.5, 1)
            axs[3].grid(True)
            axs[3].legend()

            plt.tight_layout()
            plt.savefig(OUTPUT_PNG)
            plt.close()
        #time.sleep(PLOT_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("Stopped.")
        break
