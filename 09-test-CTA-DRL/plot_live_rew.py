import matplotlib.pyplot as plt
import pandas as pd
import time
import io
import os
import json
import numpy as np

# === Configuration ===
#CSV_FILE = "/home/guardiola-pcaux/Documentos/AFC-DRL-experiment/09-test-CTA-DRL/logs_v1/PPO_V1noUDP_20250703-1717/live_rewards.csv"
#CSV_FILE = "/scratch/polsm/011-DRL-experimental/AFC-DRL-experiment-v3/09-test-CTA-DRL/logs_debug_eval/model_PPO_20250703-1841/live_rewards.csv"
with open("input_parameters_v1_20250909.json", "r") as f:
    PARAMS = json.load(f)

CSV_FILE = "./logfile-tanda1-A032.csv"

OUTPUT_PNG = "last_reward_plot_debugeval.png"
PLOT_INTERVAL_SEC = 1.0
EVAL_FREQ=PARAMS.get("eval_freq", 5000)
EPS_LENGTH = PARAMS.get("episode_length", 100)
N_EVAL_EPS = PARAMS.get("n_eval_episodes", 1)

# === Safe CSV reader for parallel writes ===
def read_csv_safely(path):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
            if not lines:
                return None
        df = pd.read_csv(io.StringIO("".join(lines)), header=None, names=["step", "reward", "action", "timestamp", "obs0", "obs1", "obs2", "obs3"])
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
            axs[0].plot(df["step"], df["reward"], label="Reward", linewidth=1.5, color='black', marker='o', markersize=3, alpha=0.1)
            axs[0].set_xlabel("Step")
            axs[0].set_ylabel("Reward")
            axs[0].set_title("Reward vs Step")
            axs[0].set_ylim(-0.5, 1.25)
            axs[0].grid(True)
            axs[0].legend()
            axs[1].plot(df["step"], df["action"], 'o', label="action vs step", markersize=3, alpha=0.1, color='blue')
            x_max = df["step"].max()

            # Create a boolean mask for evaluation intervals
            in_eval = np.full(len(df), False)

            for x in range(0, x_max + 1, EVAL_FREQ):
                start = x
                end = x + EPS_LENGTH * N_EVAL_EPS
                in_eval |= (df["step"] >= start) & (df["step"] <= end)

            # Plot actions outside eval in blue
            axs[1].plot(df["step"][~in_eval], df["action"][~in_eval], 'o',
                        label="train", markersize=3, alpha=0.01, color='blue')

            # Plot actions during eval in black
            axs[1].plot(df["step"][in_eval], df["action"][in_eval], 'o',
                        label="eval", markersize=3, alpha=0.05, color='black')

            # Add static horizontal/vertical lines for context
            axs[1].axhline(-0.8, color='black', linestyle='--', linewidth=10, alpha=0.1, label="reward = -0.8")
            #axs[1].axvline(5, color='black', linestyle='--', linewidth=10, alpha=0.2)

            # Add eval interval markers
            for x in range(0, x_max + 1, EVAL_FREQ):
                axs[1].axvline(x, color='black', linestyle='-', linewidth=0.5, alpha=0.8)
                #axs[1].axvline(x + EPS_LENGTH * N_EVAL_EPS, color='black', linestyle='-', linewidth=0.5, alpha=0.8)

            # Labeling and formatting
            axs[1].set_xlabel("step")
            axs[1].set_xlim(0, df["step"].max())
            axs[1].set_ylabel("action")
            axs[1].set_title("action vs step")
            axs[1].grid(True)
            axs[3].plot(df["obs2"], df["obs3"], 'o', label="obs3 vs obs2", markersize=3, alpha=0.05, color='red')
            axs[3].set_xlabel("obs[2]")
            axs[3].set_ylabel("obs[3]")
            axs[3].set_title("obs[3] vs obs[2]")
            axs[3].grid(True)
            axs[3].legend()
            axs[2].plot(df["action"], df["reward"], 'o', label="action vs obs2", markersize=2, alpha=0.05, color='green')
            axs[2].set_xlabel("action")
            axs[2].set_ylabel("reward")
            axs[2].set_title("action vs reward")
            axs[2].set_xlim(-1.5, 1.5)
            #axs[3].set_ylim(-0.5, 1)
            axs[2].grid(True)
            axs[2].legend()

            plt.tight_layout()
            plt.savefig(OUTPUT_PNG, dpi=800)
            plt.show()
        time.sleep(PLOT_INTERVAL_SEC)
        #plt.close()
    except KeyboardInterrupt:
        print("Stopped.")
        break
