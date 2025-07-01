import matplotlib.pyplot as plt
import pandas as pd
import time
import os

CSV_FILE = "/home/guardiola-pcaux/Documentos/AFC-DRL-experiment/09-test-CTA-DRL/logs/ppo_v1_ACTUATORS_LENTO_20250701-181328"  # <- update to your real path

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Reward")
ax.set_xlabel("Step")
ax.set_ylabel("Reward")
ax.set_title("Live Reward Plot")
ax.grid(True)
ax.legend()

def read_csv_safely(path):
    try:
        df = pd.read_csv(path, header=None, names=["step", "reward", "timestamp"])
        return df
    except Exception:
        return None

while True:
    try:
        df = read_csv_safely(CSV_FILE)
        if df is not None and not df.empty:
            line.set_xdata(df["step"])
            line.set_ydata(df["reward"])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.")
        break

