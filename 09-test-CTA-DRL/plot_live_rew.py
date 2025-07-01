import matplotlib.pyplot as plt
import time
import os
import pandas as pd

LOG_DIR = "logs/20250701-XXXXXX"  # <-- actualiza esto con el nombre del folder actual
CSV_FILE = os.path.join(LOG_DIR, "live_rewards.csv")

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Reward")
ax.set_xlabel("Step")
ax.set_ylabel("Reward")
ax.set_title("Live Reward Plot")
ax.grid(True)
ax.legend()

xdata, ydata = [], []

def read_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, header=None, names=["step", "reward", "timestamp"])
        return df["step"], df["reward"]
    return [], []

while True:
    try:
        xdata, ydata = read_data()
        line.set_data(xdata, ydata)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.5)  # refresh rate
    except KeyboardInterrupt:
        break

