# =========================
# DRL-EXPERIMENTAL KV260
# Convenience Makefile (four-mode flags)
# =========================

SHELL := /bin/bash
JSON ?=
CONF_DIR := conf
RUN := run_training_UDP_debug_v1_20250911.py

# Prefer local venv Python if present; else fall back to system 'python'
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

# -------------------------
# Help
# -------------------------
.PHONY: help
help:
	@echo "DRL-EXPERIMENTAL KV260 — targets:"
	@echo "  make venv                         # Create .venv with Python 3.9+"
	@echo "  make install                      # Install required libraries"
	@echo "  make train-online  JSON=xxx.json  # Online SB3 PPO training"
	@echo "  make infer-online  JSON=xxx.json  # Online PPO inference"
	@echo "  make train-offload JSON=xxx.json  # CRIO executes, Python trains & resends weights"
	@echo "  make infer-offload JSON=xxx.json  # CRIO executes, Python relays weights (no update)"
	@echo "  make sim-online_UDP   JSON=xxx.json     # Start ONLINE UDP simulator (loopback)"
	@echo "  make sim-offload_UDP  JSON=xxx.json     # Start OFFLOADING UDP simulator (loopback)"
	@echo "  make debug-local-online    JSON=xxx.json # Agent + ONLINE simulator"
	@echo "  make debug-local-offloading JSON=xxx.json # Agent + OFFLOADING simulator"
	@echo "  make check-json    JSON=xxx.json  # Validate JSON (fails fast)"
	@echo "  make tail-live                    # Tail csv_log/live_rewards_temp.csv"
	@echo "  make plot         JSON=xxx.json   # Live plot"
	@echo "  make plot-file    JSON=xxx.json CSV=<file>  # Plot a specific CSV in ./csv_log/"
	@echo "  make show-latest-log              # Print latest logs* directory"
	@echo "  make clean-tmp                    # Remove ./csv_log/live_rewards_temp.csv"

# -------------------------
# Environment setup
# -------------------------
.PHONY: venv
venv:
	@test -d .venv || python3 -m venv .venv
	@echo ">>> Activate with: source .venv/bin/activate"

.PHONY: install
install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install \
		stable-baselines3==2.3.2 \
		gymnasium==0.29.1 \
		numpy \
		pandas \
		matplotlib \
		torch --index-url https://download.pytorch.org/whl/cpu

# -------------------------
# Guards
# -------------------------
.PHONY: _require-json
_require-json:
	@if [[ -z "$(JSON)" ]]; then \
		echo "ERROR: pass JSON=<file.json> (must be inside $(CONF_DIR)/)"; \
		exit 1; \
	fi
	@if [[ ! -f "$(CONF_DIR)/$(JSON)" ]]; then \
		echo "ERROR: $(CONF_DIR)/$(JSON) not found."; \
		exit 1; \
	fi

# Strong JSON validation
.PHONY: check-json
check-json: _require-json
	@$(PY) - "$$@" <<'PYCODE'
import json, os, sys
path = os.path.join("$(CONF_DIR)", "$(JSON)")
def fail(msg): print("ERROR:", msg); sys.exit(1)
try:
	d = json.load(open(path, "r"))
except Exception as e:
	fail(f"cannot parse JSON: {e}")
flags = {
	"online_training": bool(d.get("online_training", False)),
	"online_inference": bool(d.get("online_inference", False)),
	"offloading_training": bool(d.get("offloading_training", False)),
	"offload_inference": bool(d.get("offload_inference", False)),
}
true_flags = [k for k,v in flags.items() if v]
if len(true_flags) != 1:
	fail(f"exactly one mode flag must be true; got {flags}")
mode = true_flags[0]
required = [
	"message_type","size_obs_array_per_UDP","total_descarte","total_descarte_used",
	"size_actuator_array","udp_port_send","udp_port_recv","action_min","action_max",
	"episode_length","total_episodes","log_dir_template","crio_ip","hp_ip","debug_ip"
]
for k in required:
	if k not in d: fail(f"missing key '{k}'")
def posint(k, allow_zero=False):
	if not isinstance(d[k], int): fail(f"'{k}' must be int")
	if allow_zero:
		if d[k] < 0: fail(f"'{k}' must be >=0")
	else:
		if d[k] <= 0: fail(f"'{k}' must be >0")
for k in ["size_obs_array_per_UDP","size_actuator_array","episode_length","total_episodes"]:
	posint(k)
posint("total_descarte", allow_zero=True)
posint("total_descarte_used", allow_zero=True)
if d["total_descarte_used"] > d["total_descarte"]:
	fail("'total_descarte_used' cannot exceed 'total_descarte'")
for k in ["udp_port_send","udp_port_recv"]:
	posint(k)
	if not (1 <= d[k] <= 65535): fail(f"'{k}' must be in 1..65535")
if flags["online_inference"]:
	mp = str(d.get("model_path","")).strip()
	if not mp: fail("online_inference requires 'model_path'")
	if not mp.endswith(".zip"): fail("'model_path' must point to an SB3 .zip file")
	if not os.path.exists(mp): fail(f"'model_path' does not exist: {mp}")
print(f"OK: JSON validated. Mode = {mode}")
PYCODE

# -------------------------
# Run modes
# -------------------------
.PHONY: train-online
train-online: check-json
	$(PY) $(RUN) --json_file $(JSON)

.PHONY: infer-online
infer-online: check-json
	$(PY) $(RUN) --json_file $(JSON)

.PHONY: train-offload
train-offload: check-json
	$(PY) $(RUN) --json_file $(JSON)

.PHONY: infer-offload
infer-offload: check-json
	$(PY) $(RUN) --json_file $(JSON)

# -------------------------
# Logs & utilities
# -------------------------
.PHONY: show-latest-log
show-latest-log:
	@latest=$$(ls -d logs* 2>/dev/null | sort -V | tail -n 1); \
	if [[ -z "$$latest" ]]; then echo "No logs* directories yet."; else echo "$$latest"; fi

.PHONY: tail-live
tail-live:
	@mkdir -p csv_log
	@touch csv_log/live_rewards_temp.csv
	@echo "Tailing ./csv_log/live_rewards_temp.csv (Ctrl-C to stop)"
	@tail -f csv_log/live_rewards_temp.csv

.PHONY: clean-tmp
clean-tmp:
	@rm -f csv_log/live_rewards_temp.csv
	@echo "Removed ./csv_log/live_rewards_temp.csv"

# -------------------------
# Plotting
# -------------------------
.PHONY: plot
plot: check-json
	@mkdir -p figs
	$(PY) plot_live.py --json_file $(JSON) --csv_file live_rewards_temp.csv

.PHONY: plot-file
plot-file: check-json
	@mkdir -p figs
	@if [[ -z "$(CSV)" ]]; then echo "Pass CSV=<file>"; exit 1; fi
	$(PY) plot_live.py --json_file $(JSON) --csv_file $(CSV)

# -------------------------
# Local simulators
# -------------------------
.PHONY: sim-online_UDP
sim-online_UDP: check-json
	@echo ">>> Starting local UDP ONLINE simulator with $(JSON)"
	$(PY) simulator_local_online.py --json_file $(JSON)

.PHONY: sim-offload_UDP
sim-offload_UDP: check-json
	@echo ">>> Starting local UDP OFFLOADING simulator with $(JSON)"
	$(PY) simulator_local_offloading.py --json_file $(JSON)

# -------------------------
# Debug runners
# -------------------------
.PHONY: debug-local-online
debug-local-online: check-json
	@echo ">>> Running agent + ONLINE simulator"
	( $(PY) $(RUN) --json_file $(JSON) ) & \
	( $(PY) simulator_local_online.py --json_file $(JSON) ); \
	wait

.PHONY: debug-local-offloading
debug-local-offloading: check-json
	@echo ">>> Running agent + OFFLOADING simulator"
	( $(PY) $(RUN) --json_file $(JSON) ) & \
	( $(PY) simulator_local_offloading.py --json_file $(JSON) ); \
	wait
