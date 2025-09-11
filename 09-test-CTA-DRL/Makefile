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
	@echo "  make venv                 # Create .venv with Python 3.9+"
	@echo "  make install              # Install required libraries into the active Python"
	@echo "  make train-online  JSON=xxx.json   # Online SB3 PPO training (JSON must enable online_training)"
	@echo "  make infer-online  JSON=xxx.json   # Online PPO inference (JSON must enable online_inference)"
	@echo "  make train-offload JSON=xxx.json   # CRIO executes, Python trains (offloading_training)"
	@echo "  make infer-offload JSON=xxx.json   # CRIO executes, Python does NOT train (offload_inference)"
	@echo "  make check-mode    JSON=xxx.json   # Validate JSON: exactly one mode flag is true"
	@echo "  make tail-live            # Tail csv_log/live_rewards_temp.csv"
	@echo "  make plot         JSON=xxx.json    # Live plot from csv_log/live_rewards_temp.csv"
	@echo "  make plot-file    JSON=xxx.json CSV=<file>  # Plot a specific CSV in ./csv_log/"
	@echo "  make show-latest-log      # Print latest logs* directory"
	@echo "  make clean-tmp            # Remove ./csv_log/live_rewards_temp.csv"
	@echo "  make sim          JSON=xxx.json    # Run local UDP simulator"
	@echo "  make debug-local  JSON=xxx.json    # Run agent + simulator together"

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
		echo "ERROR: please pass JSON=<file.json> (file must be inside $(CONF_DIR)/)"; \
		exit 1; \
	fi
	@if [[ ! -f "$(CONF_DIR)/$(JSON)" ]]; then \
		echo "ERROR: $(CONF_DIR)/$(JSON) not found."; \
		exit 1; \
	fi

# Validate exactly one mode flag is true (pure Python -c, no heredoc)
.PHONY: check-mode
check-mode: _require-json
	@$(PY) -c "import json,sys,os; \
p=os.path.join('$(CONF_DIR)','$(JSON)'); \
d=json.load(open(p)); \
fs={'online_training':bool(d.get('online_training',False)), \
    'online_inference':bool(d.get('online_inference',False)), \
    'offloading_training':bool(d.get('offloading_training',False)), \
    'offload_inference':bool(d.get('offload_inference',False))}; \
ts=[k for k,v in fs.items() if v]; \
print('OK: mode =',ts[0]) if len(ts)==1 else (print('ERROR: exactly one flag must be true. Current:',fs), sys.exit(1))"

# -------------------------
# Run modes (JSON controls the actual mode)
# -------------------------
.PHONY: train-online
train-online: _require-json check-mode
	$(PY) $(RUN) --json_file $(JSON)

.PHONY: infer-online
infer-online: _require-json check-mode
	$(PY) $(RUN) --json_file $(JSON)

.PHONY: train-offload
train-offload: _require-json check-mode
	$(PY) $(RUN) --json_file $(JSON)

.PHONY: infer-offload
infer-offload: _require-json check-mode
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
# Plotting helpers
# -------------------------
.PHONY: plot
plot: _require-json
	@mkdir -p figs
	$(PY) plot_live.py --json_file $(JSON) --csv_file live_rewards_temp.csv

.PHONY: plot-file
plot-file: _require-json
	@mkdir -p figs
	@if [[ -z "$(CSV)" ]]; then echo "Pass CSV=<file> (must exist in ./csv_log/)"; exit 1; fi
	$(PY) plot_live.py --json_file $(JSON) --csv_file $(CSV)

# -------------------------
# Local debugging simulator
# -------------------------
.PHONY: sim
sim: _require-json
	@echo ">>> Starting local UDP simulator with $(JSON)"
	$(PY) simulator_local.py --json_file $(JSON)

.PHONY: debug-local
debug-local: _require-json
	@echo ">>> Running agent and local UDP simulator together"
	( $(PY) $(RUN) --json_file $(JSON) ) & \
	( $(PY) simulator_local.py --json_file $(JSON) )
