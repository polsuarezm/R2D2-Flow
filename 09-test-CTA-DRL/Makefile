# =========================
# DRL-EXPERIMENTAL KV260 — Makefile (safe: no tabs required)
# =========================
.RECIPEPREFIX := >

SHELL := /bin/bash
JSON ?=
CONF_DIR := ./
RUN := run_training_UDP_debug_v1_20250911.py
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: help
help:
> echo "DRL-EXPERIMENTAL KV260 — targets:"
> echo "  make venv                         # Create .venv with Python 3.9+"
> echo "  make install                      # Install required libraries"
> echo "  make train-online  JSON=xxx.json  # Online SB3 PPO training"
> echo "  make infer-online  JSON=xxx.json  # Online PPO inference"
> echo "  make train-offload JSON=xxx.json  # CRIO executes, Python trains & resends weights"
> echo "  make infer-offload JSON=xxx.json  # CRIO executes, Python relays weights (no update)"
> echo "  make sim-online_UDP   JSON=xxx.json     # Start ONLINE UDP simulator (loopback)"
> echo "  make sim-offload_UDP  JSON=xxx.json     # Start OFFLOADING UDP simulator (loopback)"
> echo "  make debug-local-online    JSON=xxx.json # Agent + ONLINE simulator"
> echo "  make debug-local-offloading JSON=xxx.json # Agent + OFFLOADING simulator"
> echo "  make check-json    JSON=xxx.json  # Validate JSON (fails fast)"
> echo "  make tail-live                    # Tail csv_log/live_rewards_temp.csv"
> echo "  make plot         JSON=xxx.json   # Live plot"
> echo "  make plot-file    JSON=xxx.json CSV=<file>  # Plot a specific CSV"
> echo "  make show-latest-log              # Print latest logs* directory"
> echo "  make clean-tmp                    # Remove ./csv_log/live_rewards_temp.csv"

.PHONY: venv
venv:
> test -d .venv || python3 -m venv .venv
> echo ">>> Activate with: source .venv/bin/activate"

.PHONY: install
install:
> $(PY) -m pip install --upgrade pip
> $(PY) -m pip install \
> 	stable-baselines3==2.3.2 \
> 	gymnasium==0.29.1 \
> 	numpy \
> 	pandas \
> 	matplotlib \
> 	torch --index-url https://download.pytorch.org/whl/cpu

# -------- Guards & validation --------
.PHONY: _require-json
_require-json:
> if [[ -z "$(JSON)" ]]; then echo "ERROR: pass JSON=<file.json> in $(CONF_DIR)/"; exit 1; fi
> if [[ ! -f "$(CONF_DIR)/$(JSON)" ]]; then echo "ERROR: $(CONF_DIR)/$(JSON) not found"; exit 1; fi

.PHONY: check-json
check-json: _require-json
> $(PY) -c "import json,os,sys; \
p=os.path.join('$(CONF_DIR)','$(JSON)'); d=json.load(open(p)); \
flags={'online_training':bool(d.get('online_training',False)),'online_inference':bool(d.get('online_inference',False)),'offloading_training':bool(d.get('offloading_training',False)),'offload_inference':bool(d.get('offload_inference',False))}; \
ts=[k for k,v in flags.items() if v]; \
(len(ts)==1) or (print('ERROR: exactly one mode flag must be true:',flags) or sys.exit(1)); \
req=['message_type','size_obs_array_per_UDP','total_descarte','total_descarte_used','size_actuator_array','udp_port_send','udp_port_recv','action_min','action_max','episode_length','total_episodes','log_dir_template','crio_ip','hp_ip','debug_ip']; \
all(k in d for k in req) or (print('ERROR: missing keys:',[k for k in req if k not in d]) or sys.exit(1)); \
all(isinstance(d.get(k),int) and d[k]>0 for k in ['size_obs_array_per_UDP','size_actuator_array','episode_length','total_episodes']) or (print('ERROR: size_* and episode_* must be positive ints') or sys.exit(1)); \
all(isinstance(d.get(k),int) and d[k]>=0 for k in ['total_descarte','total_descarte_used']) or (print('ERROR: descarte values must be nonnegative ints') or sys.exit(1)); \
(d['total_descarte_used']<=d['total_descarte']) or (print('ERROR: total_descarte_used > total_descarte') or sys.exit(1)); \
all(isinstance(d.get(k),int) and 1<=d[k]<=65535 for k in ['udp_port_send','udp_port_recv']) or (print('ERROR: ports must be 1..65535') or sys.exit(1)); \
((not flags['online_inference']) or (isinstance(d.get('model_path',''),str) and d['model_path'].endswith('.zip') and os.path.exists(d['model_path']))) or (print('ERROR: online_inference needs existing .zip model_path') or sys.exit(1)); \
print('OK: JSON validated. Mode =',ts[0])"

# -------- Run modes --------
.PHONY: train-online
train-online: check-json
> $(PY) $(RUN) --json_file $(JSON)

.PHONY: infer-online
infer-online: check-json
> $(PY) $(RUN) --json_file $(JSON)

.PHONY: train-offload
train-offload: check-json
> $(PY) $(RUN) --json_file $(JSON)

.PHONY: infer-offload
infer-offload: check-json
> $(PY) $(RUN) --json_file $(JSON)

# -------- Logs & utilities --------
.PHONY: show-latest-log
show-latest-log:
> latest=$$(ls -d logs* 2>/dev/null | sort -V | tail -n 1); \
> if [[ -z "$$latest" ]]; then echo "No logs* directories yet."; else echo "$$latest"; fi

.PHONY: tail-live
tail-live:
> mkdir -p csv_log
> touch csv_log/live_rewards_temp.csv
> echo "Tailing ./csv_log/live_rewards_temp.csv (Ctrl-C to stop)"
> tail -f csv_log/live_rewards_temp.csv

.PHONY: clean-tmp
clean-tmp:
> rm -f csv_log/live_rewards_temp.csv
> echo "Removed ./csv_log/live_rewards_temp.csv"

# -------- Plotting --------
.PHONY: plot
plot: check-json
> mkdir -p figs
> $(PY) plot_live.py --json_file $(JSON) --csv_file live_rewards_temp.csv

.PHONY: plot-file
plot-file: check-json
> mkdir -p figs
> if [[ -z "$(CSV)" ]]; then echo "Pass CSV=<file>"; exit 1; fi
> $(PY) plot_live.py --json_file $(JSON) --csv_file $(CSV)

# -------- Local simulators --------
.PHONY: sim-online_UDP
sim-online_UDP: check-json
> echo ">>> Starting local UDP ONLINE simulator with $(JSON)"
> $(PY) simulator_local_online.py --json_file $(JSON)

.PHONY: sim-offload_UDP
sim-offload_UDP: check-json
> echo ">>> Starting local UDP OFFLOADING simulator with $(JSON)"
> $(PY) simulator_local_offloading.py --json_file $(JSON)

# -------- One-command debug (agent + simulator) --------
.PHONY: debug-local-online
debug-local-online: check-json
> echo ">>> Running agent + ONLINE simulator"
> ( $(PY) $(RUN) --json_file $(JSON) ) & \
> ( $(PY) simulator_local_online.py --json_file $(JSON) ); \
> wait

.PHONY: debug-local-offloading
debug-local-offloading: check-json
> echo ">>> Running agent + OFFLOADING simulator"
> ( $(PY) $(RUN) --json_file $(JSON) ) & \
> ( $(PY) simulator_local_offloading.py --json_file $(JSON) ); \
> wait
