#!/usr/bin/env bash
set -e

# Train both models on K_train=4 active users.
python train.py --model dnn --out_dir runs_bnn_wmmse --k_train 4
python train.py --model bnn --out_dir runs_bnn_wmmse --k_train 4

# Evaluate on K_active = 1,...,8 and generate CSV.
python evaluate.py --run_dir runs_bnn_wmmse

# Generate plots.
python plot_results.py --csv runs_bnn_wmmse/eval_active_users.csv --out_dir runs_bnn_wmmse/plots
