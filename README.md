# BNN vs DNN for WMMSE Beamforming in PyTorch

This project trains a deterministic DNN and a Bayesian Neural Network (BNN) to output beamforming vectors from multi-user channel inputs.

The main experiment is:

- Fixed model size: `K_model = 8` users.
- Train only on `K_train = 4` active users.
- Test on `K_active = 1,...,8`.
- Compare DNN, BNN average sampled rate, and BNN max sampled rate.
- Plot BNN variance as the uncertainty signal.

## Install

For a CPU-only environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install PyTorch using the command from the official PyTorch selector, then install the remaining packages:

```bash
pip install numpy matplotlib pandas
```

## Quick run

```bash
cd bnn_wmmse_pytorch
bash run_all.sh
```

## Manual run

```bash
python train.py --model dnn --out_dir runs_bnn_wmmse --k_train 4
python train.py --model bnn --out_dir runs_bnn_wmmse --k_train 4
python evaluate.py --run_dir runs_bnn_wmmse
python plot_results.py --csv runs_bnn_wmmse/eval_active_users.csv --out_dir runs_bnn_wmmse/plots
```

## Faster debugging run

```bash
python train.py --model dnn --steps 500 --batch_size 64 --out_dir debug_run --k_train 4
python train.py --model bnn --steps 500 --batch_size 64 --out_dir debug_run --k_train 4 --kl_beta 1e-5
python evaluate.py --run_dir debug_run --bnn_samples 10
python plot_results.py --csv debug_run/eval_active_users.csv --out_dir debug_run/plots
```

## Main outputs

The evaluation CSV contains:

- `wmmse_avg`
- `dnn_avg`
- `bnn_avg`
- `bnn_max`
- `bnn_var`
- gaps to WMMSE
- outage probabilities

The plots are saved in:

```bash
runs_bnn_wmmse/plots
```

## Interpretation

The BNN mean/average does not have to beat DNN. The useful BNN signal is `bnn_var`.

If `bnn_var` increases for `K_active > 4`, then the BNN is detecting active-user distribution shift. If `bnn_max` or outage improves, then the sampled BNN outputs are useful as candidate beamformers.
