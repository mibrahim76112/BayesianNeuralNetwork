import argparse
import csv
import json
import os

import numpy as np
import torch
from config import Config
from data import choose_device, make_batch, set_seed
from models import DNNBeamformer, BNNBeamformer
from rate import weighted_sum_rate
from wmmse import run_wmmse_single


def load_cfg(path):
    cfg = Config()
    with open(path, "r") as f:
        d = json.load(f)
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_model(model_type, ckpt_path, cfg, device):
    if model_type == "dnn":
        model = DNNBeamformer(cfg.k_model, cfg.n_ant, tuple(cfg.hidden), cfg.dropout).to(device)
    else:
        model = BNNBeamformer(cfg.k_model, cfg.n_ant, tuple(cfg.hidden), cfg.prior_sigma, cfg.rho_init).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="runs_bnn_wmmse")
    p.add_argument("--dnn_ckpt", type=str, default=None)
    p.add_argument("--bnn_ckpt", type=str, default=None)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--bnn_samples", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    dnn_ckpt = args.dnn_ckpt or os.path.join(args.run_dir, "dnn", "model.pt")
    bnn_ckpt = args.bnn_ckpt or os.path.join(args.run_dir, "bnn", "model.pt")
    cfg_path = os.path.join(args.run_dir, "dnn", "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(args.run_dir, "bnn", "config.json")
    cfg = load_cfg(cfg_path)
    if args.device is not None:
        cfg.device = args.device
    if args.bnn_samples is not None:
        cfg.bnn_eval_samples = args.bnn_samples

    out_csv = args.out_csv or os.path.join(args.run_dir, "eval_active_users.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    set_seed(cfg.seed + 1000)
    device = choose_device(cfg.device)
    print("device:", device)
    dnn = load_model("dnn", dnn_ckpt, cfg, device)
    bnn = load_model("bnn", bnn_ckpt, cfg, device)

    rows = []
    for k_active in range(cfg.k_test_min, cfg.k_test_max + 1):
        dnn_rates = []
        bnn_avg_rates = []
        bnn_max_rates = []
        bnn_var_rates = []
        wmmse_rates = []
        dnn_outages = []
        bnn_avg_outages = []
        bnn_max_outages = []

        for _ in range(cfg.test_batches):
            H_real, H_imag, mask, H_np, mask_np = make_batch(cfg, cfg.test_batch_size, k_active, device)

            with torch.no_grad():
                Vd_r, Vd_i = dnn(H_real, H_imag, mask, cfg.total_power)
                rd = weighted_sum_rate(H_real, H_imag, Vd_r, Vd_i, mask, cfg.noise_power)

                sample_rates = []
                for _s in range(cfg.bnn_eval_samples):
                    Vb_r, Vb_i = bnn(H_real, H_imag, mask, cfg.total_power, sample=True)
                    rb = weighted_sum_rate(H_real, H_imag, Vb_r, Vb_i, mask, cfg.noise_power)
                    sample_rates.append(rb)
                sample_rates = torch.stack(sample_rates, dim=0)  # [S,B]
                rb_avg = sample_rates.mean(dim=0)
                rb_max = sample_rates.max(dim=0).values
                rb_var = sample_rates.var(dim=0, unbiased=False)

            rd_np = rd.detach().cpu().numpy()
            rb_avg_np = rb_avg.detach().cpu().numpy()
            rb_max_np = rb_max.detach().cpu().numpy()
            rb_var_np = rb_var.detach().cpu().numpy()

            rw_list = []
            for b in range(cfg.test_batch_size):
                _, rw = run_wmmse_single(H_np[b], mask_np[b], cfg)
                rw_list.append(rw)
            rw_np = np.asarray(rw_list)

            dnn_rates.extend(rd_np.tolist())
            bnn_avg_rates.extend(rb_avg_np.tolist())
            bnn_max_rates.extend(rb_max_np.tolist())
            bnn_var_rates.extend(rb_var_np.tolist())
            wmmse_rates.extend(rw_np.tolist())

            thresh = cfg.outage_eta * rw_np
            dnn_outages.extend((rd_np < thresh).astype(float).tolist())
            bnn_avg_outages.extend((rb_avg_np < thresh).astype(float).tolist())
            bnn_max_outages.extend((rb_max_np < thresh).astype(float).tolist())

        row = {
            "k_active": k_active,
            "wmmse_avg": float(np.mean(wmmse_rates)),
            "dnn_avg": float(np.mean(dnn_rates)),
            "bnn_avg": float(np.mean(bnn_avg_rates)),
            "bnn_max": float(np.mean(bnn_max_rates)),
            "bnn_var": float(np.mean(bnn_var_rates)),
            "dnn_gap": float(np.mean(np.asarray(wmmse_rates) - np.asarray(dnn_rates))),
            "bnn_avg_gap": float(np.mean(np.asarray(wmmse_rates) - np.asarray(bnn_avg_rates))),
            "bnn_max_gap": float(np.mean(np.asarray(wmmse_rates) - np.asarray(bnn_max_rates))),
            "dnn_outage": float(np.mean(dnn_outages)),
            "bnn_avg_outage": float(np.mean(bnn_avg_outages)),
            "bnn_max_outage": float(np.mean(bnn_max_outages)),
        }
        rows.append(row)
        print(row)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
