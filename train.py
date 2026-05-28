import argparse
import json
import os
from dataclasses import asdict

import torch
from config import Config
from data import choose_device, make_batch, set_seed
from models import DNNBeamformer, BNNBeamformer
from rate import weighted_sum_rate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dnn", "bnn"], required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--k_train", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--kl_beta", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.out_dir is not None: cfg.out_dir = args.out_dir
    if args.steps is not None: cfg.train_steps = args.steps
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.k_train is not None: cfg.k_train = args.k_train
    if args.lr is not None: cfg.lr = args.lr
    if args.kl_beta is not None: cfg.kl_beta = args.kl_beta
    if args.device is not None: cfg.device = args.device
    if args.seed is not None: cfg.seed = args.seed

    out_dir = os.path.join(cfg.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    print("device:", device)

    if args.model == "dnn":
        model = DNNBeamformer(cfg.k_model, cfg.n_ant, cfg.hidden, cfg.dropout).to(device)
    else:
        model = BNNBeamformer(cfg.k_model, cfg.n_ant, cfg.hidden, cfg.prior_sigma, cfg.rho_init).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []

    for step in range(1, cfg.train_steps + 1):
        H_real, H_imag, mask, _, _ = make_batch(cfg, cfg.batch_size, cfg.k_train, device)
        opt.zero_grad(set_to_none=True)

        if args.model == "dnn":
            V_real, V_imag = model(H_real, H_imag, mask, cfg.total_power)
            wsr = weighted_sum_rate(H_real, H_imag, V_real, V_imag, mask, cfg.noise_power).mean()
            loss = -wsr
            kl_value = torch.tensor(0.0, device=device)
        else:
            sample_wsrs = []
            for _ in range(cfg.bnn_train_samples):
                V_real, V_imag = model(H_real, H_imag, mask, cfg.total_power, sample=True)
                sample_wsrs.append(weighted_sum_rate(H_real, H_imag, V_real, V_imag, mask, cfg.noise_power).mean())
            wsr = torch.stack(sample_wsrs).mean()
            kl_value = model.kl_divergence()
            anneal = min(1.0, step / max(1, cfg.kl_anneal_steps))
            loss = -wsr + anneal * cfg.kl_beta * kl_value / cfg.batch_size

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.eval_every == 0 or step == 1:
            row = {"step": step, "loss": float(loss.detach().cpu()), "train_wsr": float(wsr.detach().cpu()), "kl": float(kl_value.detach().cpu())}
            history.append(row)
            print(row)

    torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "model_type": args.model}, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("saved:", os.path.join(out_dir, "model.pt"))


if __name__ == "__main__":
    main()
