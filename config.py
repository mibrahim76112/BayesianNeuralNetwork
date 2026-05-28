from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # System
    k_model: int = 8              # fixed input/output user slots
    k_train: int = 4              # train only with this many active users
    n_ant: int = 8                # BS antennas
    total_power: float = 10.0
    noise_power: float = 1.0

    # Channel distribution
    path_loss: bool = False
    path_loss_min_db: float = -5.0
    path_loss_max_db: float = 5.0
    channel_noise_std: float = 0.0
    channel_noise_relative: bool = True
    channel_mean_shift: float = 0.0

    # Training
    batch_size: int = 256
    train_steps: int = 8000
    eval_every: int = 500
    lr: float = 1e-3
    seed: int = 1
    device: str = "auto"          # auto, cpu, cuda, mps

    # Model
    hidden: Tuple[int, ...] = (512, 1024, 512)
    dropout: float = 0.0

    # BNN
    prior_sigma: float = 1.0
    rho_init: float = -5.0        # sigma = softplus(rho), small initial uncertainty
    bnn_train_samples: int = 2    # MC samples per training batch
    bnn_eval_samples: int = 30    # MC samples per test channel batch
    kl_beta: float = 1e-5         # tune: 1e-6, 1e-5, 1e-4 are useful sweeps
    kl_anneal_steps: int = 2000

    # Evaluation
    test_batches: int = 200
    test_batch_size: int = 128
    k_test_min: int = 1
    k_test_max: int = 8
    outage_eta: float = 0.8

    # WMMSE reference
    wmmse_iters: int = 50
    wmmse_epsilon: float = 1e-5
    wmmse_power_tolerance: float = 1e-5

    # Output
    out_dir: str = "runs_bnn_wmmse"
