import numpy as np
import torch
from config import Config


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def generate_complex_channels_np(cfg: Config, batch_size: int, k_active: int):
    """
    Returns:
        H_full: complex ndarray [B, K_model, M]
        mask: float ndarray [B, K_model]
    Only the first k_active users are active. Others are zero padded.
    """
    B, K, M = batch_size, cfg.k_model, cfg.n_ant
    H = np.zeros((B, K, M), dtype=np.complex128)
    mask = np.zeros((B, K), dtype=np.float64)
    mask[:, :k_active] = 1.0

    for b in range(B):
        for k in range(k_active):
            path_loss_db = 0.0
            if cfg.path_loss:
                path_loss_db = np.random.uniform(cfg.path_loss_min_db, cfg.path_loss_max_db)
            scale = np.sqrt(10.0 ** (path_loss_db / 10.0)) * np.sqrt(0.5)
            h = scale * (np.random.randn(M) + 1j * np.random.randn(M))

            if cfg.channel_mean_shift != 0.0:
                h = h + (cfg.channel_mean_shift + 1j * cfg.channel_mean_shift)

            if cfg.channel_noise_std > 0.0:
                if cfg.channel_noise_relative:
                    rms = np.sqrt(np.mean(np.abs(h) ** 2) + 1e-12)
                    sigma = cfg.channel_noise_std * rms
                else:
                    sigma = cfg.channel_noise_std
                e = (sigma / np.sqrt(2.0)) * (np.random.randn(M) + 1j * np.random.randn(M))
                h = h + e

            H[b, k, :] = h

    return H, mask


def make_batch(cfg: Config, batch_size: int, k_active: int, device: torch.device):
    H_np, mask_np = generate_complex_channels_np(cfg, batch_size, k_active)
    H_real = torch.tensor(H_np.real, dtype=torch.float32, device=device)
    H_imag = torch.tensor(H_np.imag, dtype=torch.float32, device=device)
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    return H_real, H_imag, mask, H_np, mask_np
