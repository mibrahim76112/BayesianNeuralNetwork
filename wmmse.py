import numpy as np
from config import Config


def compute_wsr_np(H, V, mask, noise_power):
    K = H.shape[0]
    active = np.where(mask > 0.5)[0].tolist()
    wsr = 0.0
    for k in active:
        desired = np.abs(np.vdot(H[k, :], V[k, :])) ** 2
        interf = 0.0
        for j in active:
            if j != k:
                interf += np.abs(np.vdot(H[k, :], V[j, :])) ** 2
        sinr = desired / (noise_power + interf)
        wsr += np.log2(1.0 + sinr)
    return float(wsr)


def run_wmmse_single(H_full, mask, cfg: Config):
    """Classical WMMSE on active users, returned as full [K_model,M] precoder."""
    active = np.where(mask > 0.5)[0]
    K_full, M = H_full.shape
    K = len(active)
    V_full = np.zeros((K_full, M), dtype=np.complex128)
    if K == 0:
        return V_full, 0.0

    H = H_full[active, :]
    user_weights = np.ones(K, dtype=np.float64)

    u = np.zeros(K, dtype=np.complex128)
    w = np.ones(K, dtype=np.float64)
    V = H.copy()
    V = V * np.sqrt(cfg.total_power) / (np.linalg.norm(V) + 1e-12)

    prev = -np.inf
    eye = np.eye(M, dtype=np.complex128)

    for _ in range(cfg.wmmse_iters):
        for i in range(K):
            interf_plus_noise = cfg.noise_power
            for j in range(K):
                interf_plus_noise += np.abs(np.vdot(H[i, :], V[j, :])) ** 2
            desired = np.vdot(H[i, :], V[i, :])
            u[i] = desired / (interf_plus_noise + 1e-12)
            mse = 1.0 - (np.abs(desired) ** 2) / (interf_plus_noise + 1e-12)
            w[i] = 1.0 / max(mse, 1e-12)

        A = np.zeros((M, M), dtype=np.complex128)
        B = np.zeros((M, K), dtype=np.complex128)
        for i in range(K):
            hi = H[i, :].reshape(-1, 1)
            A += user_weights[i] * w[i] * (np.abs(u[i]) ** 2) * (hi @ hi.conj().T)
            B[:, i:i+1] = user_weights[i] * w[i] * np.conj(u[i]) * hi

        mu_low, mu_high = 0.0, 1.0
        for _ in range(50):
            V_try = np.linalg.solve(A + mu_high * eye, B)
            if np.linalg.norm(V_try) ** 2 <= cfg.total_power:
                break
            mu_high *= 2.0

        V_best = None
        for _ in range(60):
            mu = 0.5 * (mu_low + mu_high)
            V_mid = np.linalg.solve(A + mu * eye, B)
            p = np.linalg.norm(V_mid) ** 2
            V_best = V_mid
            if abs(p - cfg.total_power) <= cfg.wmmse_power_tolerance:
                break
            if p > cfg.total_power:
                mu_low = mu
            else:
                mu_high = mu

        V = V_best.T
        wsr = compute_wsr_np(H, V, np.ones(K), cfg.noise_power)
        if abs(wsr - prev) < cfg.wmmse_epsilon:
            break
        prev = wsr

    V_full[active, :] = V
    wsr_full = compute_wsr_np(H_full, V_full, mask, cfg.noise_power)
    return V_full, wsr_full
