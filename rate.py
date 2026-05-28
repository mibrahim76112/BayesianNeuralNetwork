import torch


def normalize_power(V_real, V_imag, mask, total_power: float):
    """Mask inactive users and normalize total transmit power per sample."""
    m = mask.unsqueeze(-1)
    V_real = V_real * m
    V_imag = V_imag * m
    power = (V_real.pow(2) + V_imag.pow(2)).sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    scale = (total_power / power).sqrt()
    return V_real * scale, V_imag * scale


def weighted_sum_rate(H_real, H_imag, V_real, V_imag, mask, noise_power: float):
    """
    H: [B,K,M], V: [B,K,M], mask: [B,K]
    Returns per-sample WSR: [B]
    Complex operation: h_k^H v_j.
    """
    H = torch.complex(H_real, H_imag)
    V = torch.complex(V_real, V_imag)

    # hv[b,k,j] = h_{b,k}^H v_{b,j}
    hv = torch.einsum("bkm,bjm->bkj", torch.conj(H), V)
    power_all = hv.abs().pow(2)
    signal = power_all.diagonal(dim1=1, dim2=2)

    beam_mask = mask.unsqueeze(1)       # [B,1,K], active transmitted beams
    user_mask = mask                    # [B,K], active receiving users

    total_rx_power = (power_all * beam_mask).sum(dim=2)
    interference = (total_rx_power - signal).clamp_min(0.0)
    sinr = signal / (interference + noise_power)
    rates = torch.log2(1.0 + sinr) * user_mask
    return rates.sum(dim=1)
