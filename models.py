import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rate import normalize_power


class DNNBeamformer(nn.Module):
    def __init__(self, k_model: int, n_ant: int, hidden=(512, 1024, 512), dropout=0.0):
        super().__init__()
        self.k_model = k_model
        self.n_ant = n_ant
        in_dim = 2 * k_model * n_ant + k_model
        out_dim = 2 * k_model * n_ant
        layers = []
        last = in_dim
        for width in hidden:
            layers.append(nn.Linear(last, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = width
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def _features(self, H_real, H_imag, mask):
        x_ch = torch.cat([H_real, H_imag], dim=-1).reshape(H_real.shape[0], -1)
        mu = x_ch.mean(dim=1, keepdim=True)
        sd = x_ch.std(dim=1, keepdim=True).clamp_min(1e-6)
        x_ch = (x_ch - mu) / sd
        return torch.cat([x_ch, mask], dim=1)

    def forward(self, H_real, H_imag, mask, total_power: float):
        B = H_real.shape[0]
        y = self.net(self._features(H_real, H_imag, mask))
        y = y.reshape(B, self.k_model, 2, self.n_ant)
        V_real = y[:, :, 0, :]
        V_imag = y[:, :, 1, :]
        return normalize_power(V_real, V_imag, mask, total_power)


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0, rho_init=-5.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        bound = 1.0 / math.sqrt(in_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho_init))
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))
        self.bias_rho = nn.Parameter(torch.full((out_features,), rho_init))

    def forward(self, x, sample=True):
        if sample:
            w_sigma = F.softplus(self.weight_rho)
            b_sigma = F.softplus(self.bias_rho)
            weight = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
            bias = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        # KL[N(mu,sigma^2) || N(0,prior_sigma^2)]
        prior_var = self.prior_sigma ** 2
        w_sigma = F.softplus(self.weight_rho).clamp_min(1e-8)
        b_sigma = F.softplus(self.bias_rho).clamp_min(1e-8)
        kl_w = torch.log(self.prior_sigma / w_sigma) + (w_sigma.pow(2) + self.weight_mu.pow(2)) / (2.0 * prior_var) - 0.5
        kl_b = torch.log(self.prior_sigma / b_sigma) + (b_sigma.pow(2) + self.bias_mu.pow(2)) / (2.0 * prior_var) - 0.5
        return kl_w.sum() + kl_b.sum()


class BNNBeamformer(nn.Module):
    def __init__(self, k_model: int, n_ant: int, hidden=(512, 1024, 512), prior_sigma=1.0, rho_init=-5.0):
        super().__init__()
        self.k_model = k_model
        self.n_ant = n_ant
        in_dim = 2 * k_model * n_ant + k_model
        out_dim = 2 * k_model * n_ant
        dims = [in_dim, *hidden, out_dim]
        self.layers = nn.ModuleList([
            BayesLinear(dims[i], dims[i + 1], prior_sigma=prior_sigma, rho_init=rho_init)
            for i in range(len(dims) - 1)
        ])

    def _features(self, H_real, H_imag, mask):
        x_ch = torch.cat([H_real, H_imag], dim=-1).reshape(H_real.shape[0], -1)
        mu = x_ch.mean(dim=1, keepdim=True)
        sd = x_ch.std(dim=1, keepdim=True).clamp_min(1e-6)
        x_ch = (x_ch - mu) / sd
        return torch.cat([x_ch, mask], dim=1)

    def forward(self, H_real, H_imag, mask, total_power: float, sample=True):
        B = H_real.shape[0]
        h = self._features(H_real, H_imag, mask)
        for layer in self.layers[:-1]:
            h = F.relu(layer(h, sample=sample))
        y = self.layers[-1](h, sample=sample)
        y = y.reshape(B, self.k_model, 2, self.n_ant)
        V_real = y[:, :, 0, :]
        V_imag = y[:, :, 1, :]
        return normalize_power(V_real, V_imag, mask, total_power)

    def kl_divergence(self):
        return sum(layer.kl_divergence() for layer in self.layers)
