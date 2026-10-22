"""THAT for the XRF55 Wi-Fi input, implemented from the paper:
B. Li et al., "Two-Stream Convolution Augmented Transformer for Human Activity Recognition", AAAI 2021.

Input is (batch, 270, 1000): channels (receiver 3 x subcarrier 30 x antenna 3, antenna fastest) x time.

    time stream     250 tokens of 270 values (average of 4 time steps), Gaussian range encoding,
                    5 layers, multi-scale CNN (kernels 1, 3, 5) in place of the feed-forward layer
    channel stream  90 tokens (receiver, subcarrier; antennas averaged) of 250 values, 1 layer,
                    multi-scale CNN (kernels 1, 2, 3) along the 30 subcarriers of each receiver

Three details follow the authors' public code (github.com/windofshadow/THAT) instead of the paper:
the Gaussian means start at 0 (paper: 25, 75, ...), sigma starts at 50 and is used as the variance
(paper: 8), and the multi-scale CNN averages its kernels (paper: adaptive scale attention).
Also different: 55 classes, and forward() returns logits.
n_features > 0 (not in THAT) adds feature channels (extract_features.py, one frame per time token) to
the time tokens: 270 + n_features values each. The channel stream is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_norm import FeatureNorm

N_RX, N_SC, N_ANT, N_TIME = 3, 30, 3, 1000
N_CH = N_RX * N_SC * N_ANT


class GaussianPosition(nn.Module):
    """Gaussian range encoding: each position is a mixture of K learnable vectors, weighted by
    K Gaussians over the token index with learnable mean and variance."""

    def __init__(self, d_model, n_tokens, K=10, sigma=50.0):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(K, d_model))
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.register_buffer('positions', torch.arange(n_tokens, dtype=torch.float).unsqueeze(1))
        self.mu = nn.Parameter(torch.arange(K, dtype=torch.float).unsqueeze(0) * n_tokens / K)
        self.sigma = nn.Parameter(torch.full((1, K), sigma))

    def forward(self, x):
        a = self.positions - self.mu
        log_p = -a * a / (2 * self.sigma) - torch.log(self.sigma) / 2
        return x + F.softmax(log_p, dim=1) @ self.embedding


class MultiScaleCNN(nn.Module):
    """THAT's replacement for the feed-forward layer: convolutions over neighbouring tokens."""

    def __init__(self, d_model, kernels=(1, 3, 5), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(d_model, d_model, k, padding='same') for k in kernels)
        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = sum(F.relu(self.dropout(self.bn(conv(x)))) for conv in self.convs) / len(self.convs)
        return out.transpose(1, 2)


class ReceiverMCAT(nn.Module):
    """Multi-scale CNN along the subcarriers of each receiver: (b, 90, d) is run as 3 sequences of 30
    with shared weights, so the convolutions never mix receivers."""

    def __init__(self, d_model, kernels=(1, 2, 3), dropout=0.1):
        super().__init__()
        self.cnn = MultiScaleCNN(d_model, kernels, dropout)

    def forward(self, x):
        b, m, d = x.shape
        return self.cnn(x.reshape(b * N_RX, m // N_RX, d)).reshape(b, m, d)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, feed_forward, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.dropout(self.attn(y, y, y, need_weights=False)[0])
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class Encoder(nn.Module):
    def __init__(self, d_model, layers, heads, feed_forward):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(d_model, heads, feed_forward(d_model)) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Aggregate(nn.Module):
    """CNN over the tokens, then the maximum over tokens: one vector per record."""

    def __init__(self, d_model, filters, kernels):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(d_model, filters, k) for k in kernels)

    def forward(self, x):
        x = x.transpose(1, 2)
        return torch.cat([F.relu(conv(x)).amax(-1) for conv in self.convs], 1)


class THAT(nn.Module):
    def __init__(self, num_classes=55, k=4, K=10, t_layers=5, t_heads=9, c_layers=1, c_heads=10, n_features=0):
        super().__init__()
        self.k = k
        n = N_TIME // k  # number of time tokens = size of a channel token
        d_time = N_CH + n_features  # values per time token
        if d_time % t_heads:
            raise ValueError(f'{d_time} values per time token cannot be split into {t_heads} heads')
        self.feature_norm = FeatureNorm(n_features) if n_features else None
        self.pos = GaussianPosition(d_time, n, K)
        self.temporal = Encoder(d_time, t_layers, t_heads, MultiScaleCNN)
        self.channel = Encoder(n, c_layers, c_heads, ReceiverMCAT)
        self.temporal_agg = Aggregate(d_time, 128, (10, 40))
        self.channel_agg = Aggregate(n, 16, (2, 4))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 2 + 16 * 2, num_classes)

    def forward(self, x, f=None):
        b = x.size(0)
        avg = F.avg_pool1d(x, self.k)  # (b, 270, n)
        t = avg if self.feature_norm is None else torch.cat([avg, self.feature_norm(f)], 1)
        t = self.temporal(self.pos(t.transpose(1, 2)))  # (b, n, 270 + n_features)
        c = avg.reshape(b, N_RX, N_SC, N_ANT, -1).mean(3).reshape(b, N_RX * N_SC, -1)  # (b, 90, n)
        c = self.channel(c)
        feature = torch.cat([self.temporal_agg(t), self.channel_agg(c)], 1)
        return self.fc(self.dropout(feature))


def that(n_features=0):
    return THAT(n_features=n_features)
