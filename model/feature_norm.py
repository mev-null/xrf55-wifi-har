import torch
import torch.nn as nn


class FeatureNorm(nn.Module):
    """Standardises each feature channel with train-set statistics (set by wifi_train.py through fit()).
    They are buffers, so a saved model carries them."""

    def __init__(self, channels):
        super().__init__()
        self.register_buffer('mean', torch.zeros(channels, 1))
        self.register_buffer('std', torch.ones(channels, 1))

    def fit(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float)[:, None])
        self.std.copy_(torch.as_tensor(std, dtype=torch.float)[:, None])

    def forward(self, f):
        return (f - self.mean) / self.std
