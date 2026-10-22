"""Band power features of every split record, saved as <split_root>/{train,test}_data/WiFiBand/<name>.npy.

Log power of the amplitude fluctuation in 6 octave bands (Hann window of 128 steps every 4 steps, window
mean removed, power averaged over the 30 subcarriers) per receiver-antenna pair: (9 pairs x 6 bands, 250).
At 200 Hz the bands are 1.6 / 3.1-4.7 / 6.3-10.9 / 12.5-23.4 / 25-48 / 50-100 Hz.

    python split_train_test.py && python generate_txt.py && python extract_features.py
"""

import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

N_RX, N_SC, N_ANT = 3, 30, 3
N_FFT, HOP = 128, 4
EDGES = (1, 2, 4, 8, 16, 32, 65)  # rfft bins of each band; bin 0 (the window mean) is left out
N_FEATURES = N_RX * N_ANT * (len(EDGES) - 1)  # 54
WINDOW = torch.hann_window(N_FFT, periodic=False, dtype=torch.float64)


@torch.no_grad()
def band_powers(x):
    """(270, T) amplitude -> (54, T // 4) log band power, receiver-antenna pair major, band fastest."""
    x = torch.as_tensor(x, dtype=torch.float64)
    # one window centred on every 4 steps; the record is mirrored at both ends
    pad = N_FFT // 2 - HOP // 2
    frames = F.pad(x[None], (pad, N_FFT - HOP - pad), mode='reflect')[0].unfold(-1, N_FFT, HOP)
    frames = (frames - frames.mean(-1, keepdim=True)) * WINDOW
    power = torch.fft.rfft(frames).abs().pow(2) / WINDOW.pow(2).sum()  # (270, n, bins)
    power = power.reshape(N_RX, N_SC, N_ANT, *power.shape[1:]).mean(1).flatten(0, 1)  # (9, n, bins)
    bands = torch.stack([power[..., lo:hi].sum(-1) for lo, hi in zip(EDGES[:-1], EDGES[1:])], 1)
    return torch.log1p(bands).flatten(0, 1).float().numpy()


def extract(split_root):
    for part in ('train_data/', 'test_data/'):
        src, dst = split_root + part + 'WiFi/', split_root + part + 'WiFiBand/'
        # start empty so that no file is left over from an earlier split
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        for file in tqdm(sorted(f for f in os.listdir(src) if f.endswith('.npy')), desc=part[:-1]):
            np.save(dst + file, band_powers(np.load(src + file)))


if __name__ == '__main__':
    from settings import get_settings

    extract(get_settings().split_root)
