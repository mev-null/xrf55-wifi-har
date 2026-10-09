import os
import shutil

import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import csv
import time


MODALITIES = ("RFID", "WiFi", "mmWave")


def split_train_test_new(root_ns="./dataset/Raw_dataset/", dst_wr="./dataset/XRF_dataset/", split=14):
    # Only the modalities present under root_ns are split (e.g. WiFi only).
    modalities = [m for m in MODALITIES if os.path.isdir(root_ns + m)]
    if not modalities:
        raise FileNotFoundError("no RFID/, WiFi/ or mmWave/ under " + root_ns)
    # Start from empty output dirs: leftovers from an earlier run (another split or raw_root)
    # would otherwise stay in the other part and put the same sample in both train and test.
    for part in ("train_data/", "test_data/"):
        for m in modalities:
            out = dst_wr + part + m + "/"
            if os.path.isdir(out):
                shutil.rmtree(out)
            os.makedirs(out)

    for file in tqdm(os.listdir(root_ns + modalities[0] + "/")):
        if not file.endswith(".npy"):
            continue
        filename = file.split(".")[0]  # act name
        actidx = int(filename.split("_")[2])
        # out of 20 samples of each action for each person, the first "split" are selected as the training set and the rest as the test set.
        part = "train_data/" if actidx <= split else "test_data/"
        for m in modalities:
            shutil.copy(root_ns + m + "/" + filename + ".npy", dst_wr + part + m + "/" + filename + ".npy")


if __name__ == '__main__':
    from settings import get_settings

    settings = get_settings()
    print(settings)
    split_train_test_new(root_ns=settings.raw_root, dst_wr=settings.split_root, split=14)
