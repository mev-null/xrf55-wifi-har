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

def generate_txt_mix(data_src_path, txt_save_path, txt_name):

    train_list = []
    val_list = []

    # Use the first modality that split_train_test.py produced (RFID in the original 3-modality setup).
    modality = next(m for m in ("RFID", "WiFi", "mmWave") if os.path.isdir(data_src_path + "train_data/" + m))
    train_src = data_src_path + "train_data/" + modality + "/"
    test_src = data_src_path + "test_data/" + modality + "/"

    for file in os.listdir(train_src):
        filename = file.split(".")[0]
        fulname = filename.split("_")  # vol, act, idx.txt/.mat
        vol_index = fulname[0]
        act_index = fulname[1]
        train_list.append(filename + "," + vol_index + "," + act_index + "\n")
    for file in os.listdir(test_src):
        filename = file.split(".")[0]
        fulname = filename.split("_")  # vol, act, idx.txt/.mat
        vol_index = fulname[0]
        act_index = fulname[1]
        val_list.append(filename + "," + vol_index + "," + act_index + "\n")
    #
    with open(txt_save_path + txt_name + "_train.txt", "w") as f:
        for i in range(len(train_list)):
            f.write(train_list[i])
        print("train_dataset len：" + str(len(train_list)))
    with open(txt_save_path + txt_name + "_val.txt", "w") as f:
        for i in range(len(val_list)):
            f.write(val_list[i])
        print("test_dataset len：" + str(len(val_list)))


if __name__ == '__main__':
    from settings import get_settings

    split_root = get_settings().split_root
    generate_txt_mix(split_root, split_root, "dml")
