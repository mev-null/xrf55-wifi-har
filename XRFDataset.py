import logging

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

log = logging.getLogger(__name__)


class XRFBertDatasetNewMix(Dataset):
    def __init__(self, file_path='./dataset/XRFDataset/', is_train=True, scene='dml'):
        super(XRFBertDatasetNewMix, self).__init__()
        self.word_list = np.load("./word2vec/bert_new_sentence_large_uncased.npy")
        self.file_path = file_path
        self.is_train = is_train
        self.scene = scene
        if self.is_train:
            self.file = self.file_path + self.scene + '_train.txt'
        else:
            self.file = self.file_path + self.scene + '_val.txt'
        file = open(self.file)
        val_list = file.readlines()
        self.data = {'file_name': list(), 'label': list()}
        self.path = self.file_path + self.scene + '_new_data/'
        for string in val_list:
            self.data['file_name'].append(string.split(',')[0])
            self.data['label'].append(int(string.split(',')[2]) - 1)
        log.info("load XRF dataset")

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        file_name = self.data['file_name'][idx]
        label = self.data['label'][idx]
        vector = self.word_list[label]

        wifi_data = load_wifi(file_name, self.is_train, path=self.path)
        rfid_data = load_rfid(file_name, self.is_train, path=self.path)
        mmwave_data = load_mmwave(file_name, self.is_train, path=self.path)
        return wifi_data, rfid_data, mmwave_data, label, vector


class XRFWiFiDataset(Dataset):
    """WiFi-only variant of XRFBertDatasetNewMix: same txt lists and files, no RFID/mmWave/BERT.

    subjects: keep only these subject IDs (None = all). The 14/6 split per subject is kept
    because it comes from split_train_test.py.
    features: also return the band power features (extract_features.py): ((amplitude, features), label).
    """

    def __init__(self, file_path=None, is_train=True, scene='dml', subjects=None, features=False):
        super(XRFWiFiDataset, self).__init__()
        if file_path is None:
            from settings import get_settings

            file_path = get_settings().split_root
        self.file_path = file_path
        self.is_train = is_train
        self.features = features
        list_file = file_path + scene + ('_train.txt' if is_train else '_val.txt')
        self.data = {'file_name': list(), 'subject': list(), 'label': list()}
        for string in open(list_file).readlines():
            name, vol, act = string.strip().split(',')
            if subjects is not None and int(vol) not in subjects:
                continue
            self.data['file_name'].append(name)
            self.data['subject'].append(int(vol))
            self.data['label'].append(int(act) - 1)

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        x = load_wifi(self.data['file_name'][idx], self.is_train, path=self.file_path)
        if self.features:
            x = (x, self.load_features(idx))
        return x, self.data['label'][idx]

    def load_features(self, idx):
        part = 'train_data/' if self.is_train else 'test_data/'
        name = self.data['file_name'][idx]
        return torch.from_numpy(np.load(self.file_path + part + 'WiFiBand/' + name + '.npy'))


def load_rfid(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'RFID/' + filename + ".npy")
    return torch.from_numpy(record).float()


def load_wifi(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'WiFi/' + filename + ".npy")
    return torch.from_numpy(record).float()


def load_mmwave(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    mmWave_data = np.load(path + 'mmWave/' + filename + ".npy")
    return torch.from_numpy(mmWave_data).float()
