"""Data and output locations. Only the raw data location changes per machine."""

import os

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'


class Settings:
    # split_train_test.py / generate_txt.py output (not committed)
    split_root = ROOT + 'dataset/XRF_dataset/'
    # wifi_train.py output: <group>/seed<k>/ (committed)
    result_root = ROOT + 'result/wifi/'

    def __init__(self):
        # Downloaded XRF55 data: WiFi/, RFID/, mmWave/ with XX_XX_XX.npy
        self.raw_root = os.environ.get('XRF55_RAW_ROOT', ROOT + 'dataset/Raw_dataset/')

    def __repr__(self):
        return (f'Settings(raw_root={self.raw_root!r}, '
                f'split_root={self.split_root!r}, result_root={self.result_root!r})')


def get_settings() -> Settings:
    return Settings()
