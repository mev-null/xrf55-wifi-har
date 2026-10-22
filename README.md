# XRF55 Wi-Fi HAR

Human activity recognition from Wi-Fi CSI on the [XRF55](https://aiotgroup.github.io/XRF55/) dataset:
5 seconds of CSI amplitude (270 channels x 1000 time steps at 200 Hz) are classified into 55 indoor actions.
Two models are implemented:

- **ResNet18**, the 1D ResNet18 of the XRF55 authors' code, trained alone with cross-entropy
  (5.31M parameters, of which 4.29M are used in the forward pass).
- **THAT** ([Li et al., AAAI 2021](https://doi.org/10.1609/aaai.v35i1.16103)), a two-stream Transformer with the
  multi-scale CNN (MCAT) in both the time and the channel stream (7.16M parameters).

## Getting started

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync   # creates .venv from uv.lock
```

**Data.** Download the Wi-Fi data of Scene 1 (Part 1 of the dataset; the download links are on the
[XRF55 project page](https://aiotgroup.github.io/XRF55/) and in the [XRF55 repository](https://github.com/airslab2020/XRF55-repo))
and put the files `<subject>_<action>_<repetition>.npy` in `dataset/Raw_dataset/WiFi/`, or point `XRF55_RAW_ROOT`
to the folder that contains `WiFi/`.

**Split.** Repetitions 1-14 of every action go to `train_data/`, 15-20 to `test_data/`, and the file lists are
written to `dataset/XRF_dataset/`. With subjects 1-11 this gives 8,470 training and 3,630 test samples.

```sh
uv run python split_train_test.py
uv run python generate_txt.py
```

**Train.**

```sh
# ResNet18 (settings of the XRF55 paper: Adam 1e-3, x0.5 every 40 epochs, 200 epochs, batch 64)
uv run python wifi_train.py --group resnet18 --seed 0 --subjects 1-11

# THAT (Adam 1e-4, 5-epoch warmup and cosine decay, 50 epochs, batch 16, gradient clipping 1.0)
uv run python wifi_train.py --group that --seed 0 --subjects 1-11 --model that \
    --batch_size 16 --epochs 50 --lr 1e-4 --lr_schedule cosine --warmup_epochs 5 --grad_clip 1.0
```

Each run writes `config.json`, `metrics.json` (accuracy per subject and class, confusion matrix),
`history.json` (learning curves), `predictions.json` and `model_last.pth` to
`result/wifi/<group>/seed<seed>/`. The test set is never used to pick a model.

## Repository layout

| Path | Contents |
|---|---|
| `split_train_test.py` | train/test split of the raw data |
| `generate_txt.py` | file lists of the split (`dataset/XRF_dataset/dml_train.txt`, `dml_val.txt`) |
| `XRFDataset.py` | `XRFWiFiDataset`: Wi-Fi samples of the lists, optionally for some subjects only |
| `wifi_train.py` | training and evaluation of one model |
| `model/resnet1d.py` | ResNet18 (from the XRF55 repository) |
| `model/that.py` | THAT with MCAT in both streams |
| `settings.py` | data and output locations |
| `dataset/Raw_dataset/WiFi/` | downloaded Wi-Fi data (not included) |
| `dataset/XRF_dataset/{train,test}_data/WiFi/` | split data, written by `split_train_test.py` (not included) |
| `result/wifi/<group>/seed<seed>/` | outputs of `wifi_train.py` |
| `pyproject.toml`, `uv.lock` | environment (`uv sync`) |

## License

The code is released under the [MIT License](LICENSE). It is derived from
[airslab2020/XRF55-repo](https://github.com/airslab2020/XRF55-repo) (MIT, © 2024 aiotgroup).

`model/that.py` is an implementation of THAT based on the paper ([Li et al., AAAI 2021](https://doi.org/10.1609/aaai.v35i1.16103)).

The XRF55 data are not included; follow the terms of the dataset.

## Citation

```bibtex
@article{wang2024xrf55,
  title   = {XRF55: A Radio Frequency Dataset for Human Indoor Action Analysis},
  author  = {Wang, Fei and Lv, Yizhe and Zhu, Mengdie and Ding, Han and Han, Jinsong},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume  = {8},
  number  = {1},
  year    = {2024}
}

@inproceedings{li2021that,
  title     = {Two-Stream Convolution Augmented Transformer for Human Activity Recognition},
  author    = {Li, Bing and Cui, Wei and Wang, Wei and Zhang, Le and Chen, Zhenghua and Wu, Min},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2021}
}
```
