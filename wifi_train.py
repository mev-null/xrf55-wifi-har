"""Wi-Fi only training on XRF55: ResNet18 baseline (paper Table 6: 87.26%, all subjects) and THAT.

Derived from dml_train.py in airslab2020/XRF55-repo: one Wi-Fi model trained with cross-entropy only
(no mutual-learning or BERT terms); the defaults are the optimizer and schedule of dml_train.py.
The reported accuracy is the last epoch's; the test set is never used to pick a model.

    python split_train_test.py && python generate_txt.py
    python wifi_train.py --group resnet18_baseline --seed 0
    python wifi_train.py --group that --seed 0 --subjects 1-11 --model that \
        --batch_size 16 --epochs 50 --lr 1e-4 --lr_schedule cosine --warmup_epochs 5 --grad_clip 1.0

Writes config, metrics, history, predictions and last-epoch weights to <result_root>/<group>/seed<seed>/.
"""

import argparse
import json
import math
import os
import platform
import re

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

import XRFDataset
from model import resnet1d, that
from settings import get_settings

NUM_CLASSES = 55


def parse_subjects(text):
    """'1-11' or '1,3,5' -> [1, ..., 11] / [1, 3, 5]."""
    ids = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        ids += list(range(int(lo), int(hi or lo) + 1))
    return ids


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Wi-Fi only model on XRF55')
    parser.add_argument(
        '--group',
        required=True,
        help='experiment name, e.g. resnet18_baseline; output: <result_root>/<group>/seed<seed>/',
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--model', default='resnet18', help='function in model/resnet1d.py or model/that.py')
    parser.add_argument(
        '--subjects', type=parse_subjects, default=None, help='subject IDs, e.g. "1-11" or "1,3,5" (default: all)'
    )
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001, help='Adam learning rate')
    parser.add_argument(
        '--lr_schedule',
        choices=['step', 'cosine'],
        default='step',
        help='step: x lr_gamma at each milestone; cosine: half cosine to 0, updated every batch',
    )
    parser.add_argument(
        '--lr_milestones', type=int, nargs='+', default=[40, 80, 120, 160], help='step: epochs to decay lr'
    )
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='step: lr decay factor at each milestone')
    parser.add_argument(
        '--warmup_epochs', type=int, default=0, help='cosine: raise lr linearly over the first N epochs first (0: off)'
    )
    parser.add_argument('--grad_clip', type=float, default=0.0, help='clip the gradient norm to this value (0: off)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    return parser.parse_args()


def dump_json(obj, path):
    """indent=2, but lists of numbers/strings stay on one line (e.g. "subjects": [1, 2, ..., 11])."""
    text = json.dumps(obj, indent=2)
    text = re.sub(r'\[\s+([^\[\]{}]*?)\s+\]', lambda m: '[' + re.sub(r',\s+', ', ', m.group(1)) + ']', text)
    with open(path, 'w') as f:
        f.write(text + '\n')


def environment():
    """Software versions, so that a result can be traced back to what produced it."""
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'os': platform.platform(),
    }


def build_model(name):
    for module in (resnet1d, that):
        if hasattr(module, name):
            return getattr(module, name)()
    raise ValueError(f'unknown --model {name!r}: not a function in model/resnet1d.py or model/that.py')


def cosine_schedule(warmup_steps, total_steps):
    """lr multiplier per batch: linear warmup to 1, then a half cosine down to 0."""

    def scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    return scale


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def logits_of(output):
    # resnet18() returns logits; resnet18_mutual() returns (logits, feature)
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    """Mean loss over the loader, and the predictions and labels."""
    model.eval()
    loss, preds, labels = 0.0, [], []
    for samples, label in loader:
        logits = logits_of(model(samples.to(device)))
        loss += loss_fn(logits, label.to(device)).item() * len(label)
        preds.append(logits.argmax(1).cpu())
        labels.append(label)
    return loss / len(loader.dataset), torch.cat(preds).numpy(), torch.cat(labels).numpy()


def accuracy_breakdown(preds, labels, subjects):
    """Accuracy per subject and per class (in %), and the confusion matrix (rows: true, cols: predicted)."""
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (labels, preds), 1)
    per_class = 100.0 * np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    per_subject = {
        str(s): round(100.0 * float((preds[subjects == s] == labels[subjects == s]).mean()), 2)
        for s in np.unique(subjects)
    }
    return per_subject, [round(float(a), 2) for a in per_class], confusion.tolist()


def repetitions(dataset):
    """Repetition indices (1-20) present in a dataset; file names are <subject>_<action>_<repetition>."""
    return sorted({int(name.split('_')[2]) for name in dataset.data['file_name']})


if __name__ == "__main__":
    '''========================= Setup =========================='''
    args = parse_args()
    settings = get_settings()
    torch.manual_seed(args.seed)
    device = get_device()
    out_dir = os.path.join(settings.result_root, args.group, 'seed%d' % args.seed)
    # never overwrite an existing run; checked now, created only once setup has succeeded
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} already exists")

    '''========================= Dataset =========================='''
    train_dataset = XRFDataset.XRFWiFiDataset(settings.split_root, is_train=True, subjects=args.subjects)
    test_dataset = XRFDataset.XRFWiFiDataset(settings.split_root, is_train=False, subjects=args.subjects)
    # persistent_workers: without it, macOS (spawn) spends ~20 s per epoch shutting workers down
    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == 'cuda',
    )
    train_data = Data.DataLoader(dataset=train_dataset, shuffle=True, drop_last=False, **loader_kw)
    test_data = Data.DataLoader(dataset=test_dataset, shuffle=False, **loader_kw)
    subjects = sorted(set(train_dataset.data['subject']))
    print(f"{settings}\ndevice={device} subjects={subjects} train={len(train_dataset)} test={len(test_dataset)}")

    '''========================= Model =========================='''
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_schedule == 'cosine':  # stepped after every batch
        steps = len(train_data)
        scale = cosine_schedule(args.warmup_epochs * steps, args.epochs * steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    else:  # stepped after every epoch
        if args.warmup_epochs:
            raise ValueError('--warmup_epochs needs --lr_schedule cosine')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    loss_ce = nn.CrossEntropyLoss()

    '''========================= Config =========================='''
    # Saved before training so that an unfinished run is still identifiable
    config = {
        **vars(args),
        'subjects': subjects,
        'n_train': len(train_dataset),
        'n_test': len(test_dataset),
        'train_repetitions': repetitions(train_dataset),
        'test_repetitions': repetitions(test_dataset),
        'optimizer': 'Adam',
        'loss': 'cross_entropy',
        'device': str(device),
        'env': environment(),
    }
    os.makedirs(out_dir)
    dump_json(config, os.path.join(out_dir, 'config.json'))

    '''========================= Train (CE only) =========================='''
    history = {'lr': [], 'train_loss': [], 'test_loss': []}
    for epoch in range(args.epochs):
        history['lr'].append(optimizer.param_groups[0]['lr'])
        model.train()
        losses = 0.0
        for samples, labels in train_data:
            samples, labels = samples.to(device), labels.to(device)
            outputs = logits_of(model(samples))
            loss = loss_ce(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if args.lr_schedule == 'cosine':
                scheduler.step()
            losses += loss.item() * len(labels)
        if args.lr_schedule == 'step':
            scheduler.step()
        train_loss = losses / len(train_dataset)  # running mean over the epoch, train mode
        if not np.isfinite(train_loss):
            raise RuntimeError(f"train loss became {train_loss} at epoch {epoch}")

        test_loss, preds, test_labels = evaluate(model, test_data, device, loss_ce)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        print(f"Epoch: {epoch:3d} train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}")
        dump_json(history, os.path.join(out_dir, 'history.json'))  # rewritten every epoch to survive a crash

    '''========================= Save =========================='''
    torch.save(model.state_dict(), os.path.join(out_dir, 'model_last.pth'))
    test_subjects = np.array(test_dataset.data['subject'])
    final_acc = 100.0 * float((preds == test_labels).mean())
    per_subject, per_class, confusion = accuracy_breakdown(preds, test_labels, test_subjects)
    print(f"final test accuracy: {final_acc:.2f}")
    metrics = {
        'final_acc': final_acc,
        'per_subject_acc': per_subject,
        'per_class_acc': per_class,
        'confusion': confusion,
    }
    predictions = {
        'file_name': test_dataset.data['file_name'],
        'subject': test_dataset.data['subject'],
        'label': test_labels.tolist(),
        'pred': preds.tolist(),
    }
    dump_json(metrics, os.path.join(out_dir, 'metrics.json'))  # final results
    dump_json(predictions, os.path.join(out_dir, 'predictions.json'))  # per-sample test predictions
    print("saved to", out_dir)
