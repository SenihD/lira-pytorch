# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, ImageFolder
from tqdm import tqdm

from dataset_utils import resolve_dataset_root
from wide_resnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--n_shadows", default=16, type=int)
parser.add_argument("--shadow_id", default=1, type=int)
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'malimg', 'malnet'])
parser.add_argument("--dataset_path", default=None, type=str)
parser.add_argument("--savedir", default=None, type=str)
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()
if args.savedir is None:
    args.savedir = f"exp/{args.dataset}"


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def run():
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    pl.seed_everything(seed)

    args.debug = True
    wandb.init(project="lira", mode="disabled" if args.debug else "online")
    wandb.config.update(args)

    # Dataset
    dataset_root = resolve_dataset_root(args.dataset, args.dataset_path)
    dataset_name = args.dataset.lower()

    if dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])
        train_ds = CIFAR10(root=dataset_root, train=True, download=True, transform=train_transform)
        test_ds = CIFAR10(root=dataset_root, train=False, download=True, transform=test_transform)

    elif dataset_name == 'malimg':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        traindir = dataset_root / "train"
        testdir = dataset_root / "test"

        train_ds = ImageFolder(root=traindir, transform=train_transform)
        test_ds = ImageFolder(root=testdir, transform=test_transform)

    elif dataset_name == 'malnet':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dir = dataset_root / "train"
        test_dir = dataset_root / "test"
        if train_dir.is_dir() and test_dir.is_dir():
            train_ds = ImageFolder(root=train_dir, transform=train_transform)
            test_ds = ImageFolder(root=test_dir, transform=test_transform)
        else:
            train_ds = ImageFolder(root=dataset_root, transform=train_transform)
            test_ds = ImageFolder(root=dataset_root, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    num_classes = len(train_ds.classes)

    # In - Out grouping
    size = len(train_ds)
    np.random.seed(seed)
    if args.n_shadows is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(args.n_shadows, size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.n_shadows)
        keep = np.array(keep[args.shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
    else:
        keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
        keep.sort()
    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(train_ds, keep)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # Model
    if args.model == "wresnet28-2":
        m = WideResNet(28, 2, 0.0, num_classes)
    elif args.model == "wresnet28-10":
        m = WideResNet(28, 10, 0.3, num_classes)
    elif args.model == "resnet18":
        m = models.resnet18(weights=None, num_classes=num_classes)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
    else:
        raise NotImplementedError
    m = m.to(DEVICE)

    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Train
    for i in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(m(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        wandb.log({"loss": loss_total / len(train_dl)})

    m.eval()
    print(f"[test] acc_test: {get_acc(m, test_dl):.4f}")
    wandb.log({"acc_test": get_acc(m, test_dl)})

    savedir = os.path.join(args.savedir, str(args.shadow_id))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")


@torch.no_grad()
def get_acc(model, dl):
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()


if __name__ == "__main__":
    run()
