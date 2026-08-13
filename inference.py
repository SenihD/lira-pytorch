# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/inference.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, ImageFolder
from tqdm import tqdm

from dataset_utils import resolve_dataset_root
from wide_resnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'malimg', 'malnet'])
parser.add_argument("--dataset_path", default=None, type=str)
parser.add_argument("--savedir", default=None, type=str)

args = parser.parse_args()
if args.savedir is None:
    args.savedir = f"exp/{args.dataset}"


@torch.no_grad()
def run():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    # Dataset
    dataset_root = resolve_dataset_root(args.dataset, args.dataset_path)
    dataset_name = args.dataset.lower()

    if dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            ]
        )
        train_ds = CIFAR10(root=dataset_root, train=True, download=True, transform=transform)

    elif dataset_name == "malimg":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        traindir = dataset_root / "train"
        train_ds = ImageFolder(root=traindir, transform=train_transform)

    elif dataset_name == "malnet":
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dir = dataset_root / "train"
        if train_dir.is_dir():
            train_ds = ImageFolder(root=train_dir, transform=train_transform)
        else:
            train_ds = ImageFolder(root=dataset_root, transform=train_transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    num_classes = len(train_ds.classes)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)

    # Infer the logits with multiple queries
    for path in os.listdir(args.savedir):
        # For macOS file system
        if path == ".DS_Store":
            continue
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
        m.load_state_dict(torch.load(os.path.join(args.savedir, path, "model.pt")))
        m.to(DEVICE)
        m.eval()

        logits_n = []
        for i in range(args.n_queries):
            logits = []
            for x, _ in tqdm(train_dl):
                x = x.to(DEVICE)
                outputs = m(x)
                logits.append(outputs.cpu().numpy())
            logits_n.append(np.concatenate(logits))
        logits_n = np.stack(logits_n, axis=1)
        print(logits_n.shape)

        np.save(os.path.join(args.savedir, path, "logits.npy"), logits_n)


if __name__ == "__main__":
    run()
