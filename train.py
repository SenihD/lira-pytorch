# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
import time
from pathlib import Path
from collections import OrderedDict

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

# Federated learning options (only used with --federated)
parser.add_argument("--federated", action="store_true",
                    help="train the shadow model with federated learning (server + clients, FedAvg)")
parser.add_argument("--global_epochs", default=10, type=int,
                    help="number of global communication rounds between server and clients")
parser.add_argument("--num_clients", default=10, type=int,
                    help="total number of clients that hold private data shards")
parser.add_argument("--client_fraction", default=1.0, type=float,
                    help="fraction of clients selected for training in each global round")
parser.add_argument("--local_epochs", default=1, type=int,
                    help="number of local epochs each selected client trains per global round")
parser.add_argument("--partition", default="iid", type=str, choices=["iid", "dirichlet"],
                    help="how the shadow training set is split among clients: iid or non-iid dirichlet")
parser.add_argument("--alpha", default=0.5, type=float,
                    help="Dirichlet concentration parameter for the non-iid partition (lower = more non-iid)")

args = parser.parse_args()
if args.savedir is None:
    args.savedir = f"exp/{args.dataset}"


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class Server:
    """Holds the global model and aggregates client updates with FedAvg."""

    def __init__(self, model):
        self.model = model

    def broadcast_weights(self):
        """Send the current global weights to clients (copied to CPU for transport)."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def aggregate(self, client_updates, client_sizes):
        """FedAvg: weighted average of client weights by number of local samples."""
        total = sum(client_sizes)
        if total <= 0:
            raise ValueError("FedAvg requires at least one client with data")

        avg_state = OrderedDict()
        keys = client_updates[0].keys()
        for key in keys:
            weighted = torch.zeros_like(client_updates[0][key])
            for weights, size in zip(client_updates, client_sizes):
                weighted += weights[key].float() * (size / total)
            avg_state[key] = weighted.to(client_updates[0][key].dtype)

        self.model.load_state_dict(avg_state)
        return avg_state


class Client:
    """Trains the global model locally on its private data shard and returns the update."""

    def __init__(self, client_id, dataset, num_classes, batch_size, lr, local_epochs, device,
                 num_workers=4):
        self.client_id = client_id
        self.dataset = dataset
        self.num_classes = num_classes
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers)
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device
        self.num_samples = len(dataset)

    def train(self, global_weights):
        """Load global weights, train locally for `local_epochs`, return updated weights."""
        model = build_model(self.num_classes)
        model.load_state_dict(global_weights)
        model.to(self.device)

        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        model.train()
        loss_total = 0.0
        n_batches = 0
        for _ in range(self.local_epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)

                loss = F.cross_entropy(model(x), y)
                loss_total += loss.item()
                n_batches += 1

                optim.zero_grad()
                loss.backward()
                optim.step()

        avg_loss = loss_total / max(n_batches, 1)
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}, avg_loss


def build_model(num_classes=None):
    """Create the model architecture specified by args (uses num_classes if provided)."""
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
    return m


def partition_shadow_set(indices, num_clients, dataset, partition="iid", alpha=0.5, seed=0):
    """Split the shadow training indices among clients (iid random or Dirichlet non-iid)."""
    indices = np.asarray(indices)
    np.random.seed(seed)

    if partition == "dirichlet" and alpha > 0:
        labels = np.asarray(dataset.targets)[indices]
        num_classes = len(dataset.classes)
        client_indices = [[] for _ in range(num_clients)]

        for cls in range(num_classes):
            cls_idxs = indices[labels == cls]
            if len(cls_idxs) == 0:
                continue
            proportions = np.random.dirichlet([alpha] * num_clients)
            splits = np.maximum((proportions * len(cls_idxs)).astype(int), 0)
            splits[-1] = max(len(cls_idxs) - splits[:-1].sum(), 0)

            start = 0
            for c, count in enumerate(splits):
                client_indices[c].extend(cls_idxs[start:start + count])
                start += count

        return [np.asarray(c_idxs, dtype=int) for c_idxs in client_indices]

    # iid: random split
    perm = np.random.permutation(indices)
    return np.array_split(perm, num_clients)


def fedavg_run(server, clients, test_dl):
    """Iterative federated training: broadcast -> local train -> FedAvg aggregate."""
    global_weights = server.broadcast_weights()
    num_selected = max(1, int(round(args.client_fraction * len(clients))))

    for global_round in range(args.global_epochs):
        selected = np.random.choice(len(clients), size=num_selected, replace=False)
        updates, sizes, losses = [], [], []

        for cid in selected:
            client = clients[int(cid)]
            weights, avg_loss = client.train(global_weights)
            updates.append(weights)
            sizes.append(client.num_samples)
            losses.append(avg_loss)
            print(f"  round {global_round + 1}/{args.global_epochs} "
                  f"client {client.client_id}: loss {avg_loss:.4f}, samples {client.num_samples}")

        server.aggregate(updates, sizes)
        global_weights = server.broadcast_weights()

        avg_loss = float(np.mean(losses))
        test_acc = get_acc(server.model, test_dl)
        wandb.log({"round": global_round, "loss": avg_loss, "acc_test": test_acc})
        print(f"round {global_round + 1}/{args.global_epochs} "
              f"[{num_selected} clients] avg client loss: {avg_loss:.4f}, "
              f"test acc: {test_acc:.4f}")

    return global_weights


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

    full_train_ds = train_ds
    train_ds = torch.utils.data.Subset(train_ds, keep)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    if args.federated:
        # ---- Federated learning: server + clients ----
        print(f"[fed] {args.num_clients} clients, {args.global_epochs} global rounds, "
              f"{args.local_epochs} local epochs/round, partition={args.partition}")

        server = Server(build_model(num_classes))
        server.model.to(DEVICE)

        # Partition the shadow training subset among the clients.
        # `keep` holds indices into the original dataset, so build each client
        # shard directly from `full_train_ds` to keep the index space consistent.
        client_keep = partition_shadow_set(keep, args.num_clients, full_train_ds,
                                           partition=args.partition, alpha=args.alpha, seed=seed)
        clients = [
            Client(cid, torch.utils.data.Subset(full_train_ds, c_idx.tolist()), num_classes,
                   batch_size=128, lr=args.lr, local_epochs=args.local_epochs, device=DEVICE)
            for cid, c_idx in enumerate(client_keep)
        ]

        fedavg_run(server, clients, test_dl)
        m = server.model
    else:
        # ---- Centralized training (original pipeline) ----
        m = build_model(num_classes)
        m = m.to(DEVICE)

        optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

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
