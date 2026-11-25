"""Fine-tune MobileNetV3-Large on the four crocodilian Groups.

Reads training/data/images/{train,val}/<group>/*.jpg (from fetch_inat.py) and
writes the best checkpoint by validation accuracy to checkpoints/best.pt.

Usage: python train.py [--epochs 10]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from preprocess import MEAN, STD, SIZE, load_eval

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "images"
CHECKPOINTS = ROOT / "checkpoints"
GROUPS = ["alligator", "caiman", "crocodile", "gharial"]


class EvalFolder(Dataset):
    """Image folder using the exact deployed preprocessing."""

    def __init__(self, split: str):
        self.items = [
            (p, GROUPS.index(g))
            for g in GROUPS
            for p in sorted((IMAGES / split / g).glob("*.jpg"))
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return load_eval(path), label


ARCH = "vit_small_patch14_dinov2.lvd142m"
TRAINABLE_BLOCKS = 4  # of 12; earlier blocks stay frozen (CPU budget, small dataset)


def build_model(pretrained: bool = True) -> nn.Module:
    """DINOv2 ViT-S/14 at 224x224 with a 4-way head.

    MobileNetV3-Large topped out at ~64% val accuracy; frozen DINOv2 ViT-S
    features alone reached ~80% (see docs/adr/0001).
    """
    import timm

    return timm.create_model(ARCH, pretrained=pretrained, num_classes=len(GROUPS), img_size=SIZE)


def freeze_early_layers(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks[-TRAINABLE_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in list(model.norm.parameters()) + list(model.head.parameters()):
        p.requires_grad = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(1234)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(IMAGES / "train", transform=train_tf)
    assert train_ds.classes == GROUPS, train_ds.classes
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=3)
    val_dl = DataLoader(EvalFolder("val"), batch_size=64, num_workers=3)

    model = build_model()
    freeze_early_layers(model)
    head = list(model.head.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": 2e-5},
        {"params": head, "lr": 1e-3},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    CHECKPOINTS.mkdir(exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for x, y in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
        scheduler.step()
        val_acc = evaluate(model, val_dl)
        print(f"epoch {epoch}: loss {running / len(train_ds):.3f}  val_acc {val_acc:.3f}  "
              f"({time.time() - start:.0f}s)", flush=True)
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), CHECKPOINTS / "best.pt")
    print(f"best val_acc {best:.3f}")


if __name__ == "__main__":
    main()
