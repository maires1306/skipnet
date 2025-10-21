#!/usr/bin/env python3
import os
import csv
import time
import argparse
import json
from typing import List, Optional

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import cifar10_resnet_38_masked
from data import prepare_train_data, prepare_test_data  # se preferir usar seus loaders

# ======================
# Utils
# ======================
def remove_module_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def save_checkpoint(state, is_best, ckpt_dir, filename="checkpoint.pth.tar"):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(ckpt_dir, "model_best.pth.tar")
        torch.save(state, best_path)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)   # (k, B)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def parse_masks_file(path: str, expected_len: int) -> List[List[int]]:
    masks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # aceita "1,0,1,..." ou "1 0 1 ..." ou com colchetes
            line = line.replace("[", "").replace("]", "")
            sep = "," if "," in line else " "
            parts = [p for p in line.split(sep) if p != ""]
            bits = [int(x) for x in parts]
            if len(bits) != expected_len:
                raise ValueError(f"Linha com {len(bits)} bits; esperado {expected_len}: {line}")
            if any(b not in (0,1) for b in bits):
                raise ValueError(f"Máscara inválida (apenas 0/1): {line}")
            masks.append(bits)
    return masks

def mask_exec_ratio(mask: List[int]) -> float:
    return sum(mask) / float(len(mask)) if len(mask) else 0.0

# ======================
# Treino / Validação
# ======================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            top1, = accuracy(logits, targets, topk=(1,))
        bs = targets.size(0)
        total += bs
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs

    return running_loss / total, running_top1 / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        top1, = accuracy(logits, targets, topk=(1,))
        bs = targets.size(0)
        total += bs
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs
    return running_loss / total, running_top1 / total

# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser("Fine-tune ResNet-38 masked (sweep)")
    parser.add_argument("--save-folder", default="save_mask_sweep", type=str)
    parser.add_argument("--baseline-ckpt", default="save_checkpoints/cifar10_resnet_38/model_best.pth.tar", type=str)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"], type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--masks-file", default="", type=str,
                        help="arquivo com máscaras (cada linha = 18 bits 0/1). Se vazio, usa exemplos embutidos.")
    parser.add_argument("--resume", action="store_true",
                        help="tenta retomar caso exista checkpoint por máscara.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = True

    # --------- Data (você pode trocar para prepare_train_data/prepare_test_data se preferir) ---------
    # (mantendo transforms padrão CIFAR-10)
    if args.dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size,
                                                   shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        raise NotImplementedError

    # --------- Modelo base + carregar baseline ---------
    model = cifar10_resnet_38_masked().to(device)
    ckpt = torch.load(args.baseline_ckpt, map_location=device)
    sd = remove_module_prefix(ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=True)

    total_blocks = 18  # ResNet-38 CIFAR: 6+6+6

    # --------- Máscaras ---------
    if args.masks_file:
        masks_list = parse_masks_file(args.masks_file, expected_len=total_blocks)
    else:
        # exemplos embutidos (edite à vontade)
        masks_list = [
            [1]*total_blocks,  # baseline (executa tudo) - sempre começo por aqui
            [0]*total_blocks,  # tudo pulado (vai performar mal, só para referência)
            # executa 3 primeiras e pula 3 últimas em cada grupo
            [1,1,1,0,0,0,  1,1,1,0,0,0,  1,1,1,0,0,0],
            # alternando 1/0
            [1,0,1,0,1,0,  1,0,1,0,1,0,  1,0,1,0,1,0],
            # metade inicial on, metade final off
            [1]*9 + [0]*9,
            # metade inicial off, metade final on
            [0]*9 + [1]*9,
        ]

    # --------- Loop por máscara ---------
    criterion = nn.CrossEntropyLoss().to(device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for mi, mask in enumerate(masks_list):
        exec_ratio = mask_exec_ratio(mask)
        mask_name = f"mask_{mi:02d}_exec{int(exec_ratio*100):02d}p"
        out_dir = os.path.join(args.save_folder, f"{timestamp}_{mask_name}")
        os.makedirs(out_dir, exist_ok=True)

        # aplica máscara fixa no modelo
        model.set_mask(mask)

        # (opcional) reinit do otimizador a cada máscara (recomendado)
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )

        start_epoch = 0
        best_acc = -1.0

        # retomar?
        latest_ckpt = os.path.join(out_dir, "checkpoint_latest.pth.tar")
        if args.resume and os.path.isfile(latest_ckpt):
            ck = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ck["state_dict"], strict=True)
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = ck["epoch"] + 1
            best_acc = ck.get("best_acc", best_acc)
            print(f"[{mask_name}] Resumido de época {start_epoch} (best_acc={best_acc:.2f})")

        print(f"\n=== Treinando com {mask_name} ({sum(mask)}/{len(mask)} blocos ativos) ===")

        # salva a máscara usada
        with open(os.path.join(out_dir, "mask.json"), "w") as f:
            json.dump({"mask": mask, "exec_ratio": exec_ratio}, f)

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, test_loader, criterion, device)
            dt = time.time() - t0

            print(f"[{mask_name}] epoch {epoch+1}/{args.epochs} "
                  f"| train_loss {train_loss:.4f} acc {train_acc:.2f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.2f} "
                  f"| {dt:.1f}s")

            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc

            # salvar checkpoint da época e best
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "mask": mask,
                "exec_ratio": exec_ratio,
            }
            save_checkpoint(state, is_best, out_dir, filename=f"checkpoint_epoch_{epoch:03d}.pth.tar")
            # link/arquivo para o latest
            save_checkpoint(state, is_best, out_dir, filename="checkpoint_latest.pth.tar")

        print(f"[{mask_name}] Fim. Best Val Acc = {best_acc:.2f}% | dir={out_dir}")

    print("\nSweep concluído ✅")

if __name__ == "__main__":
    main()
