# train_pspnet.py
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from model import PSPNet
from datasets import SimpleSegDataset
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import os

def poly_lr_lambda(curr_iter, max_iter, power=0.9, base_lr=0.01):
    return (1 - curr_iter / max_iter) ** power

def train(args):
    transforms = T.Compose([T.Resize((args.crop_h, args.crop_w)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    train_ds = SimpleSegDataset(args.train_images, args.train_masks, transforms=transforms)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSPNet(n_classes=args.num_classes, backbone=args.backbone, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda it: poly_lr_lambda(it, args.max_iter, power=args.power))
    os.makedirs(args.output_dir, exist_ok=True)

    iters = 0
    for epoch in range(0, args.epochs):
        model.train()
        pbar = tqdm(loader)
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            iters += 1
            pbar.set_description(f"Epoch[{epoch}] Loss:{loss.item():.4f} LR:{scheduler.get_last_lr()[0]:.6f}")
            if iters >= args.max_iter:
                break
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"psp_epoch_{epoch}.pth"))
        if iters >= args.max_iter:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", required=True)
    parser.add_argument("--train_masks", required=True)
    parser.add_argument("--output_dir", default="./psp_output")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=90000)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--crop_h", type=int, default=473)
    parser.add_argument("--crop_w", type=int, default=473)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--backbone", default="resnet50")
    args = parser.parse_args()
    train(args)
