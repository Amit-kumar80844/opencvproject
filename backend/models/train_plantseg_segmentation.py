"""Train and evaluate dual-head lesion segmentation and severity on PlantSeg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from .dataset import generate_grabcut_mask, IMAGENET_MEAN, IMAGENET_STD


class PlantSegMaskDataset(Dataset):
    """PlantSeg images plus two targets: GrabCut leaf and annotated lesion."""

    def __init__(self, root: Path, split: str, size: int = 256, train: bool = False):
        self.root, self.split, self.size = Path(root), split, size
        image_dir = self.root / "images" / split
        mask_dir = self.root / "annotations" / split
        self.samples = []
        for image_path in sorted(image_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                self.samples.append((image_path, mask_path))
        if not self.samples:
            raise FileNotFoundError(f"No paired samples under {image_dir} and {mask_dir}")
        spatial = [A.Resize(size, size)]
        if train:
            spatial += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2), A.RandomRotate90(p=0.5)]
        self.transform = A.Compose(spatial + [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        disease = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or disease is None:
            raise IOError(f"Could not read {image_path} or {mask_path}")
        bgr = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        disease = cv2.resize(disease, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        leaf = generate_grabcut_mask(bgr)
        augmented = self.transform(
            image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            masks=[leaf, (disease > 0).astype(np.uint8)],
        )
        return {
            "image": augmented["image"].float(),
            "mask": torch.stack([torch.as_tensor(m).float() for m in augmented["masks"]]),
        }


def severity_percent(leaf: torch.Tensor, disease: torch.Tensor) -> torch.Tensor:
    leaf = leaf.bool()
    disease = disease.bool() & leaf
    leaf_area = leaf.flatten(1).sum(1).float().clamp_min(1.0)
    return disease.flatten(1).sum(1).float() / leaf_area * 100.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    disease_dice, severity_errors, count = 0.0, 0.0, 0
    for batch in loader:
        images, target = batch["image"].to(device), batch["mask"].to(device)
        probabilities = torch.sigmoid(model(images))
        prediction = probabilities > 0.5
        truth = target > 0.5
        intersection = (prediction[:, 1] & truth[:, 1]).flatten(1).sum(1).float()
        denom = prediction[:, 1].flatten(1).sum(1) + truth[:, 1].flatten(1).sum(1)
        disease_dice += ((2 * intersection + 1) / (denom + 1)).sum().item()
        severity_errors += torch.abs(
            severity_percent(prediction[:, 0], prediction[:, 1])
            - severity_percent(truth[:, 0], truth[:, 1])
        ).sum().item()
        count += images.shape[0]
    return {"disease_dice": disease_dice / max(count, 1),
            "severity_mae_percent": severity_errors / max(count, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = PlantSegMaskDataset(args.root, "train", train=True)
    val_ds = PlantSegMaskDataset(args.root, "val")
    test_ds = PlantSegMaskDataset(args.root, "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = smp.Unet(
        encoder_name="mobilenet_v3_small", encoder_weights="imagenet",
        in_channels=3, classes=2, decoder_attention_type="scse",
    ).to(device)
    criterion = smp.losses.TverskyLoss(mode="multilabel", alpha=0.3, beta=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            images, target = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), target)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={loss.item():.4f} "
              f"val_dice={metrics['disease_dice']:.4f} "
              f"val_severity_mae={metrics['severity_mae_percent']:.2f}%")
        if metrics["severity_mae_percent"] < best:
            best = metrics["severity_mae_percent"]
            torch.save({"model_state_dict": model.state_dict(),
                        "img_size": 256, "metrics": metrics}, args.output)
    model.load_state_dict(torch.load(args.output, map_location=device, weights_only=False)["model_state_dict"])
    results = {"device": str(device), "train_images": len(train_ds),
               "val_images": len(val_ds), "test_images": len(test_ds),
               "test": evaluate(model, test_loader, device)}
    results_path = args.output.with_suffix(".json")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
