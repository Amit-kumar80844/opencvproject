"""Train the plant gate and one disease classifier per plant.

Example:
  python -m backend.models.train_plant --root C:\\Users\\amits\\Downloads\\plantseg\\plantseg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Work around the CPU torchvision wheel's missing unused NMS operator.
try:
    torch.library.Library("torchvision", "DEF").define(
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    )
except RuntimeError:
    pass
from torchvision import transforms

from .plant import (
    PlantSegClassificationDataset,
    build_label_maps,
    build_mobilenet_classifier,
    load_plantseg_metadata,
)


def train_model(train_rows, val_rows, label_map, label_key, output_path,
                epochs: int, batch_size: int, device: torch.device) -> None:
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_ds = PlantSegClassificationDataset(train_rows, tf,
                                             plant_to_index=label_map if label_key == "plant" else {},
                                             disease_to_index=label_map if label_key == "disease" else {})
    val_ds = PlantSegClassificationDataset(val_rows, val_tf,
                                           plant_to_index=label_map if label_key == "plant" else {},
                                           disease_to_index=label_map if label_key == "disease" else {})
    model = build_mobilenet_classifier(len(label_map)).to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            target = batch[label_key].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch["image"].to(device)), target)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                target = batch[label_key].to(device)
                pred = model(batch["image"].to(device)).argmax(1)
                correct += int((pred == target).sum())
                total += target.numel()
        accuracy = correct / max(total, 1)
        print(f"{label_key} epoch {epoch + 1}/{epochs}: val_acc={accuracy:.4f}")
        if accuracy >= best:
            best = accuracy
            torch.save({"model_state_dict": model.state_dict(),
                        "labels": label_map, "accuracy": accuracy}, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("backend/checkpoints"))
    args = parser.parse_args()
    rows = load_plantseg_metadata(args.root)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    plant_to_index, diseases_by_plant = build_label_maps(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(train_rows, val_rows, plant_to_index, "plant",
                args.output / "plant_mobilenet_v3_small.pth",
                args.epochs, args.batch_size, device)
    disease_registry = {}
    for plant, diseases in diseases_by_plant.items():
        disease_to_index = {name: i for i, name in enumerate(diseases)}
        plant_train = [r for r in train_rows if r["plant_key"] == plant]
        plant_val = [r for r in val_rows if r["plant_key"] == plant]
        if len(disease_to_index) < 2 or not plant_val:
            continue
        path = args.output / f"disease_{plant}_mobilenet_v3_small.pth"
        train_model(plant_train, plant_val, disease_to_index, "disease",
                    path, args.epochs, args.batch_size, device)
        # Store portable filenames; the API resolves them inside its checkpoint
        # directory after the artifacts are copied from Colab.
        disease_registry[plant] = {"labels": disease_to_index, "checkpoint": path.name}
    (args.output / "plant_model_registry.json").write_text(
        json.dumps({"plants": plant_to_index, "diseases": disease_registry}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
