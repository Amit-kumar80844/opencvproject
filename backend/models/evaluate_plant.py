"""Evaluate the trained PlantSeg plant gate and routed disease models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader
# Compatibility with CPU torchvision wheels that omit the unused NMS operator.
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


def _load_model(path: Path, num_classes: int):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_mobilenet_classifier(num_classes, pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    return model.eval(), state


def _predict(model, loader, key: str, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(batch[key].tolist())
    return y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = load_plantseg_metadata(args.root)
    test_rows = [row for row in rows if row["split"] == "test"]
    plant_to_index, diseases_by_plant = build_label_maps(rows)
    index_to_plant = {v: k for k, v in plant_to_index.items()}
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Plant-gate evaluation.
    plant_ds = PlantSegClassificationDataset(test_rows, tf, plant_to_index=plant_to_index)
    plant_loader = DataLoader(plant_ds, batch_size=args.batch_size, shuffle=False)
    plant_model, _ = _load_model(args.checkpoints / "plant_mobilenet_v3_small.pth",
                                 len(plant_to_index))
    plant_model.to(device)
    plant_true, plant_pred = _predict(plant_model, plant_loader, "plant", device)
    plant_names = [index_to_plant[i] for i in range(len(index_to_plant))]
    plant_report = classification_report(
        plant_true, plant_pred, labels=list(range(len(plant_names))),
        target_names=plant_names, output_dict=True, zero_division=0
    )
    with (args.output / "plant_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(plant_report, f, indent=2)
    np.savetxt(args.output / "plant_confusion_matrix.csv",
               confusion_matrix(plant_true, plant_pred,
                                labels=list(range(len(plant_names)))),
               delimiter=",", fmt="%d")
    fig, ax = plt.subplots(figsize=(18, 16))
    ConfusionMatrixDisplay.from_predictions(
        plant_true, plant_pred, labels=list(range(len(plant_names))),
        display_labels=plant_names, xticks_rotation="vertical", ax=ax, colorbar=False
    )
    fig.tight_layout()
    fig.savefig(args.output / "plant_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Routed disease evaluation for every available plant model.
    disease_results = {}
    registry_path = args.checkpoints / "plant_model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for plant, entry in registry["diseases"].items():
        plant_rows = [row for row in test_rows if row["plant_key"] == plant]
        if not plant_rows:
            continue
        labels = entry["labels"]
        disease_to_index = {name: int(index) for name, index in labels.items()}
        disease_ds = PlantSegClassificationDataset(
            plant_rows, tf, disease_to_index=disease_to_index
        )
        disease_loader = DataLoader(disease_ds, batch_size=args.batch_size, shuffle=False)
        model, _ = _load_model(args.checkpoints / entry["checkpoint"], len(labels))
        model.to(device)
        true, pred = _predict(model, disease_loader, "disease", device)
        names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
        disease_results[plant] = classification_report(
            true, pred, labels=list(range(len(names))), target_names=names,
            output_dict=True, zero_division=0
        )
    (args.output / "disease_reports_by_plant.json").write_text(
        json.dumps(disease_results, indent=2), encoding="utf-8"
    )

    summary = {
        "device": str(device),
        "test_images": len(test_rows),
        "plant_accuracy": plant_report["accuracy"],
        "plant_macro_f1": plant_report["macro avg"]["f1-score"],
        "disease_models_evaluated": len(disease_results),
        "disease_accuracy_by_plant": {
            plant: report["accuracy"] for plant, report in disease_results.items()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
