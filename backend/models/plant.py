"""PlantSeg metadata, datasets, classifiers, and model-routing utilities.

PlantSeg provides image-level ``Plant`` and ``Disease`` labels in Metadata.csv
and lesion polygons in COCO JSON files.  This module keeps plant recognition
separate from disease recognition so an inference request can be routed to a
plant-specific disease model.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANTSEG_ROOT = Path.home() / "Downloads" / "plantseg" / "plantseg"


def load_plantseg_metadata(root: Path = DEFAULT_PLANTSEG_ROOT) -> List[dict]:
    """Read PlantSeg Metadata.csv and attach each image's absolute path."""
    root = Path(root)
    metadata_path = root / "Metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing PlantSeg metadata: {metadata_path}")
    rows = []
    # The distributed CSV includes a UTF-8 BOM before the first ``Name`` header.
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            split = row["Split"].strip().lower()
            split_name = {"training": "train", "validation": "val", "test": "test"}.get(split, split)
            row["split"] = split_name
            row["image_path"] = str(root / "images" / split_name / row["Name"])
            row["plant_key"] = row["Plant"].strip().lower().replace(" ", "_")
            row["disease_key"] = row["Disease"].strip().lower().replace(" ", "_")
            rows.append(row)
    return rows


def build_label_maps(rows: List[dict]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    plants = sorted({row["plant_key"] for row in rows})
    diseases_by_plant = defaultdict(set)
    for row in rows:
        diseases_by_plant[row["plant_key"]].add(row["disease_key"])
    return (
        {name: index for index, name in enumerate(plants)},
        {plant: sorted(names) for plant, names in diseases_by_plant.items()},
    )


class PlantSegClassificationDataset(Dataset):
    """Image-level dataset for plant or plant-specific disease training."""

    def __init__(self, rows: List[dict], transform=None,
                 plant_to_index: Optional[Dict[str, int]] = None,
                 disease_to_index: Optional[Dict[str, int]] = None) -> None:
        self.rows = rows
        self.transform = transform
        self.plant_to_index = plant_to_index or {}
        self.disease_to_index = disease_to_index or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "plant": self.plant_to_index[row["plant_key"]]
            if self.plant_to_index else -1,
            "disease": self.disease_to_index[row["disease_key"]]
            if self.disease_to_index else -1,
            "plant_name": row["plant_key"],
            "disease_name": row["disease_key"],
        }


def build_mobilenet_classifier(num_classes: int, pretrained: bool = True):
    """Create a lightweight MobileNetV3-Small image classifier."""
    try:
        # Some Windows CPU torchvision wheels omit the compiled NMS operator
        # while torchvision registers its fake implementation at import time.
        # The classifier does not use NMS, so declare the operator schema.
        try:
            torch.library.Library("torchvision", "DEF").define(
                "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
            )
        except RuntimeError:
            pass
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        raise ImportError("torchvision is required for plant classification") from exc
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def model_registry(root: Path = DEFAULT_PLANTSEG_ROOT) -> dict:
    """Return the plant and disease labels used by the routing layer."""
    rows = load_plantseg_metadata(root)
    plant_to_index, diseases_by_plant = build_label_maps(rows)
    return {
        "plant_to_index": plant_to_index,
        "index_to_plant": {v: k for k, v in plant_to_index.items()},
        "diseases_by_plant": diseases_by_plant,
    }
