"""
test_classification.py — Test classifier with 5 images from each class
========================================================================
Tests the EfficientNet-B4 classifier on sample images from each disease class.
"""

import os
import sys
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path (go up 2 levels from notebooks/tea-leaf-disease-classifier)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from backend.models.unet import create_classifier, CLASSIFIER_CONFIG

# Constants
DATASET_ROOT = PROJECT_ROOT / "datasets" / "tea_sickness" / "tea sickness dataset"
CHECKPOINTS_DIR = PROJECT_ROOT / "backend" / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_CLASS = 4


def load_classifier():
    """Load the EfficientNet-B4 classifier."""
    checkpoint_path = CHECKPOINTS_DIR / CLASSIFIER_CONFIG["checkpoint"]
    
    if checkpoint_path.exists():
        classifier = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
        print(f"[TEST] Loaded classifier from: {checkpoint_path}")
    else:
        print(f"[TEST] WARNING: No checkpoint at {checkpoint_path}")
        classifier = create_classifier(num_classes=8)
    
    classifier.to(DEVICE).eval()
    return classifier


def get_transform():
    """Create preprocessing transform matching training."""
    return A.Compose([
        A.Resize(CLASSIFIER_CONFIG["input_size"], CLASSIFIER_CONFIG["input_size"]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_image(path: Path, transform):
    """Load and preprocess a single image."""
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    transformed = transform(image=img_np)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
    return tensor


def main():
    print("=" * 70)
    print("Classification Test — 5 random images per class")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    print()
    
    # Load classifier
    classifier = load_classifier()
    transform = get_transform()
    
    # Track results
    results = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": []})
    confusion = defaultdict(lambda: defaultdict(int))
    
    # Test each class
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = DATASET_ROOT / class_name
        
        if not class_dir.exists():
            print(f"[SKIP] Class directory not found: {class_dir}")
            continue
        
        # Get all images in class
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        
        if len(images) == 0:
            print(f"[SKIP] No images in: {class_dir}")
            continue
        
        # Sample random images
        sample_images = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
        
        print(f"\n{'-' * 70}")
        print(f"Testing: {class_name.upper()} ({len(images)} images total, testing {len(sample_images)})")
        print(f"{'-' * 70}")
        
        for img_path in sample_images:
            # Load and preprocess
            tensor = load_image(img_path, transform)
            
            # Run inference
            with torch.no_grad():
                logits = classifier(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(logits.argmax(dim=1).item())
                pred_name = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx].item())
            
            # Track results
            is_correct = (pred_idx == class_idx)
            results[class_name]["total"] += 1
            if is_correct:
                results[class_name]["correct"] += 1
            
            confusion[class_name][pred_name] += 1
            
            # Display
            status = "✓" if is_correct else "✗"
            print(f"  {status} {img_path.name:40} → {pred_name:15} ({confidence*100:.1f}%)")
            
            # Show top-3 if wrong
            if not is_correct:
                top3_probs, top3_idx = probs.topk(3)
                top3_names = [CLASS_NAMES[i] for i in top3_idx.tolist()]
                top3_conf = [f"{p*100:.1f}%" for p in top3_probs.tolist()]
                print(f"       Top-3: {list(zip(top3_names, top3_conf))}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_correct = 0
    total_samples = 0
    
    for class_name in CLASS_NAMES:
        r = results[class_name]
        if r["total"] > 0:
            acc = r["correct"] / r["total"] * 100
            total_correct += r["correct"]
            total_samples += r["total"]
            print(f"  {class_name:20} : {r['correct']}/{r['total']} ({acc:.1f}%)")
    
    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
        print(f"\n  {'OVERALL':20} : {total_correct}/{total_samples} ({overall_acc:.1f}%)")
    
    # Show confusion for gray light specifically
    print("\n" + "=" * 70)
    print("GRAY LIGHT CONFUSION ANALYSIS")
    print("=" * 70)
    
    if "gray light" in confusion:
        print("  Gray light images were classified as:")
        for pred_class, count in sorted(confusion["gray light"].items(), key=lambda x: -x[1]):
            print(f"    → {pred_class:20} : {count}")
    
    # Show what was misclassified as healthy
    print("\n" + "=" * 70)
    print("ALL MISCLASSIFICATIONS TO 'HEALTHY'")
    print("=" * 70)
    
    for true_class in CLASS_NAMES:
        if true_class != "healthy" and confusion[true_class]["healthy"] > 0:
            print(f"  {true_class:20} → healthy : {confusion[true_class]['healthy']}")


if __name__ == "__main__":
    main()
