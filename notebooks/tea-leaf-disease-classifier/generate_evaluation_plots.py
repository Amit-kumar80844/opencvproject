"""
generate_evaluation_plots.py — Generate Classification Evaluation Visualizations
================================================================================
BSc Data Science Capstone — NIBM, 2026
Author  : Manula Fernando

This script generates:
1. 8×8 Confusion Matrix visualization
2. ROC curves (One-vs-Rest for each class)
3. Per-class metrics table
4. Classification report

Run: python generate_evaluation_plots.py
Output: Saved to backend/outputs/
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.preprocessing import label_binarize

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

IMG_SIZE = 512
CLASS_NAMES = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight',
               'gray light', 'healthy', 'red leaf spot', 'white spot']
NUM_CLASSES = len(CLASS_NAMES)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "tea_sickness" / "tea sickness dataset"
MODEL_PATH = PROJECT_ROOT / "backend" / "checkpoints" / "tea_leaves_disease_EfficientNetB4_512_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "outputs"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ════════════════════════════════════════════════════════════════════════════
# Transforms
# ════════════════════════════════════════════════════════════════════════════

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ════════════════════════════════════════════════════════════════════════════
# Load Model
# ════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load the EfficientNet-B4 classifier."""
    print(f"Loading model from: {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_all_images(model, max_per_class=None):
    """Evaluate model on all images and collect predictions with probabilities."""
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("\nEvaluating images...")
    
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = DATASET_DIR / cls_name
        if not cls_dir.exists():
            print(f"WARNING: Directory not found: {cls_dir}")
            continue
        
        image_files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        
        if max_per_class:
            image_files = random.sample(image_files, min(max_per_class, len(image_files)))
        
        for img_path in tqdm(image_files, desc=f"[{cls_name}]"):
            try:
                # Load and preprocess
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                augmented = val_transform(image=image)
                input_tensor = augmented['image'].unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(probs, 1)
                
                all_labels.append(cls_idx)
                all_preds.append(predicted.item())
                all_probs.append(probs.cpu().numpy()[0])
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ════════════════════════════════════════════════════════════════════════════
# Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save 8x8 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize for percentage display
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 10))
    
    # Create annotations with count and percentage
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            if count > 0:
                annot[i, j] = f'{count}\n({percent:.1f}%)'
            else:
                annot[i, j] = '0'
    
    # Plot heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix — EfficientNet-B4 Tea Disease Classification\n'
              f'Overall Accuracy: {accuracy_score(y_true, y_pred)*100:.1f}%',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {save_path}")
    return cm


# ════════════════════════════════════════════════════════════════════════════
# ROC Curves
# ════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(y_true, y_probs, save_path):
    """Generate and save ROC curves for each class (One-vs-Rest)."""
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, NUM_CLASSES))
    
    # Calculate ROC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    # Plot each class
    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    # Plot micro-average
    plt.plot(fpr_micro, tpr_micro, color='black', lw=3, linestyle='--',
             label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Chance (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — One-vs-Rest Multi-Class Classification\n'
              'EfficientNet-B4 Tea Disease Classifier',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curves saved to: {save_path}")
    return roc_auc


# ════════════════════════════════════════════════════════════════════════════
# Per-Class Metrics Table
# ════════════════════════════════════════════════════════════════════════════

def generate_metrics_table(y_true, y_pred, save_path):
    """Generate per-class metrics table and save as image."""
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(NUM_CLASSES)
    )
    
    # Calculate accuracy per class
    cm = confusion_matrix(y_true, y_pred)
    accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Table data
    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Support']
    table_data = []
    
    for i, cls_name in enumerate(CLASS_NAMES):
        table_data.append([
            cls_name,
            f'{precision[i]:.4f}',
            f'{recall[i]:.4f}',
            f'{f1[i]:.4f}',
            f'{accuracy[i]*100:.1f}%',
            f'{support[i]}'
        ])
    
    # Add weighted average row
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    total_acc = accuracy_score(y_true, y_pred)
    table_data.append([
        'Weighted Avg',
        f'{w_prec:.4f}',
        f'{w_rec:.4f}',
        f'{w_f1:.4f}',
        f'{total_acc*100:.1f}%',
        f'{len(y_true)}'
    ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.1]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Header style
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#2E7D32')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Highlight weighted average row
    for j in range(len(headers)):
        table[(len(CLASS_NAMES) + 1, j)].set_facecolor('#E8F5E9')
        table[(len(CLASS_NAMES) + 1, j)].set_text_props(fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(CLASS_NAMES) + 1):
        color = '#F5F5F5' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Per-Class Classification Metrics — EfficientNet-B4\n'
              'Tea Leaf Disease Classification',
              fontsize=14, fontweight='bold', pad=20, y=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics table saved to: {save_path}")
    return precision, recall, f1, accuracy


# ════════════════════════════════════════════════════════════════════════════
# Generate Classification Report
# ════════════════════════════════════════════════════════════════════════════

def save_classification_report(y_true, y_pred, save_path):
    """Save detailed classification report as text file."""
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT — EfficientNet-B4 Tea Disease Classifier\n")
        f.write("="*70 + "\n\n")
        f.write(report)
        f.write("\n" + "="*70 + "\n")
        f.write(f"Overall Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%\n")
        f.write(f"Total Samples: {len(y_true)}\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Classification report saved to: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("TeaVision AI — Classification Evaluation Visualizations")
    print("="*70)
    
    # Load model
    model = load_model()
    
    # Evaluate all images (or set max_per_class for faster testing)
    y_true, y_pred, y_probs = evaluate_all_images(model, max_per_class=None)
    
    print(f"\nTotal samples evaluated: {len(y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Confusion Matrix
    cm_path = OUTPUT_DIR / "confusion_matrix_8x8.png"
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # 2. ROC Curves
    roc_path = OUTPUT_DIR / "roc_curves_multiclass.png"
    plot_roc_curves(y_true, y_probs, roc_path)
    
    # 3. Per-Class Metrics Table
    metrics_path = OUTPUT_DIR / "per_class_metrics_table.png"
    generate_metrics_table(y_true, y_pred, metrics_path)
    
    # 4. Classification Report (text)
    report_path = OUTPUT_DIR / "classification_report.txt"
    save_classification_report(y_true, y_pred, report_path)
    
    print("\n" + "="*70)
    print("✓ All visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
