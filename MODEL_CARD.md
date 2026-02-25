# Model Card — TeaVision AI

> **Project:** TeaVision AI — BSc Data Science Capstone (NIBM, 2026)
> **Author:** Manula Fernando
> **Last Updated:** February 2026

---

## Model Overview

TeaVision AI uses a **2-Stage Pipeline** for tea leaf disease detection:

| Stage | Model | Task | Input Size | Output |
|-------|-------|------|------------|--------|
| **Stage 1** | EfficientNet-B3 + SCSE U-Net | Segmentation | 256×256 | Binary mask + severity % |
| **Stage 2** | EfficientNet-B4 | Classification | 512×512 | Disease class (8 classes) |

---

## Stage 1: Segmentation Model

### Architecture

| Component | Specification |
|-----------|---------------|
| **Encoder** | EfficientNet-B3 (ImageNet pretrained) |
| **Decoder** | U-Net with SCSE attention |
| **Attention** | Squeeze & Excitation + Spatial (SCSE) |
| **Parameters** | ~12M |
| **Output** | 1-channel binary mask (disease vs. background) |
| **Framework** | PyTorch + segmentation-models-pytorch |

### Checkpoint

| File | Size | Location |
|------|------|----------|
| `efficientnet_scse_best.pth` | 48 MB | `backend/checkpoints/` |

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Dice Coefficient** | **0.7845** | Overlap measure (1.0 = perfect) |
| **IoU (Jaccard)** | **0.6511** | Intersection over Union |
| **Sensitivity (Recall)** | **0.9129** | True positive rate (minimizes missed disease) |
| **Specificity** | 0.9784 | True negative rate |
| **Training Epochs** | 100 | Google Colab T4 GPU |
| **Best Epoch** | 95 | Early stopping with patience=15 |

### Ablation Study Results

| Model | Dice | IoU | Sensitivity | Loss |
|-------|------|-----|-------------|------|
| SEA-UNet (Baseline) | 0.7536 | 0.6097 | 0.8493 | 0.3279 |
| ResNet34-UNet | 0.7741 | 0.6352 | 0.8726 | 0.2942 |
| **EfficientNet-B3 + SCSE** | **0.7845** | **0.6511** | **0.9129** | **0.2577** |

**Key Finding:** SCSE attention + EfficientNet encoder achieves **+7.5% Sensitivity improvement**, critical for minimizing false negatives (missed disease).

---

## Stage 2: Classification Model

### Architecture

| Component | Specification |
|-----------|---------------|
| **Backbone** | EfficientNet-B4 (ImageNet pretrained) |
| **Input Size** | 512×512 RGB |
| **Output** | 8-class softmax |
| **Parameters** | ~19M |
| **Framework** | PyTorch + timm |

### Checkpoint

| File | Size | Location |
|------|------|----------|
| `tea_leaves_disease_EfficientNetB4_512_model.pth` | 76 MB | `backend/checkpoints/` |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **96.1%** |
| **F1 Score (Weighted)** | **96.0%** |
| **Precision** | 96.0% |
| **Recall** | 96.1% |

### Per-Class Accuracy

| Class | Accuracy | Notes |
|-------|----------|-------|
| Anthracnose | 89.0% | Most confused with bird eye spot |
| Algal Leaf | 99.0% | Near-perfect |
| Bird Eye Spot | 96.0% | Good |
| Brown Blight | 95.0% | Good |
| Gray Light | 96.0% | Good |
| **Healthy** | **100.0%** | Perfect |
| **Red Leaf Spot** | **100.0%** | Perfect |
| White Spot | 95.0% | Good |

### Evaluation Visualizations

See `backend/outputs/` for generated visualizations:
- `confusion_matrix_8x8.png` — 8×8 confusion matrix with counts and percentages
- `roc_curves_multiclass.png` — One-vs-Rest ROC curves for each disease
- `per_class_metrics_table.png` — Detailed precision/recall/F1 breakdown
- `classification_report.txt` — Full sklearn classification report

### Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | Algal Leaf | *Cephaleuros virescens* infection |
| 1 | Anthracnose | *Colletotrichum camelliae* fungal disease |
| 2 | Bird Eye Spot | *Cercospora theae* fungal spots |
| 3 | Brown Blight | *Colletotrichum gloeosporioides* blight |
| 4 | Gray Light | *Pestalotiopsis theae* fungal lesions |
| 5 | **Healthy** | No disease detected |
| 6 | Red Leaf Spot | Multiple pathogens including *Phyllosticta spp.* |
| 7 | White Spot | Fungal white/cream spots |

---

## Training Configuration

### Segmentation Model

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning Rate | 1e-4 (OneCycleLR) |
| Loss Function | Focal Tversky Loss (α=0.3, β=0.7, γ=0.75) |
| Batch Size | 16 |
| Epochs | 100 |
| Early Stopping | patience=15, min_delta=1e-4 |
| Gradient Clipping | max_norm=1.0 |

### Classification Model

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning Rate | 3e-4 (ReduceLROnPlateau) |
| Loss Function | Focal Loss (γ=2.0) |
| Batch Size | 32 |
| Epochs | 50 |
| Label Smoothing | 0.1 |

---

## Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT: Tea Leaf Image (any size, RGB)                                  │
│      │                                                                  │
│      ▼                                                                  │
│  Preprocessing: Resize to 256×256, Normalize (ImageNet stats)          │
│      │                                                                  │
│      ▼                                                                  │
│  Stage 1: EfficientNet-B3+SCSE → Binary Mask + Severity %              │
│      │                                                                  │
│      ├── if severity_pct < 5%  → OUTPUT: "Healthy" (skip Stage 2)      │
│      │                                                                  │
│      ▼ (if severity_pct >= 5%)                                         │
│  Stage 2: EfficientNet-B4 → Disease Class (8 classes)                  │
│      │                                                                  │
│      ▼                                                                  │
│  LangGraph Agent Pipeline → Treatment Recommendation                    │
│      │                                                                  │
│      ▼                                                                  │
│  OUTPUT: Disease class, severity %, treatment plan, risk analysis      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Intended Use

### Primary Use Cases
- ✅ Tea leaf disease detection for Sri Lankan tea plantations
- ✅ Early warning system for disease outbreaks
- ✅ Decision support for fungicide application timing
- ✅ Educational tool for tea farmers and agronomists

### Out-of-Scope Use Cases
- ❌ Medical diagnosis or healthcare applications
- ❌ Real-time drone-based detection (model not optimized for edge deployment)
- ❌ Diseases not in the training set (e.g., blister blight, red rust)

---

## Limitations

### Model Limitations
1. **8 Classes Only**: Cannot detect diseases outside the training classes
2. **Ground-Based Photos**: May not generalize to aerial/drone imagery
3. **Single Leaf Focus**: Performance may degrade on images with multiple leaves
4. **Lighting Sensitivity**: Extreme lighting conditions may affect accuracy
5. **No Weather/Soil/Crop Stage Inputs**: See design rationale below

### Design Decision: Image-Only Detection

**Why weather/soil/crop stage inputs are NOT included:**

While environmental context (temperature, humidity, soil pH, crop growth stage) can improve disease prediction accuracy, this system intentionally focuses on **image-based detection only** for the following reasons:

| Factor | Rationale |
|--------|-----------|
| **Deployment Simplicity** | Farmers only need a smartphone camera, no sensors required |
| **Cost Accessibility** | IoT weather/soil sensors add LKR 50,000+ per installation |
| **Immediate Usability** | No calibration or sensor network setup needed |
| **Scope Definition** | Project focuses on computer vision capabilities for capstone demonstration |
| **Future Extensibility** | Architecture allows adding environmental embeddings in future versions |

**Future Work**: Environmental context can be integrated by:
1. Adding weather API inputs (OpenWeatherMap for field location)
2. Concatenating weather embeddings to classifier feature vector
3. Including crop calendar metadata (days since pruning, flush stage)

This is documented as a **deliberate scope limitation**, not an oversight.

### Data Limitations
1. **Dataset Size**: 885 images may limit generalization
2. **Pseudo-Masks**: GrabCut-generated masks, not expert-annotated
3. **Single Source**: Kaggle dataset may not represent all tea regions

---

## Operational Risk Analysis

### False Positive (FP) Impact
- **Cost**: ~LKR 2,500/hectare (unnecessary pesticide application)
- **Risk**: Ecological damage, pesticide residue on tea, wasted resources
- **Mitigation**: High specificity (0.9784) minimizes FP rate

### False Negative (FN) Impact
- **Cost**: ~LKR 30,000/hectare (yield loss from undetected outbreak)
- **Risk**: Disease spread to neighboring plants, season-level crop loss
- **Mitigation**: High sensitivity (0.9129) prioritized during training

**Design Decision**: Focal Tversky Loss with β=0.7 penalizes FN 2.33× more than FP, reflecting the asymmetric cost structure in real-world tea farming.

---

## Ethical Considerations

### Bias & Fairness
- Model trained on publicly available data; may not represent all tea cultivars
- Recommend validation with local Sri Lankan dataset before deployment

### Environmental Impact
- Proper disease detection can reduce unnecessary pesticide use
- Model promotes precision agriculture and sustainable farming

### Human-in-the-Loop
- **Required**: Agronomist validation before fungicide application
- **Agent provides**: Evidence-backed recommendations with TRI citations
- **User can**: Override, modify, or reject recommendations

---

## Model Versioning

| Version | Date | Changes |
|---------|------|---------|
| v1.0.0 | Feb 2026 | Initial release — EfficientNet-B3+SCSE segmentation |
| v2.0.0 | Feb 2026 | Added EfficientNet-B4 classification (2-stage pipeline) |

---

## References

1. Tan & Le. "EfficientNet: Rethinking Model Scaling." ICML 2019.
2. Roy et al. "Concurrent Spatial and Channel Squeeze & Excitation (SCSE)." MICCAI 2018.
3. Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
4. Salehi et al. "Tversky Loss for Highly Imbalanced Data." MICCAI Workshop 2017.
5. Tea Research Institute of Sri Lanka — Disease Management Guidelines.

---

## Citation

```bibtex
@misc{teavision2026,
  author = {Fernando, Manula},
  title = {TeaVision AI: Automated Tea Leaf Disease Detection with Agentic Advisory},
  year = {2026},
  institution = {NIBM, Sri Lanka},
  type = {BSc Data Science Capstone Project}
}
```

---

*Generated: February 2026*
