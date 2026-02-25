# Dataset Card — Tea Leaf Disease Dataset

> **Project:** TeaVision AI — BSc Data Science Capstone (NIBM, 2026)
> **Author:** Manula Fernando
> **Last Updated:** February 2026

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Name** | Tea Sickness Dataset |
| **Source** | Kaggle — [`shashwatwork/identifying-disease-in-tea-leafs`](https://www.kaggle.com/datasets/shashwatwork/identifying-disease-in-tea-leafs) |
| **License** | CC BY 4.0 (Public Domain, Attribution Required) |
| **Total Images** | 885 |
| **Image Type** | Ground-based RGB photographs |
| **Resolution** | Variable (resized to 256×256 for segmentation, 512×512 for classification) |
| **Format** | JPEG/PNG |

---

## Class Distribution

| Class | Count | Percentage | Description |
|-------|-------|------------|-------------|
| Algal Leaf | 105 | 11.9% | Velvety green/orange patches from *Cephaleuros virescens* |
| Anthracnose | 122 | 13.8% | Brown/black lesions from *Colletotrichum camelliae* |
| Bird Eye Spot | 95 | 10.7% | Small circular spots with gray center |
| Brown Blight | 118 | 13.3% | Brown lesions from *Colletotrichum gloeosporioides* |
| Gray Light | 108 | 12.2% | Gray/silver lesions from *Pestalotiopsis theae* |
| **Healthy** | 147 | 16.6% | Disease-free tea leaves |
| Red Leaf Spot | 91 | 10.3% | Red/brown spots with yellow halo |
| White Spot | 99 | 11.2% | Fungal white/cream colored spots |
| **Total** | **885** | **100%** | |

### Class Imbalance Handling
- **WeightedRandomSampler**: Inverse-frequency weights applied during training
- **Focal Tversky Loss**: α=0.7, β=0.3 to penalize false negatives
- **Data Augmentation**: Heavy augmentation to simulate diverse field conditions

---

## Data Splits

| Split | Percentage | Count | Purpose |
|-------|------------|-------|---------|
| Training | 70% | 619 | Model training |
| Validation | 15% | 133 | Hyperparameter tuning, early stopping |
| Test | 15% | 133 | Final evaluation (held-out) |

- **Stratification**: Preserved class ratios across all splits
- **Random Seed**: Fixed for reproducibility (seed=42)

---

## Data Collection

### Source Context
- **Origin**: Field photography from tea plantations
- **Capture Method**: Handheld camera, natural lighting
- **Geographic Context**: Tea-growing regions (applicable to Sri Lankan tea estates)

### Quality Considerations
- **Labeling**: Human-annotated by agricultural experts
- **Ground Truth Masks**: Generated using GrabCut algorithm with disease-specific HSV color windows (pseudo-segmentation)
- **Potential Noise**: Some images may have multiple symptoms, varying illumination, or partial occlusion

---

## Preprocessing Pipeline

### For Segmentation (256×256)
1. **Resize**: Bicubic interpolation to 256×256
2. **Normalize**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Mask Generation**: GrabCut algorithm with per-disease HSV thresholds

### For Classification (512×512)
1. **Resize**: Bicubic interpolation to 512×512
2. **Normalize**: ImageNet statistics
3. **Center Crop**: Applied during validation

---

## Data Augmentation (Training Only)

| Augmentation | Purpose | Parameters |
|--------------|---------|------------|
| RandomBrightnessContrast | Illumination variation | limit=0.2, p=0.5 |
| HueSaturationValue | Color variation | h_shift=10, s_shift=20, v_shift=20 |
| CLAHE | Contrast enhancement | clip_limit=4.0 |
| GaussNoise | Robustness to sensor noise | p=0.3 |
| ElasticTransform | Noisy annotation tolerance | α=80, σ=10 |
| GridDistortion | Leaf shape variation | num_steps=5 |
| HorizontalFlip | Spatial invariance | p=0.5 |
| VerticalFlip | Spatial invariance | p=0.5 |
| RandomRotate90 | Orientation invariance | p=0.5 |
| ShiftScaleRotate | Spatial jitter | shift=0.1, scale=0.1, rotate=15° |
| CoarseDropout (CutOut) | Occlusion robustness | 8 holes, 16×16 pixels |

---

## Limitations & Known Issues

### Dataset Limitations
1. **No Drone/UAV Imagery**: Only ground-based photos; model may not generalize to aerial views
2. **Single Region**: Dataset may not represent all tea cultivar varieties globally
3. **No Temporal Data**: Single snapshots, no progression tracking
4. **No Severity Labels**: Original dataset only has class labels, not severity grades

### Annotation Limitations
1. **Classification Only**: Original labels are class-level, not pixel-level
2. **Pseudo-Masks**: GrabCut-generated masks are approximations, not expert-annotated
3. **Multi-Symptom Images**: Some leaves may exhibit multiple disease symptoms

---

## Ethical Considerations

### Data Privacy
- ✅ No personal data or farm-identifiable information
- ✅ Publicly available dataset with permissive license

### Bias & Fairness
- ⚠️ Dataset may underrepresent rare disease variants
- ⚠️ Color-based features may be affected by camera calibration

### Recommended Use
- ✅ Research and educational purposes
- ✅ Tea disease detection system development
- ⚠️ Should be validated with local data before field deployment

---

## References

1. Kaggle Dataset: https://www.kaggle.com/datasets/shashwatwork/identifying-disease-in-tea-leafs
2. Tea Research Institute of Sri Lanka (TRI) Disease Guidelines
3. PlantVillage Dataset Methodology (Hughes & Salathé, 2015)

---

*Generated: February 2026*
