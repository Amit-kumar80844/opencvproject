"""
download_papers.py
Autonomously downloads all foundational, tea-specific, and Sri Lankan context 
research paper PDFs into the research_papers/ directory.
"""

import os
import requests
from pathlib import Path
import urllib3

# Disable SSL warnings for some university repositories
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Project root
ROOT = Path(__file__).resolve().parent

# Papers to download: (filename, url, target_subdirectory)
PAPERS = [
    # ══════════════════════════════════════════════════════════════════════════════
    # Section 1: Foundational Architectures (SOTA Justification Papers)
    # ══════════════════════════════════════════════════════════════════════════════
    (
        "UNet_Ronneberger_2015.pdf",
        "https://arxiv.org/pdf/1505.04597.pdf",
        "1_foundational_architectures",
    ),
    (
        "ResNet_He_2015.pdf",
        "https://arxiv.org/pdf/1512.03385.pdf",
        "1_foundational_architectures",
    ),
    (
        "Attention_UNet_Oktay_2018.pdf",
        "https://arxiv.org/pdf/1804.03999.pdf",
        "1_foundational_architectures",
    ),
    (
        "SENet_Hu_2018.pdf",
        "https://arxiv.org/pdf/1709.01507.pdf",
        "1_foundational_architectures",
    ),
    (
        "CBAM_Woo_2018.pdf",
        "https://arxiv.org/pdf/1807.06521.pdf",
        "1_foundational_architectures",
    ),
    # ─── NEW: EfficientNet & SCSE Papers (Ablation Study Justification) ───────────
    (
        "EfficientNet_Tan_2019.pdf",
        "https://arxiv.org/pdf/1905.11946.pdf",  # arXiv:1905.11946
        "1_foundational_architectures",
    ),
    (
        "SCSE_Attention_Roy_2018.pdf",
        "https://arxiv.org/pdf/1803.02579.pdf",  # arXiv:1803.02579 - Concurrent Spatial & Channel SE
        "1_foundational_architectures",
    ),
    (
        "UNet_Plus_Plus_Zhou_2018.pdf",
        "https://arxiv.org/pdf/1807.10165.pdf",  # arXiv:1807.10165 - UNet++: Nested U-Net
        "1_foundational_architectures",
    ),
    (
        "SegmentationModelsPyTorch_Yakubovskiy_2019.pdf",
        "https://arxiv.org/pdf/1911.02855.pdf",  # arXiv:1911.02855 - Encoder-Decoder survey
        "1_foundational_architectures",
    ),
    (
        "FocalLoss_Lin_2017.pdf",
        "https://arxiv.org/pdf/1708.02002.pdf",  # arXiv:1708.02002 - Focal Loss for Dense Object Detection
        "1_foundational_architectures",
    ),
    (
        "TverskyLoss_Salehi_2017.pdf",
        "https://arxiv.org/pdf/1706.05721.pdf",  # arXiv:1706.05721 - Tversky Loss for Imbalanced Segmentation
        "1_foundational_architectures",
    ),
    
    # ══════════════════════════════════════════════════════════════════════════════
    # Section 2: Tea Leaf Disease Detection
    # ══════════════════════════════════════════════════════════════════════════════
    (
        "ML_Tea_Disease_Review_2023.pdf",
        "https://arxiv.org/pdf/2311.03240.pdf",
        "2_tea_disease_detection",
    ),
    (
        "CNN_Tea_Disease_Recognition_2019.pdf",
        "https://arxiv.org/pdf/1901.02694.pdf", 
        "2_tea_disease_detection",
    ),
    (
        "DAONet_YOLOv8_Tea_Detection_2025.pdf",
        "https://arxiv.org/pdf/2511.23222.pdf",
        "2_tea_disease_detection",
    ),
    # ─── NEW: Additional Plant Disease Segmentation Papers ────────────────────────
    (
        "PlantVillage_Dataset_Hughes_2015.pdf",
        "https://arxiv.org/pdf/1511.08060.pdf",  # arXiv:1511.08060 - PlantVillage Dataset
        "2_tea_disease_detection",
    ),
    (
        "DeepPlantPhenomics_Ubbens_2017.pdf",
        "https://www.frontiersin.org/articles/10.3389/fpls.2017.01190/pdf",  # Deep Plant Phenomics
        "2_tea_disease_detection",
    ),
    
    # ══════════════════════════════════════════════════════════════════════════════
    # Section 4: Classification Architecture Papers (EfficientNet-B4 Ablation)
    # ══════════════════════════════════════════════════════════════════════════════
    (
        "ImageNet_Deng_2009.pdf",
        "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5206848",  # ImageNet CVPR paper
        "4_classification_architectures",
    ),
    (
        "TransferLearning_Survey_Pan_2010.pdf",
        "https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf",  # Transfer Learning Survey
        "4_classification_architectures",
    ),
    (
        "AdamW_Loshchilov_2017.pdf",
        "https://arxiv.org/pdf/1711.05101.pdf",  # arXiv:1711.05101 - Decoupled Weight Decay
        "4_classification_architectures",
    ),
    (
        "CosineAnnealingLR_Loshchilov_2017.pdf",
        "https://arxiv.org/pdf/1608.03983.pdf",  # arXiv:1608.03983 - SGDR: Warm Restarts
        "4_classification_architectures",
    ),
    (
        "LabelSmoothing_Szegedy_2016.pdf",
        "https://arxiv.org/pdf/1512.00567.pdf",  # arXiv:1512.00567 - Rethinking Inception
        "4_classification_architectures",
    ),
    (
        "MixUp_Zhang_2018.pdf",
        "https://arxiv.org/pdf/1710.09412.pdf",  # arXiv:1710.09412 - MixUp Data Augmentation
        "4_classification_architectures",
    ),
    (
        "Albumentations_Buslaev_2020.pdf",
        "https://arxiv.org/pdf/1809.06839.pdf",  # arXiv:1809.06839 - Albumentations Library
        "4_classification_architectures",
    ),
    
    # ══════════════════════════════════════════════════════════════════════════════
    # Section 3: Sri Lankan Context (Direct open-access links)
    # ══════════════════════════════════════════════════════════════════════════════
    (
        "KDU_Blister_Blight_Deep_Learning_2024.pdf",
        "https://ir.kdu.ac.lk/bitstream/handle/345/8007/Early%20Detection%20and%20Identification%20of%20Blister%20Blight%20Disease%20in.pdf",
        "3_sri_lankan_context",
    ),
    (
        "IIT_LeafCheck_Deep_Learning_Sri_Lanka.pdf",
        "http://dlib.iit.ac.lk/xmlui/bitstream/handle/123456789/1801/2019740.pdf",
        "3_sri_lankan_context",
    ),
    (
        "SPIS_TS_Tea_Smallholdings_Sri_Lanka.pdf",
        "https://irjiet.com/common_src/article_file/1698328122_e03fcb987a_7_irjiet.pdf",
        "3_sri_lankan_context",
    ),
]

def setup_directories() -> None:
    """Create all necessary directories."""
    directories = [
        "1_foundational_architectures",
        "2_tea_disease_detection",
        "3_sri_lankan_context",
        "4_classification_architectures"
    ]
    for subdir in directories:
        target_dir = ROOT / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

def download_pdf(filename: str, url: str, subdir: str) -> None:
    """Download a single PDF file into the specified subdirectory."""
    target_dir = ROOT / subdir
    filepath = target_dir / filename

    if filepath.exists():
        print(f"  [SKIP] Already exists: {filepath.relative_to(ROOT)}")
        return

    print(f"  [DOWNLOADING] {filename} from {url} ...")
    
    # Using a standard browser user-agent to avoid bot-blocks
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # verify=False is used because some local Sri Lankan university repositories have strict/expired SSL certificates
    response = requests.get(url, headers=headers, timeout=60, stream=True, verify=False)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_kb = filepath.stat().st_size / 1024
    print(f"  [OK] Saved: {filepath.relative_to(ROOT)} ({size_kb:.1f} KB)")

def main() -> None:
    print("=" * 60)
    print("Research Paper PDF Downloader (Fully Automated)")
    print("=" * 60)
    
    setup_directories()
    
    success_count = 0
    fail_count = 0

    for filename, url, subdir in PAPERS:
        try:
            download_pdf(filename, url, subdir)
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")
            fail_count += 1

    print("=" * 60)
    print(f"Done. {success_count} downloaded, {fail_count} failed.")
    print("=" * 60)

if __name__ == "__main__":
    main()