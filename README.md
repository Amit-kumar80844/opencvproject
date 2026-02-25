# 🍃 TeaVision AI — Automated Tea Leaf Disease Detection & Treatment Advisory

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-U--Net%20%2B%20SCSE-green?style=for-the-badge" alt="U-Net + SCSE"/>
  <img src="https://img.shields.io/badge/Classifier-EfficientNet--B4-blue?style=for-the-badge" alt="EfficientNet-B4"/>
  <img src="https://img.shields.io/badge/Agents-LangGraph-purple?style=for-the-badge" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/RAG-ChromaDB-orange?style=for-the-badge" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?style=for-the-badge" alt="GitHub Actions"/>
</p>

> **BSc Data Science Capstone Project — NIBM, 2026**  
> Automated Agricultural Image Segmentation for Crop Stress and Pest/Disease Localization

---

## 📋 Project Overview

TeaVision AI is a **2-stage deep learning pipeline** for detecting and classifying tea leaf diseases, combined with an **AI-powered agronomic advisory system** that provides evidence-based treatment recommendations using Sri Lankan Tea Research Institute (TRI) guidelines.

### 🔬 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Tea Leaf Image (256×256 RGB)                                        │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │ Stage 1: U-Net Segmentation (EfficientNet-B3 + SCSE)        │           │
│  │ → Binary disease mask + severity percentage                  │           │
│  │ → Dice: 0.7845 | Sensitivity: 0.9129 (SOTA)                 │           │
│  └─────────────────────────────────────────────────────────────┘           │
│      │                                                                      │
│      ▼ (if severity ≥ 5%)                                                   │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │ Stage 2: Classification (EfficientNet-B4, 512×512)          │           │
│  │ → 8-class disease identification                            │           │
│  │ → Accuracy: 96.1% | F1: 96.0%                               │           │
│  └─────────────────────────────────────────────────────────────┘           │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │ LangGraph Multi-Agent Pipeline                               │           │
│  │ Agent 1: Evidence Retrieval (ChromaDB RAG)                   │           │
│  │ Agent 2: Risk & Contraindication Analysis                    │           │
│  │ Agent 3: Treatment Synthesis (Ollama LLM)                    │           │
│  └─────────────────────────────────────────────────────────────┘           │
│      │                                                                      │
│      ▼                                                                      │
│  OUTPUT: Treatment recommendation + FP/FN cost analysis (LKR)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **U-Net + SCSE Attention** | State-of-the-art segmentation with Squeeze & Excitation blocks |
| **2-Stage Pipeline** | Segmentation → Classification for accurate disease identification |
| **8 Disease Classes** | Algal Leaf, Anthracnose, Bird Eye Spot, Brown Blight, Gray Light, Healthy, Red Leaf Spot, White Spot |
| **ChromaDB RAG** | 3 collections: Guidelines, Treatment Protocols, Contraindications |
| **LangGraph Agents** | Multi-agent workflow for evidence-grounded recommendations |
| **Operational Risk KPIs** | FP/FN cost estimation in Sri Lankan Rupees (LKR) |
| **Real-time SSE Streaming** | Progressive UI updates during analysis |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Ollama** (for LLM inference)

### Installation

```bash
# Clone repository
git clone https://github.com/Manula-Fernando/vision-unet-segmentation-for-tea-leaves.git
cd vision-unet-segmentation-for-tea-leaves

# Create Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install Python dependencies
pip install torch torchvision segmentation-models-pytorch chromadb fastapi uvicorn ollama pillow numpy

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Download Ollama Model (Required for Agent)

```bash
# Download the fast qwen2:1.5b model (recommended)
ollama pull qwen2:1.5b

# Alternative models (slower but also work)
ollama pull phi3
ollama pull llama3.2
```

### Initialize ChromaDB

```bash
cd backend/api
python rag_store.py --reset-all
```

### Run the Application

```bash
# Terminal 1: Start backend
cd backend/api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 📊 Model Performance

### Segmentation (100-Epoch Ablation Study)

| Model | Dice | Sensitivity | IoU |
|-------|------|-------------|-----|
| ResNet-34 U-Net | 0.7012 | 0.8456 | 0.5789 |
| EfficientNet-B3 U-Net | 0.7523 | 0.8891 | 0.6234 |
| **EfficientNet-B3 + SCSE** | **0.7845** | **0.9129** | **0.6567** |

### Classification (EfficientNet-B4, 512×512)

| Metric | Score |
|--------|-------|
| **Accuracy** | **96.1%** |
| **F1 Score (Weighted)** | **96.0%** |
| Precision | 96.0% |
| Recall | 96.1% |

### Per-Class Accuracy

| Class | Accuracy |
|-------|----------|
| Anthracnose | 89.0% |
| Algal Leaf | 99.0% |
| Bird Eye Spot | 96.0% |
| Brown Blight | 95.0% |
| Gray Light | 96.0% |
| **Healthy** | **100.0%** |
| **Red Leaf Spot** | **100.0%** |
| White Spot | 95.0% |

### Evaluation Visualizations

Generated visualizations are saved in `backend/outputs/`:
- `confusion_matrix_8x8.png` — 8×8 confusion matrix with counts and percentages
- `roc_curves_multiclass.png` — One-vs-Rest ROC curves for each disease class
- `per_class_metrics_table.png` — Detailed precision/recall/F1 breakdown table
- `classification_report.txt` — Full sklearn classification report

---

## 🗃️ ChromaDB Knowledge Architecture

| Collection | Documents | Purpose |
|------------|-----------|---------|
| `tea_agronomy_guidelines` | 19 | TRI guidelines, MRL limits, regulatory info |
| `tea_treatment_protocols` | 8 | Disease-specific treatment protocols |
| `tea_contraindications` | 8 | Safety restrictions, PHI days, warnings |

### CLI Commands

```bash
# Check status of all collections
python rag_store.py --status

# Reset and re-ingest all collections
python rag_store.py --reset-all

# Query treatment protocol
python rag_store.py --treatment "brown blight"

# Query contraindications
python rag_store.py --contraindication "algal leaf"
```

---

## 📁 Project Structure

```
vision-unet-segmentation-for-tea-leaves/
├── backend/
│   ├── api/
│   │   ├── main.py           # FastAPI endpoints + SSE streaming
│   │   ├── agent_graph.py    # LangGraph multi-agent pipeline
│   │   ├── rag_store.py      # ChromaDB vector store management
│   │   └── logger.py         # JSON logging system
│   ├── models/
│   │   ├── unet.py           # U-Net + SCSE architecture
│   │   ├── dataset.py        # Data loading + augmentation
│   │   ├── metrics.py        # Dice, IoU, FP/FN risk calculation
│   │   └── train.py          # Training loop
│   ├── checkpoints/          # Model weights (.pth)
│   ├── chroma_db/            # ChromaDB persistent storage
│   └── tests/
│       └── test_teavision.py # Unit tests (pytest)
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   │       ├── MaskVisualizationPanel.jsx
│   │       ├── AgentAdvicePanel.jsx
│   │       ├── ReasoningTrace.jsx
│   │       ├── SettingsPanel.jsx
│   │       └── LogConsole.jsx
│   └── package.json
├── notebooks/
│   ├── colab_tea_segmentation_v2.ipynb   # GPU training notebook
│   └── tea-leaf-disease-classifier/      # EfficientNet-B4 classifier
├── datasets/
│   └── tea_sickness/                     # 8-class disease dataset
├── research_papers/
│   ├── LITERATURE_REVIEW.md
│   └── download_papers.py
├── ARCHITECTURE_DIAGRAMS.md    # System architecture (Mermaid)
├── COMPLIANCE_ETHICS.md        # Regulatory & ethics documentation
├── DATASET_CARD.md             # Dataset documentation
├── MODEL_CARD.md               # Model documentation
├── Agent.md                    # Agent architecture documentation
├── current-status.md           # Project progress tracker
└── PROJECT_RUBRIC_ANALYSIS.md  # Rubric self-assessment
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) | System architecture diagrams (Mermaid) |
| [COMPLIANCE_ETHICS.md](COMPLIANCE_ETHICS.md) | Regulatory compliance & AI ethics |
| [DATASET_CARD.md](DATASET_CARD.md) | Dataset documentation (ML best practice) |
| [MODEL_CARD.md](MODEL_CARD.md) | Model documentation (ML best practice) |
| [LITERATURE_REVIEW.md](research_papers/LITERATURE_REVIEW.md) | Academic literature review |
| [Agent.md](Agent.md) | LangGraph agent architecture |
| [current-status.md](current-status.md) | Project progress & milestones |
| [PROJECT_RUBRIC_ANALYSIS.md](PROJECT_RUBRIC_ANALYSIS.md) | Rubric self-assessment |

### Testing & CI/CD

```bash
# Run unit tests
pytest backend/tests/test_teavision.py -v

# Run with coverage
pytest backend/tests/test_teavision.py --cov=backend

# Run integration tests only
pytest backend/tests/test_teavision.py -v -k "integration"
```

#### GitHub Actions CI/CD Pipeline

The project includes automated CI/CD via `.github/workflows/ci.yml`:
- **Backend Tests**: flake8 linting + pytest unit tests
- **Frontend Tests**: npm build + lint
- **Model Validation**: Forward-pass validation of trained models

Triggered on: Push to `main`/`develop`, Pull Requests

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `PREFERRED_MODEL` | `qwen2:1.5b` | Fast LLM model for agent |
| `FALLBACK_MODEL` | `phi3` | Fallback if preferred unavailable |
| `FAST_MODE_ENABLED` | `True` | Use optimized <30s agent pipeline |

---

### Model Checkpoints

Place model weights in `backend/checkpoints/`:
- `efficientnet_scse_best.pth` — Segmentation model
- `tea_leaves_disease_EfficientNetB4_512_model.pth` — Classifier

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Manula Fernando**  
BSc Data Science — NIBM, 2026

---

## 🙏 Acknowledgments

- Tea Research Institute of Sri Lanka (TRI) for agronomic guidelines
- NIBM faculty for project supervision
- Kaggle dataset: [Identifying Disease in Tea Leafs](https://www.kaggle.com/datasets/shashwatwork/identifying-disease-in-tea-leafs)

---

<p align="center">
  <i>Built with ❤️ for Sri Lankan tea cultivation</i>
</p>
