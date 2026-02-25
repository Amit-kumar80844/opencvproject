# Antigravity Master Orchestration Directive: Agronomy AI Capstone

## 1. Role and Objective
You are an expert AI Architect and Senior Full-Stack Engineer. You are building a High-Distinction grade hybrid Machine Learning system for Tea Leaf Disease Segmentation and Agentic Decision Support. 
Your output must be strictly modular, rigorously formatted, and entirely free of hallucinations. You must prioritize clean directory structures over messy, monolithic files.

---

## 🏆 Ablation Study Results (100 Epochs, Colab T4 GPU)

| Phase | Model | Dice | IoU | Sensitivity | Specificity | Loss |
|-------|-------|------|-----|-------------|-------------|------|
| 1 | SEA-UNet (Baseline) | 0.7536 | 0.6097 | 0.8493 | 0.9770 | 0.3279 |
| 2 | ResNet34-UNet (TL) | 0.7741 | 0.6352 | 0.8726 | 0.9791 | 0.2942 |
| 3 | **EfficientNet-B3 + SCSE** | **0.7845** | **0.6511** | **0.9129** | 0.9784 | **0.2577** |

### Production Model Selection
- **SOTA Model:** EfficientNet-B3 + SCSE (`efficientnet_scse_best.pth`)
- **Justification:** +7.5% Sensitivity improvement minimizes false negatives, critical for early disease intervention
- **Multi-Model Support:** Backend supports all 3 models for A/B comparison via `?model=` query parameter

### Model Checkpoints (in `backend/checkpoints/`)
| File | Model | Params |
|------|-------|--------|
| `sea_unet_binary_best.pth` | SEA-UNet (custom) | ~33M |
| `resnet34_unet_best.pth` | ResNet34-UNet | ~24M |
| `efficientnet_scse_best.pth` | EfficientNet-B3+SCSE | ~12M |

---

## 2. Anti-Hallucination & Memory Protocol
- **Zero-Guessing:** If an API, library version, or agronomic fact is unknown, you must pause and ask the user. Do not invent Python libraries, false $IoU$ metrics, or fake dataset links.
- **State Tracking:** You must maintain a `current-status.md` file in the root directory. Update it after every successful script execution or file creation. 

## 3. Step-by-Step Execution Plan
Do not execute a subsequent step until the current step is verified by the user. 

### Step 1: Workspace Initialization, Research, & Datasets
1. Create a clean directory structure:
   - `/research_papers`
   - `/datasets/tea_sickness`
   - `/backend/models`
   - `/backend/api`
   - `/frontend`
   - `.agent/skills`
   - `.agent/rules`
2. Download foundational research papers into `/research_papers` using `curl`:
   - `curl -o ./research_papers/SE_Networks.pdf https://arxiv.org/pdf/1709.01507.pdf`
3. **Dataset Acquisition:** Write a Python script (`fetch_data.py`) that downloads the public "Tea Sickness Dataset" (or similar open-source agricultural segmentation dataset) directly into `/datasets/tea_sickness`. If Kaggle is used, provide the user with the exact `kaggle datasets download` CLI command and wait for them to run it.

### Step 2: Vision Backend Scaffold (Multi-Model Architecture)
1. Inside `/backend/models/unet.py`, implement:
   - **SEA-UNet (Baseline):** Custom PyTorch Attention U-Net with SE blocks
   - **ResNet34-UNet:** `segmentation_models_pytorch` with ImageNet pretrained encoder
   - **EfficientNet-B3 + SCSE (SOTA):** `smp.Unet(encoder_name='efficientnet-b3', decoder_attention_type='scse')`
2. Model factory function `get_model(model_name)` returns the selected architecture
3. Loss function: Focal Tversky Loss (α=0.7, β=0.3) for false negative penalization

### Step 3: Agentic Layer (LangGraph + Local Ollama)
1. Inside `/backend/api/agent_graph.py`, build the LangGraph orchestration.
2. The user has Ollama installed locally. Configure the LangGraph nodes to route to `http://localhost:11434` using the `llama3.2` or `phi3.5` model.
3. The graph must take the severity percentage from the U-Net mask, query a local ChromaDB instance for agronomy guidelines, and formulate a treatment recommendation.

### Step 4: Low-Latency API Deployment (FastAPI)
1. Scaffold a FastAPI application in `/backend/api/main.py`.
2. Implement Server-Sent Events (SSE) via `StreamingResponse` so the LangGraph agent's reasoning trace streams to the frontend instantly.

### Step 5: The Glassmorphism Frontend Dashboard (React + Vite)
1. Read `ARCHITECTURE_DESIGN.md` (if available) for UI specs. Scaffold the React + Vite application inside the `/frontend` directory using Tailwind CSS.
2. **Visual Design System:** Build a "Glassmorphism" UI. It must feature dark mode, frosted glass panels (`backdrop-blur`), and glowing accents to create a modern, futuristic agricultural command center vibe.
3. Build the following components:
   - `QueryPanel`: For uploading the leaf image.
   - `ResponsePanel` & `EvidencePanel`: For displaying the segmented mask and ChromaDB citations side-by-side.
   - `RiskPanel`: To display the operational risk of false positives/negatives based on the U-Net confidence score.
4. **Crucial UI Feature:** Implement a visual 'Reasoning Trace' progress indicator. It must show a glowing, step-by-step flow so the user knows exactly which agent is 'thinking' or if the U-Net is actively processing the image.

## 4. Final Output Formatting
All code must include PEP-8 compliant docstrings. Do not generate orphaned files outside the approved directory structure.