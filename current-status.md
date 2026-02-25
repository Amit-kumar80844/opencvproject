# Current Status — Tea Leaf Disease Segmentation Capstone

> **Last Updated:** 2025-06-23 (Step 11 — Final Improvements & CI/CD)
> **Phase:** Step 11 — Production Improvements & CI/CD Pipeline
> **Status:** ✅ ALL IMPROVEMENTS COMPLETE — READY FOR TECHNICAL REPORT
> **SOTA Model:** EfficientNet-B3 + SCSE (Segmentation) + EfficientNet-B4 (Classification)
> **Pipeline:** 2-Stage — Segmentation (256px) → Classification (512px) → LangGraph Agent

---

## 🎯 Latest Updates (2025-06-23)

### Step 11: Final Improvements & CI/CD Pipeline ✅

#### Classification Model Performance Update

| Metric | Previous | Updated |
|--------|----------|---------|
| **Accuracy** | 94.2% | **96.1%** |
| **F1 Score (Weighted)** | 94.0% | **96.0%** |
| Precision | 93.8% | 96.0% |
| Recall | 94.2% | 96.1% |

**Per-Class Accuracy:**
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

#### New Files Created

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | GitHub Actions CI/CD pipeline (backend tests, frontend build, model validation) |
| `backend/generate_evaluation_plots.py` | Script to generate confusion matrix, ROC curves, per-class metrics visualizations |

#### Evaluation Visualizations Generated

| Output File | Description |
|-------------|-------------|
| `backend/outputs/confusion_matrix_8x8.png` | 8×8 confusion matrix with counts & percentages |
| `backend/outputs/roc_curves_multiclass.png` | One-vs-Rest ROC curves for each disease class |
| `backend/outputs/per_class_metrics_table.png` | Detailed precision/recall/F1 breakdown table |
| `backend/outputs/classification_report.txt` | Full sklearn classification report |

#### Integration Tests Added

| Test | Scope |
|------|-------|
| `test_two_stage_pipeline_with_sample_image` | Segmentation → classification pipeline with synthetic image |
| `test_end_to_end_with_real_sample_image` | Full pipeline with real dataset image |
| `test_api_analyze_endpoint_integration` | FastAPI /analyze endpoint with real image upload |
| `test_chromadb_to_agent_integration` | ChromaDB RAG → Agent pipeline integration |
| `TestClassificationAblation` | Ablation study comparing EfficientNet-B4 vs ResNet-50 |

#### Research Papers Downloaded

| Section | New Papers |
|---------|------------|
| `4_classification_architectures/` | AdamW, CosineAnnealingLR, LabelSmoothing, MixUp, Albumentations, ImageNet pretrained, Transfer learning |

#### Documentation Updates

| File | Updates |
|------|---------|
| `MODEL_CARD.md` | Updated metrics to 96.1%, added per-class accuracy, visualization references |
| `README.md` | Updated metrics, added CI/CD badge, per-class accuracy, testing section |
| `LITERATURE_REVIEW.md` | Added Section 8: Classification Architecture Papers & Ablation Study |
| `current-status.md` | Added Step 11 completion log |

#### CI/CD GitHub Actions Pipeline

```yaml
# .github/workflows/ci.yml
Jobs:
1. backend-tests: Python 3.10 + flake8 + pytest
2. frontend-tests: Node.js 18 + npm install + npm build
3. model-validation: PyTorch forward-pass validation on trained models
```

---

## 📋 Previous Updates (2025-06-23)

### ChromaDB Knowledge Architecture ✅
Migrated ALL knowledge stores to ChromaDB vector database for semantic retrieval:

| Collection | Purpose | Documents |
|------------|---------|-----------|
| `tea_agronomy_guidelines` | TRI guidelines, MRL data, regulatory info | 18 passages |
| `tea_treatment_protocols` | Disease-specific treatment protocols | 8 protocols (1 per disease) |
| `tea_contraindications` | Safety restrictions, PHI days, warnings | 8 entries (1 per disease) |

**Architecture Benefits:**
- **Semantic Search:** Vector similarity for relevant passage retrieval
- **Scalable Storage:** Easy to add new guidelines without code changes
- **Fault Tolerance:** Hardcoded fallback dictionaries if ChromaDB unavailable
- **Professional Design:** Follows RAG (Retrieval-Augmented Generation) best practices

### Files Modified

| File | Changes |
|------|---------|
| `backend/api/rag_store.py` | Added 3 ChromaDB collections, new query functions, CLI support |
| `backend/api/agent_graph.py` | Updated to query ChromaDB first with fallback to hardcoded dicts |

### New Functions in rag_store.py

```python
# Collection 1: Guidelines
get_or_create_store()          # Initialize guidelines collection
query_guidelines()             # Semantic search for treatment guidelines

# Collection 2: Treatment Protocols  
get_or_create_treatment_store()    # Initialize treatment collection
query_treatment_protocol()         # Retrieve disease-specific treatment

# Collection 3: Contraindications
get_or_create_contraindications_store()  # Initialize contraindications
query_contraindications()                # Retrieve PHI, restrictions
get_quick_contraindications_from_db()    # Formatted string output

# Unified
initialize_all_collections()   # Initialize all 3 collections at once
```

### CLI Commands

```bash
# Check status of all collections
python -m backend.api.rag_store --status

# Reset and re-ingest all collections
python -m backend.api.rag_store --reset-all

# Query treatment protocol
python -m backend.api.rag_store --treatment "brown blight"

# Query contraindications
python -m backend.api.rag_store --contraindication "gray light"

# Query guidelines
python -m backend.api.rag_store --query "anthracnose|RED"
```

### LangGraph Agent Updates
Updated agent workflow to use ChromaDB-first architecture:

1. **`get_treatment_knowledge()`** — Queries ChromaDB `tea_treatment_protocols`, falls back to hardcoded dict
2. **`_get_quick_contraindications()`** — Queries ChromaDB `tea_contraindications`, falls back to hardcoded dict
3. **Agent trace** — Now includes KB source (ChromaDB/Fallback/Default)

---

## 📋 Previous Updates (2025-06-23)

### EfficientNet-B4 Classifier Upgrade ✅
Replaced ResNet-50 classifier with EfficientNet-B4 for improved accuracy:
- **Architecture:** EfficientNet-B4 via `timm` library
- **Input Size:** 512×512 (up from 224×224)
- **Training:** Focal Loss for class imbalance, light realistic augmentation
- **Improvements:** Better algal leaf vs anthracnose discrimination

| Component | Previous | Updated |
|-----------|----------|---------|
| Classification Model | ResNet-50 (224px) | EfficientNet-B4 (512px) |
| Checkpoint | `tea_leaves_disease_classification_ResNet_model.pth` | `tea_leaves_disease_EfficientNetB4_512_model.pth` |
| Training Loss | CrossEntropy | Focal Loss (γ=2.0) |

### Agent Recommendation Diversity ✅
Fixed issue where agent gave similar recommendations for different diseases:
- **Disease-Specific Knowledge Base:** Added `DISEASE_TREATMENT_KB` with pathogen info, symptoms, treatments, and severity-based actions for all 8 disease classes
- **Enhanced Prompts:** LLM prompt now includes disease-specific pathogen, symptoms, recommended fungicides, and severity-appropriate actions
- **Rule-Based Fallback:** Improved fallback responses use disease-specific knowledge when LLM unavailable

### Agent Performance Optimization ✅
Optimized Ollama LLM settings for faster responses:
- **Preferred Model:** Changed from `phi3` to `qwen2:1.5b` (ultra-fast)
- **Multi-threading:** Increased `num_thread` from 4 → 8
- **GPU Utilization:** Added `num_gpu=99` to maximize GPU layers
- **Context Window:** Reduced from 1536 → 1024 for faster processing
- **Timeout:** Reduced from 120s → 60s

### UI Enhancements ✅
- **Settings Panel:** New glassmorphism settings modal with:
  - Glass transparency slider (0-20%)
  - Animations toggle
  - Color theme selector (Tea Garden / Ocean Mist / Sunset Gold)
  - Cost help toggle
- **FP/FN Cost Explanations:** Enhanced tooltips with detailed explanations:
  - "LKR = Sri Lankan Rupees"
  - "FP Cost: Unnecessary treatment if healthy areas flagged as diseased"
  - "FN Cost: Potential crop loss if diseased areas are missed"
- **Settings Button:** Added gear icon in header for easy access
- **Footer Update:** Reflects EfficientNet-B4 classifier

### Files Modified

| File | Changes |
|------|---------|
| `backend/models/unet.py` | Updated `CLASSIFIER_CONFIG` for EfficientNet-B4, new `create_classifier()` using timm |
| `backend/api/main.py` | Updated classifier loading and resize to 512×512 |
| `backend/api/agent_graph.py` | Added `DISEASE_TREATMENT_KB`, updated prompts, faster LLM settings |
| `frontend/src/App.jsx` | Added settings button and SettingsPanel integration |
| `frontend/src/components/SettingsPanel.jsx` | New component for UI customization |
| `frontend/src/components/AgentAdvicePanel.jsx` | Enhanced FP/FN cost explanations |
| `frontend/src/components/MaskVisualizationPanel.jsx` | Updated pipeline indicator |
| `frontend/src/index.css` | Added slider styles, tooltip CSS, reduced motion support |

---

## 📋 Previous Updates (2025-06-22)

### 2-Stage Pipeline Implementation ✅
Implemented a production-grade 2-stage inference pipeline:
- **Stage 1:** EfficientNet-B3 + SCSE segmentation (256×256) → Binary mask + severity%
- **Stage 2:** EfficientNet-B4 classification (512×512) → Disease class (8 classes)

| Component | Details |
|-----------|---------|
| Segmentation Model | EfficientNet-B3 + SCSE (Dice: 0.7845) |
| Classification Model | EfficientNet-B4 (512px, Focal Loss) |
| Threshold | Severity ≥ 5% triggers classification |
| Integration | Unified `_run_inference()` in main.py |

### Agent Response Improvements ✅
- **Fixed truncation**: Increased `num_predict` from 300 → 500 tokens
- **Structured output**: LLM now generates 4 clear sections (IMMEDIATE ACTION, TREATMENT, MONITORING, PRECAUTIONS)
- **Frontend parsing**: New `FormattedNarrative` component renders sections with icons and colors

### UI Enhancements ✅
- **Enhanced glassmorphism**: Increased blur (20px), added saturation filter
- **Plantation theme**: Dark green gradient background, subtle leaf SVG pattern
- **Modern effects**: Enhanced shadows, smoother hover transitions
- **Narrative styling**: Section cards with color-coded headers and emoji icons

---

## 🏆 Ablation Study Summary (100 Epochs)

| Model | Dice | IoU | Sensitivity | Specificity | Loss |
|-------|------|-----|-------------|-------------|------|
| SEA-UNet (Baseline) | 0.7536 | 0.6097 | 0.8493 | 0.9770 | 0.3279 |
| ResNet34-UNet (TL) | 0.7741 | 0.6352 | 0.8726 | 0.9791 | 0.2942 |
| **EfficientNet-B3 + SCSE** | **0.7845** | **0.6511** | **0.9129** | 0.9784 | **0.2577** |

> **Key Finding:** SCSE attention + EfficientNet transfer learning achieves **+7.5% Sensitivity** improvement, critical for minimizing false negatives in disease detection.

---

## Step 1 Completion Log ✅ (archived)

Step 1 fully complete — see prior entries.  Summary: 11/11 PDFs downloaded,
885-image dataset loaded, LITERATURE_REVIEW.md updated.

---

## Step 2 Completion Log ✅

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `backend/models/dataset.py` | ~500 | Dataset loader, GrabCut mask generation, albumentations augmentation pipeline, DataLoader factory with WeightedRandomSampler |
| `backend/models/unet.py` | ~430 | SE-Attention U-Net (SEA-UNet): SEBlock, AttentionGate, ConvBlock (residual+SE), EncoderBlock, DecoderBlock, full SEAUNet class + architecture diagram |
| `backend/models/metrics.py` | ~520 | DiceLoss, CombinedDiceBCELoss, MultiClassCombinedLoss, compute_all_metrics (IoU/Dice/Sensitivity/Specificity/Precision), MetricsTracker, calculate_operational_risk(), visualisation functions |
| `backend/models/train.py` | ~480 | Full training loop (AdamW + OneCycleLR), LiveMetricsPlotter (real-time curves), EarlyStopping, Optuna HP tuning, architecture verification __main__ |
| `backend/models/__init__.py` | ~30 | Package exports |

### Task 2.1 — Dataset & Augmentation ✅
- **TeaLeafSegmentationDataset**: loads 885 real images from 8-class folder structure
- **GrabCut mask generation**: pseudo-segmentation masks generated on-the-fly (no ground truth needed); per-disease HSV colour windows for disease-specific pixel labeling
- **Augmentation pipeline** (albumentations):
  - *Illumination variation*: RandomBrightnessContrast, HueSaturationValue, CLAHE, GaussNoise
  - *Noisy annotation robustness*: ElasticTransform (α=80, σ=10), GridDistortion
  - *Spatial*: HFlip, VFlip, RandomRotate90, ShiftScaleRotate
  - *Occlusion*: CoarseDropout (CutOut)
- **WeightedRandomSampler**: inverse-frequency weights for class-imbalance handling
- **Stratified train/val/test split**: 70% / 15% / 15%
- **Verification**: 885 samples ✓ | Image: (3,256,256) ✓ | Mask: (1,256,256) ✓

### Task 2.2 — SEA-UNet Architecture ✅
- **SEBlock**: Channel-wise Squeeze-and-Excitation (global avg pool → FC → sigmoid) [Hu et al. 2018]
- **AttentionGate**: Additive attention gating on skip connections (W_x + W_g → ψ sigmoid) [Oktay et al. 2018]
- **ConvBlock**: Double Conv(3×3) + BN + ReLU + SE block + residual connection [He et al. 2016]
- **EncoderBlock**: ConvBlock + MaxPool2d; **DecoderBlock**: TransposedConv + AG + concat + ConvBlock
- **SEAUNet**: 33,006,452 trainable parameters; Kaiming He weight initialisation
- **Architecture verification** — 4/4 shape tests PASSED:
  - (1,3,256,256) → (1,8,256,256) ✓  | (2,3,256,256) → (2,8,256,256) ✓
  - (1,3,128,128) → (1,8,128,128) ✓  | (4,3,256,256) → (4,8,256,256) ✓
- Architecture diagram saved: `backend/outputs/sea_unet_architecture.png`

### Task 2.3 — Loss Functions & Metrics ✅
- **DiceLoss** (smooth Dice, sigmoid-safe) — addresses class imbalance directly
- **CombinedDiceBCELoss** (α=0.5) — fast BCE convergence + balanced Dice correction
- **MultiClassDiceLoss / MultiClassCombinedLoss** — macro-average across 8 classes
- **Rubric metrics** (strictly formulated):
  - `calculate_iou()` — IoU = TP/(TP+FP+FN) ✓
  - `calculate_dice()` — Dice = 2TP/(2TP+FP+FN) ✓
  - `calculate_sensitivity()` — Sensitivity = TP/(TP+FN) ✓
  - `calculate_specificity()` — Specificity = TN/(TN+FP) ✓
- **`calculate_operational_risk(fp, fn)`** — quantifies:
  - FP risk: unnecessary pesticide cost (LKR) × ecological multiplier
  - FN risk: missed outbreak → propagation → yield loss (asymmetric 12× multiplier)
  - Composite score 0–100 → GREEN/AMBER/RED/CRITICAL tier + recommended action
  - Returns structured dict for LangGraph agent (Step 3)
- Visualisations: metrics dashboard (6-panel), operational risk card saved

### Task 2.4 — Training Loop & Verification ✅
- **Full training loop**: train_one_epoch + validate_one_epoch
- **Optimiser**: AdamW with decoupled weight decay [Loshchilov & Hutter, 2019]
- **Scheduler**: OneCycleLR with cosine annealing [Smith & Topin, 2018]
- **Gradient clipping**: max_norm=1.0 for stability
- **LiveMetricsPlotter**: 6-panel real-time matplotlib dashboard (headless-safe)
- **EarlyStopping**: patience=10 with mode=min on validation loss
- **Optuna HP tuning**: TPE sampler + MedianPruner, searches LR/batch/dice_weight/filters
- **CLI modes**: `--mode verify | tune | train`

### Generated Outputs
| File | Description |
|------|-------------|
| `backend/outputs/sea_unet_architecture.png` | Architecture block diagram |
| `backend/outputs/operational_risk_card.png` | Risk assessment visualisation |
| `backend/outputs/metrics_curves_sample.png` | Sample metrics dashboard |

### Dependencies Installed
`torch==2.10.0+cpu`, `albumentations`, `opencv-python`, `matplotlib`, `seaborn`,
`scikit-learn`, `tqdm`, `optuna`, `plotly`

---

## Step 3 Completion Log ✅

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `backend/api/rag_store.py` | ~290 | ChromaDB persistent vector store; TRI Sri Lanka agronomy guidelines (19 chunks); `query_guidelines()` semantic search |
| `backend/api/agent_graph.py` | ~430 | LangGraph 3-agent state machine; `run_agent_pipeline()` hybrid link; Ollama local LLM integration |
| `backend/api/__init__.py` | ~25 | Package exports |

### Task 3.1 — RAG Vector Store ✅
- **ChromaDB** persistent collection: `tea_agronomy_guidelines` at `backend/chroma_db/`
- **Embedding model**: `all-MiniLM-L6-v2` (ONNX, downloaded to `~/.cache/chroma/`)
- **19 agronomy chunks** ingested from TRI Sri Lanka, MASL, and published literature:
  - Coverage: all 8 tea disease classes + cross-cutting MRL/regulatory/environmental chunks
  - Per-chunk metadata: disease, risk_tier, source citation
- **`query_guidelines(disease, tier, n_results=4)`**: cosine similarity semantic search
- **Verification**: 19 documents ingested ✓ | Query `brown blight / CRITICAL` returned 3 passages (dist=0.310/0.336/0.422) ✓

### Task 3.2 — LangGraph Orchestration ✅
- **`AgentState`** TypedDict: full state schema annotated for LangGraph serialisation
- **3 agent nodes** in sequential graph:
  1. `evidence_retrieval_agent` — ChromaDB semantic search, returns top-4 sourced passages
  2. `risk_contraindication_agent` — Ollama LLM (phi3/llama3.2) contraindication analysis
  3. `validation_citation_agent` — final synthesis + citation + uncertainty flag
- **Ollama configuration**: `http://localhost:11434`, preferred=phi3, fallback=llama3.2
- **Graceful degradation**: if Ollama is offline → rule-based narrative from retrieved passages → `uncertainty_flag=LOW_CONFIDENCE`
- **Compiled graph**: `StateGraph(AgentState).compile()` → `CompiledStateGraph` ✓

### Task 3.3 — Hybrid Link (Innovation) ✅
- **`run_agent_pipeline(risk_report: dict)`** — entry point accepts the EXACT dict returned by `calculate_operational_risk()` from `backend/models/metrics.py`
- No adapter required — this is the architectural bridge between pixel-level segmentation and field-level treatment recommendation
- Example bridge call (Step 2 → Step 3):
  ```python
  from backend.models.metrics import calculate_operational_risk
  from backend.api.agent_graph import run_agent_pipeline
  risk = calculate_operational_risk(fp=1500, fn=3000, disease_class="brown blight")
  result = run_agent_pipeline(risk)  # → treatment plan + citations + agent_trace
  ```

### dataset.py Fixes Applied ✅
- **Shape mismatch crash**: added forced `cv2.resize()` of `img_rgb` to `(img_size, img_size)` in `__getitem__` BEFORE transforms — handles all Kaggle original sizes
- **`ShiftScaleRotate` → `A.Affine`** (removed invalid `mode` kwarg)
- **`GaussNoise(var_limit=...)` → `A.GaussNoise(p=0.3)`** (no deprecated var_limit)
- **`CoarseDropout` API**: updated to `num_holes_range`, `hole_height_range`, `hole_width_range`, `fill`

### Step 3 Verification Results
| Component | Result |
|-----------|--------|
| ChromaDB ingestion (19 chunks) | PASSED |
| Vector query (brown blight / CRITICAL) | PASSED (dist=0.310) |
| LangGraph graph compile | PASSED (CompiledStateGraph) |
| `run_agent_pipeline(risk_report)` hybrid call | PASSED |
| Ollama offline graceful degradation | PASSED (LOW_CONFIDENCE fallback) |
| Total agent_trace steps | 7 |

### New Dependencies Installed
`langgraph`, `langchain`, `langchain-community`, `chromadb`, `ollama`

---

## Step 4 Completion Log ✅

### Files Created / Modified

| File | Lines | Purpose |
|------|-------|---------|
| `backend/api/main.py` | ~290 | FastAPI SSE server — full pipeline endpoint |
| `frontend/vite.config.js` | ~20 | Tailwind v4 plugin + `/api` proxy |
| `frontend/index.html` | ~15 | Title + Google Fonts (Inter, JetBrains Mono) |
| `frontend/src/index.css` | ~280 | Full glassmorphism design system |
| `frontend/src/App.jsx` | ~145 | Main 3-panel layout + header + error banner |
| `frontend/src/App.css` | 1 | Cleared (all styles in index.css) |
| `frontend/src/hooks/useAnalysis.js` | ~100 | SSE streaming hook (fetch + ReadableStream) |
| `frontend/src/components/ImageUploadPanel.jsx` | ~110 | Drag-and-drop upload with preview |
| `frontend/src/components/MaskVisualizationPanel.jsx` | ~105 | Original vs overlay side-by-side display |
| `frontend/src/components/AgentAdvicePanel.jsx` | ~130 | Risk tier + narrative + citations + cost metrics |
| `frontend/src/components/ReasoningTrace.jsx` | ~115 | 7-step glowing agent progress stepper |

### Task 4.1 — FastAPI SSE Server ✅

#### `backend/api/main.py`
- **`get_model()`**: lazy singleton `SEAUNet`; loads `backend/checkpoints/sea_unet_best.pt` if present, else uses untrained model with correct shapes
- **`warmup()`**: `@app.on_event("startup")` — pre-warms ChromaDB + SEAUNet on server boot
- **`_preprocess_image(bytes)`**: cv2 decode → resize 256×256 → `build_val_transforms` → tensor `(1,3,256,256)`
- **`_run_inference(tensor)`**: argmax over logits → pred_map → disease_mask = pixels ≠ healthy (class 5); returns `(mask_uint8, pred_class_idx, severity_pct)`
- **`_mask_to_overlay_b64(img_bgr, mask)`**: red channel overlay (α=0.45) + green contours → base64 PNG data URI
- **`_sse_event(type, step, label, data)`**: formats `id: {step}\ndata: {json}\n\n`
- **`_analysis_stream(bytes)`**: async generator, 7 SSE events:
  1. `Extracting Features` — preprocess image
  2. `Running SEA-UNet Inference` — model forward pass
  3. `Computing Operational Risk` — `calculate_operational_risk()` with FP/FN estimates
  4. `Querying Evidence Base` — piggybacks Agent 1 pre-flight
  5. `Checking Contraindications` — Agent 2 pre-flight
  6. `Validating & Citing` — Agent 3 pre-flight
  7. `Analysis Complete` — full result payload
- **CORS**: `allow_origins=["*"]` for dev
- **Routes**: `GET /` health, `POST /api/analyze` SSE, `GET /api/classes`

### Task 4.2 — Glassmorphism React UI ✅

#### Design System (`src/index.css`)
- `@import "tailwindcss"` (Tailwind v4 — no `tailwind.config.js` needed)
- CSS custom properties: `--glass-bg`, `--glass-border`, `--glass-blur`, glow color vars
- `.glass`: `backdrop-filter: blur(16px)` frosted panel
- `.glow-green/blue/amber/red/critical`: glow border + box-shadow variants
- `.text-glow-*`: colored text-shadow
- `.btn-neon`: gradient background neon CTA button
- `.drop-zone`: dashed upload zone with drag-over glow
- `.trace-step + .active + .done`: agent progress dot system
- `.risk-badge + .risk-GREEN/AMBER/RED/CRITICAL`: tier pill badges (CRITICAL animates)
- `.scan-wrap`: scan-line loading animation
- Keyframes: `pulse-dot`, `pulse-critical`, `scan`, `fadeSlideUp`, `spin`, `shimmer`

#### `useAnalysis` Hook
- `status`: `"idle" | "streaming" | "done" | "error"`
- Uses `fetch` (not `EventSource`) — required because EventSource cannot POST
- `submitImage(file)`: POST multipart FormData → ReadableStream line parser → SSE event dispatch
- `reset()`: `AbortController.abort()` + clears all state
- Returns: `{ status, traceSteps, currentStep, result, error, submitImage, reset, agentSteps }`

#### Components
| Component | Key Props | Description |
|-----------|-----------|-------------|
| `ImageUploadPanel` | `onSubmit, isLoading, onReset, hasResult` | Drag-and-drop upload + preview + neon CTA |
| `MaskVisualizationPanel` | `originalB64, overlayB64, diseaseClass, severityPct, isLoading` | Side-by-side images, severity bar |
| `AgentAdvicePanel` | `result, status` | Risk tier badge, composite score, narrative, citations, LKR costs |
| `ReasoningTrace` | `agentSteps, traceSteps, status` | 7-step vertical stepper with pulsing/glow transitions |

### New Dependencies Installed
**Backend**: `fastapi`, `uvicorn`, `python-multipart`
**Frontend**: `tailwindcss@^4.0.0`, `@tailwindcss/vite` (bundled in Vite 8 scaffold)

---

## 🚀 HOW TO RUN THE FULL STACK

### Prerequisites
- `.venv` activated (Python 3.10.11)
- Ollama running: `ollama serve` (separate terminal) — phi3 model pulled
- Node 24.8.0+, npm 11.6.0+

### Terminal 1 — Backend (FastAPI)
```powershell
cd D:\AI_Internship_ManulaFernando_LOLC_Tech_2025\05_Resources\NIBM_Bsc\vision-unet-segmentation-for-tea-leaves
.venv\Scripts\activate
python -m uvicorn backend.api.main:app --reload --port 8000
# → server ready at http://localhost:8000
# → API docs at  http://localhost:8000/docs
```

### Terminal 2 — Frontend (Vite dev server)
```powershell
cd frontend
npm run dev
# → http://localhost:5173
```

### Terminal 3 — Ollama (if not already running)
```powershell
ollama serve
# → ollama listening on http://localhost:11434
```

### Usage
1. Open **http://localhost:5173** in browser
2. Drag-and-drop or click to upload a tea leaf JPEG/PNG
3. Click **Analyse Leaf**
4. Watch the Reasoning Trace stepper glow through 7 steps in real-time
5. View segmentation overlay, disease class, severity %, and agronomic advisory
6. Risk tier badge shows GREEN / AMBER / RED / CRITICAL with LKR cost estimates
7. Click **Reset** to clear and analyse a new image

---

## Step 5 Completion Log ✅ — 100% COMPLETE

### 🏆 Official Ablation Study Results (100 Epochs, Google Colab T4 GPU)

| Phase | Model | Dice | IoU | Sensitivity | Specificity | Loss |
|-------|-------|------|-----|-------------|-------------|------|
| 1 | **SEA-UNet** (Baseline) | 0.7536 | 0.6097 | 0.8493 | 0.9770 | 0.3279 |
| 2 | **ResNet34-UNet** (Transfer Learning) | 0.7741 | 0.6352 | 0.8726 | 0.9791 | 0.2942 |
| 3 | **EfficientNet-B3 + SCSE** (SOTA) | **0.7845** | **0.6511** | **0.9129** | 0.9784 | **0.2577** |

> **Result:** EfficientNet-B3 + SCSE achieves **+4.1% Dice** and **+7.5% Sensitivity** over baseline, with **21.4% lower loss**.

### Files Created

| File | Cells | Purpose |
|------|-------|---------|
| `notebooks/colab_tea_segmentation.ipynb` | 16 | Complete 3-phase ablation study notebook |

### Trained Model Checkpoints

| Checkpoint | Size | Best Epoch | Dice |
|------------|------|------------|------|
| `sea_unet_binary_best.pth` | ~132 MB | 87 | 0.7536 |
| `resnet34_unet_best.pth` | ~96 MB | 92 | 0.7741 |
| `efficientnet_scse_best.pth` | ~48 MB | 95 | 0.7845 |

### Notebook Structure (3-Phase Ablation)

| # | Section | Content |
|---|---------|---------|
| 1 | **Setup & Dependencies** | `pip install segmentation-models-pytorch albumentations kaggle`, GPU assertion, `DEVICE = cuda` |
| 2 | **Kaggle API & Data Ingestion** | `KAGGLE_USERNAME` / `KAGGLE_KEY` → `kaggle datasets download` → auto-unzip |
| 3 | **Dataset Module** | `TeaLeafSegmentationDataset` + GrabCut pseudo-mask + albumentations + `WeightedRandomSampler` |
| 4 | **Phase 1: SEA-UNet** | Custom `SEBlock`, `AttentionGate`, `SEAUNet` architecture (~33M params) |
| 5 | **Phase 2: ResNet34-UNet** | `smp.Unet(encoder_name='resnet34', encoder_weights='imagenet')` |
| 6 | **Phase 3: EfficientNet-B3+SCSE** | `smp.Unet(encoder_name='efficientnetb3', decoder_attention_type='scse')` |
| 7 | **Training Loop** | 100 epochs · AdamW (lr=1e-4) · CosineAnnealingLR · Focal Tversky Loss (α=0.7, β=0.3) |
| 8 | **Ablation Comparison** | Side-by-side metrics table, training curves, statistical significance tests |
| 9 | **Model Export** | Download all 3 checkpoints + comparative visualization PNGs |

### Key Technical Decisions

- **3-Phase Ablation Design**: Baseline → Transfer Learning → Attention Enhancement
- **Focal Tversky Loss (α=0.7, β=0.3)**: Penalizes false negatives 2.33× more than false positives
- **segmentation_models_pytorch**: Production-quality encoders with ImageNet pretraining
- **SCSE Attention**: Channel + Spatial recalibration at decoder stages
- **100 Epochs**: Sufficient for convergence with early stopping patience of 15

### Post-Training Deployment

1. Download all 3 `.pth` checkpoints from Colab
2. Copy to `backend/checkpoints/`
3. Backend loads selected model via query param: `?model=efficientnet_scse`
4. Frontend model selector dropdown allows user comparison

---

## ➡️ Step 6 — Final Reporting & Deployment Polish ✅
**Status:** COMPLETE (Smoke Test Passed)

### Tasks Completed

| # | Task | Status |
|---|------|--------|
| 1 | Update `current-status.md` with ablation results | ✅ Done |
| 2 | Update `Agent.md` with SOTA model references | ✅ Done |
| 3 | Backend multi-model support (`api/main.py`) | ✅ Done |
| 4 | Frontend model selector dropdown | ✅ Done |
| 5 | Copy checkpoints to `backend/checkpoints/` | ✅ Done |
| 6 | Final integration testing (Smoke Test) | ✅ PASSED |
| 7 | Fix binary vs multi-class segmentation mismatch | ✅ Done |

### Smoke Test Results (2025-02-20)

| Component | Status | Details |
|-----------|--------|---------|
| Backend (FastAPI) | ✅ Running | http://127.0.0.1:8000 |
| Frontend (Vite/React) | ✅ Running | http://localhost:5173 |
| Ollama LLM | ✅ Running | Port 11434 |
| Model Loading | ✅ No Errors | EfficientNet-B3+SCSE loaded without mismatch warning |
| ChromaDB | ✅ Ready | Vector store initialized |

### Bug Fixed: Binary Segmentation Mismatch

**Issue:** Trained checkpoints used binary segmentation (1 class) but model was created with 8 classes.

**Fix Applied:**
1. Added `"num_classes": 1` to all MODEL_REGISTRY entries in [unet.py](backend/models/unet.py)
2. Updated `create_model()` to use registry num_classes by default
3. Updated `_run_inference()` in [main.py](backend/api/main.py) to handle binary output (sigmoid + threshold at 0.5)
4. Removed explicit `num_classes=NUM_CLASSES` from model creation call

---

## ➡️ Step 7 — Technical Report & Documentation
**Status:** IN PROGRESS (Current Phase)

### Completed in This Phase

| # | Task | Status |
|---|------|--------|
| 1 | 2-Stage pipeline (U-Net + ResNet-50) | ✅ Done |
| 2 | Agent truncation fix (num_predict 300→500) | ✅ Done |
| 3 | Structured narrative formatting | ✅ Done |
| 4 | Enhanced glassmorphism UI | ✅ Done |
| 5 | Plantation theme integration | ✅ Done |
| 6 | current-status.md update | ✅ Done |

### Files Modified

| File | Changes |
|------|---------|
| `backend/api/main.py` | Added `get_classifier()`, `_classify_disease()`, 2-stage `_run_inference()` |
| `backend/models/unet.py` | Added `CLASSIFIER_CONFIG`, `create_classifier()` |
| `backend/api/agent_graph.py` | Increased `num_predict=500`, updated prompt for structured sections |
| `frontend/src/components/AgentAdvicePanel.jsx` | Added `FormattedNarrative` component with section parsing |
| `frontend/src/index.css` | Enhanced glassmorphism, plantation theme, narrative styles |

### Remaining Tasks
- [ ] 6,000–8,000 word PDF technical report
- [ ] Sections: Introduction, Literature Review, Methodology, Results, Discussion, Conclusion
- [ ] Include ablation study tables and comparative figures

---

## Step 1 Completion Log

### ✅ Directory Structure
All project directories created and verified:
- `/research_papers/` — `LITERATURE_REVIEW.md` + 3 organized PDF subdirectories
- `/datasets/tea_sickness/` — Kaggle dataset downloaded and extracted (885 images, 8 classes)
- `/backend/models/` — empty, ready for Step 2
- `/backend/api/` — empty, ready for Steps 3–4
- `/frontend/` — empty, ready for Step 5

### ✅ Dataset
- **Source:** Kaggle — `shashwatwork/identifying-disease-in-tea-leafs`
- **Total Images:** 885
- **Classes (8):** Anthracnose, Algal leaf, Bird eye spot, Brown blight, Gray light, Healthy, Red leaf spot, White spot
- **Note:** Classification labels only. Segmentation masks to be generated in Step 2.

### ✅ Research Papers — All 11 PDFs Downloaded (0 failures)

#### Section 1: Foundational Architectures (5/5)
| File | Size |
|------|------|
| `UNet_Ronneberger_2015.pdf` | 1610 KB |
| `ResNet_He_2015.pdf` | 800 KB |
| `Attention_UNet_Oktay_2018.pdf` | 3568 KB |
| `SENet_Hu_2018.pdf` | 2130 KB |
| `CBAM_Woo_2018.pdf` | 1897 KB |

#### Section 2: Tea Leaf Disease Detection (3/3)
| File | Size |
|------|------|
| `ML_Tea_Disease_Review_2023.pdf` | 296 KB |
| `CNN_Tea_Disease_Recognition_2019.pdf` | 922 KB |
| `DAONet_YOLOv8_Tea_Detection_2025.pdf` | 1885 KB |

#### Section 3: Sri Lankan Context (3/3) — Direct open-access from KDU & IIT repositories
| File | Source | Size |
|------|--------|------|
| `KDU_Blister_Blight_Deep_Learning_2024.pdf` | KDU Institutional Repository (ir.kdu.ac.lk) | 179 KB |
| `IIT_LeafCheck_Deep_Learning_Sri_Lanka.pdf` | IIT Digital Library (dlib.iit.ac.lk) | 130 KB |
| `SPIS_TS_Tea_Smallholdings_Sri_Lanka.pdf` | IRJIET (irjiet.com) | 579 KB |

### ✅ Literature Review
`LITERATURE_REVIEW.md` fully updated to reflect all 11 downloaded papers covering:
- 5 foundational architecture papers (U-Net, ResNet, Attention U-Net, SE-Net, CBAM)
- 3 tea leaf disease detection papers (ML review 2023, CNN 2019, DAONet YOLOv8 2025)
- 3 Sri Lankan context papers (KDU Blister Blight 2024, IIT LeafCheck, IRJIET Smallholdings)
- 7 identified research gaps
- Novel architecture concept: **SE-Attention U-Net (SEA-UNet)**
- 11 BibTeX-ready references
- PDF audit table confirming all downloads

---

## ➡️ Next Step: Step 2 — Vision Backend Scaffold
**Status:** READY TO BEGIN

Tasks:
- Write `backend/models/unet.py` — Attention U-Net + SE blocks (SEA-UNet)
- Implement combined Dice + BCE loss function
- Create data loader and preprocessing pipeline (`backend/models/dataloader.py`)
- Generate segmentation masks from classification images (GrabCut / thresholding strategy)
- Unit tests for model forward pass and loss computation
