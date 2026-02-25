"""
main.py — FastAPI Application & SSE Streaming Endpoint
=======================================================
BSc Data Science Capstone — NIBM, 2026
Author  : Manula Fernando

Task 4.1 — I build the REST API layer that connects the SEA-UNet vision model
and the LangGraph agentic pipeline to the glassmorphism React frontend.

Architecture
------------
  POST /api/analyze  (multipart/form-data: image file)
      │
      ├─ Step 1: Load & preprocess the uploaded leaf image
      ├─ Step 2: Run SEA-UNet inference → segmentation mask
      ├─ Step 3: Compute class probabilities → disease_class + severity_pct
      ├─ Step 4: calculate_operational_risk(fp, fn, disease_class)
      ├─ Step 5: run_agent_pipeline(risk_report)     [LangGraph — 3 agents]
      └─ SSE stream: yield agent_trace events in real time

Server-Sent Events (SSE) design
--------------------------------
  I use FastAPI's ``StreamingResponse`` with ``text/event-stream`` media type.
  Each SSE event carries a JSON payload with:
      { "type": "trace" | "result" | "error",
        "step": 1–7,
        "label": "...",
        "data": {...} }

  The frontend subscribes with a native ``EventSource`` (no WebSocket required)
  so latency is low and the connection is unidirectional (server→client).

  I chose SSE over WebSockets here because the analysis is unidirectional
  (server pushes progress to client) and SSE is simpler to implement, auto-
  reconnects, and works through standard HTTP proxies.

References
----------
  [1] FastAPI streaming responses: https://fastapi.tiangolo.com/advanced/custom-response/
  [2] MDN SSE spec: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
  [3] Lewis et al. "RAG for Knowledge-Intensive NLP." NeurIPS 2020.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── vision model imports ──────────────────────────────────────────────────────
from backend.models.unet    import (
    SEAUNet, NUM_CLASSES, 
    create_model, get_available_models, MODEL_REGISTRY, DEFAULT_MODEL,
    create_classifier, get_classifier_config, CLASSIFIER_CONFIG
)
from backend.models.dataset import CLASS_NAMES, build_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from backend.models.metrics import calculate_operational_risk

# ── agentic layer imports ─────────────────────────────────────────────────────
from backend.api.agent_graph import run_agent_pipeline, run_agent_pipeline_fast, FAST_MODE_ENABLED
from backend.api.rag_store   import get_or_create_store  # warm up on startup

# ──────────────────────────────────────────────────────────────────────────────
# App configuration
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINTS_DIR = PROJECT_ROOT / "backend" / "checkpoints"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE        = 256

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Tea Leaf Disease Analysis API",
    description=(
        "SEA-UNet segmentation model + LangGraph agentic treatment recommender "
        "for tea leaf disease detection (BSc Capstone — NIBM 2026)"
    ),
    version="1.0.0",
)

# Allow all origins during development.  In production, restrict to the
# frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Model cache — supports multi-model architecture (ablation study)
# ──────────────────────────────────────────────────────────────────────────────

import torch.nn as nn
import torch.nn.functional as F

_model_cache: Dict[str, nn.Module] = {}
_classifier_cache: Optional[nn.Module] = None
_current_model_name: str = DEFAULT_MODEL


def get_model(model_name: Optional[str] = None) -> nn.Module:
    """
    Load a segmentation model from the ablation study.

    Parameters
    ----------
    model_name : str, optional
        One of 'sea_unet', 'resnet34_unet', 'efficientnet_scse'.
        If None, uses the DEFAULT_MODEL (efficientnet_scse).

    Returns
    -------
    nn.Module
        Model in eval mode on the appropriate device (GPU/CPU).
    
    Notes
    -----
    Models are cached in memory. The first request for a model loads it;
    subsequent requests return the cached instance.
    """
    global _model_cache, _current_model_name
    
    if model_name is None:
        model_name = DEFAULT_MODEL
    model_name = model_name.lower().strip()
    
    # Validate model name
    if model_name not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Valid options: {valid}")
    
    # Return cached model if available
    if model_name in _model_cache:
        _current_model_name = model_name
        return _model_cache[model_name]
    
    # Create and load model
    print(f"[API] Loading model: {model_name}...")
    # Use registry num_classes (binary segmentation = 1 class)
    model = create_model(model_name)
    
    # Load checkpoint
    checkpoint_name = MODEL_REGISTRY[model_name]["checkpoint"]
    checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
    
    # Also check for .pt extension (in case file was renamed)
    if not checkpoint_path.exists():
        checkpoint_path_pt = CHECKPOINTS_DIR / checkpoint_name.replace(".pth", ".pt")
        if checkpoint_path_pt.exists():
            checkpoint_path = checkpoint_path_pt
    
    if checkpoint_path.exists():
        state = torch.load(str(checkpoint_path), map_location=DEVICE)
        # Handle both raw state_dict and wrapped checkpoint dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[API] Loaded checkpoint: {checkpoint_path}")
    else:
        print(
            f"[API] WARNING: No checkpoint at {checkpoint_path}. "
            "Using un-trained model (shapes will be correct)."
        )
    
    model.to(DEVICE).eval()
    _model_cache[model_name] = model
    _current_model_name = model_name
    
    return model


def get_current_model_info() -> Dict[str, Any]:
    """Return metadata about the currently selected model."""
    info = MODEL_REGISTRY.get(_current_model_name, {}).copy()
    info["name"] = _current_model_name
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Disease Classifier (2-Stage Pipeline: Segmentation → Classification)
# ──────────────────────────────────────────────────────────────────────────────

# Classifier checkpoint directory (in backend/checkpoints/)
CLASSIFIER_CHECKPOINT = CHECKPOINTS_DIR / CLASSIFIER_CONFIG["checkpoint"]


def get_classifier() -> nn.Module:
    """
    Load the EfficientNet-B4 disease classifier for 2-stage inference.
    
    The classifier is used after segmentation to predict the actual
    disease class (instead of relying on folder names or hardcoded defaults).
    
    Returns
    -------
    nn.Module
        EfficientNet-B4 classifier in eval mode on the appropriate device.
    """
    global _classifier_cache
    
    if _classifier_cache is not None:
        return _classifier_cache
    
    print("[API] Loading disease classifier (EfficientNet-B4 512x512)...")
    
    if CLASSIFIER_CHECKPOINT.exists():
        # The checkpoint contains the full model (not just state_dict)
        # This is common when using torch.save(model, path) instead of
        # torch.save(model.state_dict(), path)
        classifier = torch.load(
            str(CLASSIFIER_CHECKPOINT), 
            map_location=DEVICE,
            weights_only=False  # Required for full model checkpoints
        )
        print(f"[API] Loaded classifier checkpoint: {CLASSIFIER_CHECKPOINT}")
    else:
        # Create a new classifier with random weights (fallback)
        print(
            f"[API] WARNING: No classifier checkpoint at {CLASSIFIER_CHECKPOINT}. "
            "Classification will use random weights!"
        )
        classifier = create_classifier(num_classes=CLASSIFIER_CONFIG["num_classes"])
    
    classifier.to(DEVICE).eval()
    _classifier_cache = classifier
    
    return classifier


@torch.no_grad()
def _classify_disease(tensor: torch.Tensor) -> int:
    """
    Run the disease classifier on a preprocessed image tensor.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Preprocessed input tensor (1, 3, H, W), already normalized.
    
    Returns
    -------
    int
        Predicted class index (0-7 for the 8 tea disease classes).
    """
    classifier = get_classifier()
    classifier_input_size = CLASSIFIER_CONFIG["input_size"]  # 512 for EfficientNet-B4
    
    # Resize to classifier's expected input size (512x512 for EfficientNet-B4)
    if tensor.shape[-1] != classifier_input_size:
        tensor_resized = F.interpolate(
            tensor, 
            size=(classifier_input_size, classifier_input_size), 
            mode="bilinear", 
            align_corners=False
        )
    else:
        tensor_resized = tensor
    
    logits = classifier(tensor_resized)  # (1, 8)
    pred_class_idx = int(logits.argmax(dim=1).item())
    
    return pred_class_idx


# ──────────────────────────────────────────────────────────────────────────────
# Startup: warm up ChromaDB so first request doesn't pay the load penalty
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def warmup() -> None:
    """Pre-load the ChromaDB collection, models, and classifier on server start."""
    print("[API] Warming up ChromaDB vector store...")
    try:
        get_or_create_store()
        print("[API] ChromaDB ready.")
    except Exception as exc:
        print(f"[API] WARNING: ChromaDB warmup failed: {exc}")

    print(f"[API] Pre-loading default segmentation model ({DEFAULT_MODEL})...")
    try:
        get_model(DEFAULT_MODEL)
        print(f"[API] {DEFAULT_MODEL} loaded on {DEVICE}.")
    except Exception as exc:
        print(f"[API] WARNING: Segmentation model pre-load failed: {exc}")

    print("[API] Pre-loading disease classifier (ResNet-50)...")
    try:
        get_classifier()
        print(f"[API] ResNet-50 classifier loaded on {DEVICE}.")
    except Exception as exc:
        print(f"[API] WARNING: Classifier pre-load failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_image(file_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
    """
    Decode uploaded bytes → numpy BGR array + preprocessed model input tensor.

    I forcefully resize to IMG_SIZE × IMG_SIZE here (matching the NTFS-safe
    fix we applied in dataset.py) so the model always receives a consistent
    256 × 256 input regardless of the uploaded image size.

    Returns
    -------
    (img_bgr, tensor)
        img_bgr : (256, 256, 3) uint8 for OpenCV operations
        tensor  : normalised FloatTensor (1, 3, 256, 256) for model input
    """
    nparr   = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode uploaded image.")

    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    tf        = build_val_transforms(IMG_SIZE)
    augmented = tf(image=img_rgb, mask=np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))
    tensor    = augmented["image"].float().unsqueeze(0).to(DEVICE)   # (1, 3, H, W)
    return img_bgr, tensor


@torch.no_grad()
def _run_inference(tensor: torch.Tensor, model_name: str = DEFAULT_MODEL) -> tuple[np.ndarray, int, float]:
    """
    Run 2-stage inference: Segmentation → Classification.

    Stage 1: Segmentation model detects diseased regions (binary mask)
    Stage 2: Classification model predicts the disease class

    Parameters
    ----------
    tensor : torch.Tensor
        Preprocessed input tensor (1, 3, H, W)
    model_name : str
        Name of the segmentation model to use for inference

    Returns
    -------
    (mask_uint8, pred_class_idx, severity_pct)
        mask_uint8      : (256, 256) uint8 — binary disease mask [0,255]
        pred_class_idx  : int — predicted class index from ResNet-50 classifier
        severity_pct    : float — percentage of pixels classified as diseased
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 1: SEGMENTATION — Detect diseased regions
    # ═══════════════════════════════════════════════════════════════════════════
    model  = get_model(model_name)
    logits = model(tensor)           # (1, num_classes, H, W)
    
    # Get number of output classes from model registry
    model_info = MODEL_REGISTRY.get(model_name, {})
    num_classes = model_info.get("num_classes", 1)
    
    if num_classes == 1:
        # Binary segmentation: apply sigmoid and threshold
        # Using lower threshold (0.3) to capture more affected areas including early-stage disease
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
        disease_mask = (probs > 0.3).astype(np.uint8)
    else:
        # Multi-class segmentation: take argmax across class dimension
        pred_map = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W) int
        healthy_idx = CLASS_NAMES.index("healthy")
        disease_mask = (pred_map != healthy_idx).astype(np.uint8)
    
    # ─── Calculate severity as % of LEAF area (not entire image) ───
    # Strategy: Use the CONVEX HULL of the disease mask to estimate total leaf area
    # Since disease spots are distributed across the leaf, the convex hull of all
    # diseased regions gives us a good approximation of the leaf boundary
    
    diseased_pixels = int(disease_mask.sum())
    total_pixels = disease_mask.shape[0] * disease_mask.shape[1]
    
    # Find contours of the disease mask
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Combine all contour points to find the overall convex hull
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        
        # Create a mask of the convex hull (this approximates the leaf area)
        hull_mask = np.zeros_like(disease_mask)
        cv2.fillPoly(hull_mask, [hull], 1)
        
        # Leaf area = pixels inside the convex hull
        leaf_pixels = int(hull_mask.sum())
        
        # Ensure leaf_pixels >= diseased_pixels (logical consistency)
        if leaf_pixels < diseased_pixels:
            leaf_pixels = diseased_pixels
    else:
        # No disease detected - fallback to total image
        leaf_pixels = total_pixels
    
    # Severity = diseased pixels / estimated leaf area
    if leaf_pixels > 100:
        severity_pct = float(diseased_pixels / leaf_pixels * 100)
        severity_pct = min(100.0, severity_pct)
    else:
        severity_pct = float(diseased_pixels / total_pixels * 100)
    
    # Debug print
    print(f"[SEVERITY] diseased={diseased_pixels}, leaf_hull={leaf_pixels}, total={total_pixels}, severity={severity_pct:.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 2: CLASSIFICATION — Identify disease type
    # ═══════════════════════════════════════════════════════════════════════════
    healthy_idx = CLASS_NAMES.index("healthy")
    gray_light_idx = CLASS_NAMES.index("gray light")
    
    # ALWAYS run the classifier for better accuracy
    # (Previously used 5% threshold which missed subtle diseases like gray light)
    pred_class_idx = _classify_disease(tensor)
    
    # Get classifier confidence for decision logic
    classifier = get_classifier()
    classifier_input_size = CLASSIFIER_CONFIG["input_size"]  # 512
    tensor_cls = F.interpolate(tensor, size=(classifier_input_size, classifier_input_size), mode="bilinear", align_corners=False)
    logits_cls = classifier(tensor_cls)
    probs = torch.softmax(logits_cls, dim=1)[0]
    confidence = float(probs[pred_class_idx].item())
    
    # Special handling: If severity is very low (<2%) AND classifier is confident
    # about healthy (>0.7), then trust the healthy classification
    if severity_pct < 2.0 and pred_class_idx == healthy_idx and confidence > 0.7:
        pass  # Keep healthy classification
    elif severity_pct < 5.0 and pred_class_idx == healthy_idx:
        # Low severity but classifier says healthy - check second-best
        # This catches subtle diseases like gray light
        probs_copy = probs.clone()
        probs_copy[healthy_idx] = 0.0
        second_best_idx = int(probs_copy.argmax().item())
        second_best_conf = float(probs_copy[second_best_idx].item())
        
        # If second-best (a disease) has reasonable confidence, use it
        if second_best_conf > 0.15:
            pred_class_idx = second_best_idx
    
    # Double-check: if classifier says healthy but mask shows significant disease,
    # trust the segmentation (edge case handling)
    if pred_class_idx == healthy_idx and severity_pct >= 20.0:
        # Force a disease class based on classifier's second-best prediction
        probs_copy = probs.clone()
        probs_copy[healthy_idx] = 0.0
        pred_class_idx = int(probs_copy.argmax().item())

    mask_uint8 = (disease_mask * 255).astype(np.uint8)
    return mask_uint8, pred_class_idx, severity_pct


def _mask_to_overlay_b64(img_bgr: np.ndarray, mask_uint8: np.ndarray) -> str:
    """
    Blend the binary disease mask over the original image as a red overlay.

    I return a base64-encoded PNG so the frontend can embed it directly in an
    ``<img>`` src tag without needing a separate file-serving route.

    Returns
    -------
    str
        Data URI: ``data:image/png;base64,...``
    """
    overlay    = img_bgr.copy()
    red_layer  = np.zeros_like(img_bgr)
    red_layer[:, :, 2] = mask_uint8          # OpenCV is BGR — channel 2 = red
    alpha      = 0.45
    overlay    = cv2.addWeighted(img_bgr, 1 - alpha, red_layer, alpha, 0)
    # Draw mask contours for better delineation
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 100), 1)

    _, buf = cv2.imencode(".png", overlay)
    b64    = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _original_b64(img_bgr: np.ndarray) -> str:
    """Encode the preprocessed original image as base64 PNG data URI."""
    _, buf = cv2.imencode(".png", img_bgr)
    b64    = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ──────────────────────────────────────────────────────────────────────────────
# SSE helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sse_event(event_type: str, step: int, label: str, data: Any) -> str:
    """
    Format a single SSE event string.

    The SSE spec requires lines of the form ``data: <text>\\n\\n``.
    I include an ``id`` field equal to the step number so the browser EventSource
    can resume from the correct position on reconnection.

    Parameters
    ----------
    event_type : str  — "trace" | "result" | "error"
    step       : int  — sequential step counter (1–N)
    label      : str  — human-readable step label for the UI progress bar
    data       : Any  — JSON-serialisable payload

    Returns
    -------
    str
        Fully formatted SSE event string.
    """
    payload = json.dumps({"type": event_type, "step": step, "label": label, "data": data}, default=str)
    return f"id: {step}\ndata: {payload}\n\n"


def _clean_narrative(text: str) -> str:
    """
    Clean markdown artifacts from LLM-generated narrative for display.
    
    Removes or converts:
    - Bold markers (**text**) → plain text
    - Section dividers (====, ----)
    - Extra whitespace and blank lines
    """
    import re
    
    # Remove bold markers **text** → text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    
    # Remove section dividers
    text = re.sub(r'[=\-]{3,}', '', text)
    
    # Remove headings starting with #
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Collapse multiple newlines to double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace
    return text.strip()


async def _analysis_stream(file_bytes: bytes, model_name: str = DEFAULT_MODEL) -> AsyncGenerator[str, None]:
    """
    Core async generator that drives the full analysis pipeline and yields
    SSE events at each significant processing step.

    Parameters
    ----------
    file_bytes : bytes
        Raw uploaded image file bytes
    model_name : str
        Model to use for segmentation inference

    The 7 steps emitted to the frontend are:
      1. trace  — Extracting Features (preprocessing)
      2. trace  — Running Model Inference
      3. trace  — Computing Operational Risk
      4. trace  — Querying Evidence Base (Agent 1)
      5. trace  — Checking Contraindications (Agent 2)
      6. trace  — Validating & Citing (Agent 3)
      7. result — Full analysis result

    I yield a small ``asyncio.sleep(0)`` between blocking operations to let
    the event loop breathe and ensure the SSE chunks are actually flushed to
    the HTTP layer before the next blocking call begins.
    """
    model_info = MODEL_REGISTRY.get(model_name, {})
    model_desc = model_info.get("description", model_name)
    
    step = 0
    try:
        # ── Step 1: Preprocess ───────────────────────────────────────────────
        step += 1
        yield _sse_event("trace", step, "Extracting Features", {
            "message": "Preprocessing image (resize, normalise, tensor conversion)..."
        })
        await asyncio.sleep(0)

        img_bgr, tensor = await asyncio.get_event_loop().run_in_executor(
            None, _preprocess_image, file_bytes
        )
        original_b64 = await asyncio.get_event_loop().run_in_executor(
            None, _original_b64, img_bgr
        )

        # ── Step 2: Model inference ───────────────────────────────────────
        step += 1
        yield _sse_event("trace", step, f"Running {model_name} Inference", {
            "message": f"Forwarding through {model_desc}...",
            "model": model_name,
        })
        await asyncio.sleep(0)

        mask_uint8, pred_class_idx, severity_pct = await asyncio.get_event_loop().run_in_executor(
            None, _run_inference, tensor, model_name
        )
        disease_class   = CLASS_NAMES[pred_class_idx]
        overlay_b64     = await asyncio.get_event_loop().run_in_executor(
            None, _mask_to_overlay_b64, img_bgr, mask_uint8
        )

        # ── Emit early segmentation result so UI shows it immediately ────────
        # This allows users to see the segmentation while agent processes
        yield _sse_event("segmentation", step, "Segmentation Complete", {
            "message": f"Detected: {disease_class} ({severity_pct:.1f}% pixels affected)",
            "disease_class": disease_class,
            "classification_method": "EfficientNet-B4" if severity_pct >= 5.0 else "threshold",
            "severity_pct":  round(severity_pct, 2),
            "original_image_b64": original_b64,
            "overlay_image_b64":  overlay_b64,
            "model_name": model_name,
            "model_description": model_desc,
        })
        await asyncio.sleep(0)

        # ── Step 3: Operational risk ─────────────────────────────────────────
        step += 1
        yield _sse_event("trace", step, "Computing Operational Risk", {
            "message": "Calculating FP/FN operational costs (LKR)..."
        })
        await asyncio.sleep(0)

        # Convert severity_pct to approximate FP/FN pixel counts for the
        # risk calculator.  Total image pixels = 256 × 256 = 65 536.
        total_px    = IMG_SIZE * IMG_SIZE
        diseased_px = int(severity_pct / 100 * total_px)
        # FN estimate: diseased pixels that the model might have missed
        # (we use a conservative 20% of detected pixels as proxy)
        fn_px = int(diseased_px * 0.20)
        fp_px = int((total_px - diseased_px) * 0.05)

        risk_report = await asyncio.get_event_loop().run_in_executor(
            None, calculate_operational_risk,
            fp_px, fn_px, disease_class
        )
        
        # Override risk_tier based on severity_pct directly for consistency
        # This ensures the gauge (showing severity_pct) matches the tier color
        if severity_pct < 10:
            severity_based_tier = "GREEN"
        elif severity_pct < 30:
            severity_based_tier = "AMBER"
        elif severity_pct < 60:
            severity_based_tier = "RED"
        else:
            severity_based_tier = "CRITICAL"
        
        # Use severity-based tier for display consistency
        risk_tier = severity_based_tier
        
        # Calculate total cost from fp + fn estimates
        total_cost_lkr = risk_report["fp_estimated_cost_lkr"] + risk_report["fn_estimated_cost_lkr"]

        yield _sse_event("trace", step, "Computing Operational Risk", {
            "message": (
                f"Risk tier: {risk_tier}  |  "
                f"Severity: {severity_pct:.1f}%  |  "
                f"Total cost: LKR {total_cost_lkr:.2f}"
            ),
            "risk_tier":             risk_tier,
            "composite_risk_score":  risk_report["composite_risk_score"],
            "total_cost_lkr":        total_cost_lkr,
        })
        await asyncio.sleep(0)

        # ── Steps 4–6: LangGraph agents with real-time updates ────────────────
        # Use a thread-safe queue to receive real-time agent updates
        import queue
        agent_queue = queue.Queue()
        
        def agent_callback(agent_num: int, data: dict):
            """Thread-safe callback to emit agent step updates."""
            agent_queue.put((agent_num, data))
        
        # Run pipeline in background thread with callback
        loop = asyncio.get_event_loop()
        pipeline_task = loop.run_in_executor(
            None, 
            lambda: run_agent_pipeline_fast(risk_report, callback=agent_callback) if FAST_MODE_ENABLED else run_agent_pipeline(risk_report)
        )
        
        # Poll queue for real-time agent events while pipeline runs
        agent_step_map = {1: 4, 2: 5, 3: 6}  # Map agent number to SSE step number
        agent_label_map = {
            1: "Querying Evidence Base",
            2: "Checking Contraindications", 
            3: "Validating & Citing"
        }
        
        while not pipeline_task.done():
            try:
                # Check for agent updates (non-blocking with short timeout)
                agent_num, agent_data = agent_queue.get(timeout=0.1)
                sse_step = agent_step_map.get(agent_num, 4)
                sse_label = agent_label_map.get(agent_num, "Processing")
                
                # Emit agent event with real data
                yield _sse_event("agent_step", sse_step, sse_label, {
                    "agent_num": agent_num,
                    **agent_data
                })
                await asyncio.sleep(0)
            except queue.Empty:
                # No updates, yield CPU
                await asyncio.sleep(0.05)
        
        # Drain any remaining events in queue
        while not agent_queue.empty():
            try:
                agent_num, agent_data = agent_queue.get_nowait()
                sse_step = agent_step_map.get(agent_num, 4)
                sse_label = agent_label_map.get(agent_num, "Processing")
                yield _sse_event("agent_step", sse_step, sse_label, {
                    "agent_num": agent_num,
                    **agent_data
                })
            except queue.Empty:
                break
        
        # Get final result
        treatment = await pipeline_task
        
        # Clean narrative of markdown artifacts for cleaner display
        raw_narrative = treatment.get("narrative_summary", "")
        clean_narrative = _clean_narrative(raw_narrative)

        # ── Step 7: Final result event ────────────────────────────────────────
        yield _sse_event("result", 7, "Analysis Complete", {
            "disease_class":         disease_class,
            "classification_method": "EfficientNet-B4" if severity_pct >= 5.0 else "threshold",
            "severity_pct":          round(severity_pct, 2),
            "risk_tier":             risk_tier,  # Use severity-based tier
            "composite_risk_score":  risk_report["composite_risk_score"],
            "fp_cost_lkr":           risk_report["fp_estimated_cost_lkr"],
            "fn_cost_lkr":           risk_report["fn_estimated_cost_lkr"],
            "total_cost_lkr":        total_cost_lkr,
            "uncertainty_flag":      treatment.get("uncertainty_flag", "UNKNOWN"),
            "citations":             treatment.get("citations", []),
            "narrative_summary":     clean_narrative,
            "agent_trace":           treatment.get("agent_trace", []),
            "original_image_b64":    original_b64,
            "overlay_image_b64":     overlay_b64,
            "model_name":            model_name,
            "model_description":     model_desc,
            "model_metrics":         {
                "dice": model_info.get("dice"),
                "sensitivity": model_info.get("sensitivity"),
            },
        })

    except Exception as exc:
        tb = traceback.format_exc()
        yield _sse_event("error", step, "Analysis Failed", {
            "message": str(exc),
            "traceback": tb,
        })


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Basic health-check endpoint."""
    model_info = get_current_model_info()
    return {
        "status": "ok",
        "model": model_info.get("name", DEFAULT_MODEL),
        "description": model_info.get("description", ""),
        "device": str(DEVICE),
        "version": "2.0.0",  # Updated for multi-model support
    }


@app.get("/api/models", tags=["Metadata"])
async def list_models() -> Dict[str, Any]:
    """
    List all available segmentation models from the ablation study.
    
    Returns model metadata including checkpoints, descriptions, and metrics.
    Use the model name as the 'model' query parameter in /api/analyze.
    """
    return {
        "default": DEFAULT_MODEL,
        "current": _current_model_name,
        "models": get_available_models(),
        "usage": "Add ?model=<name> to /api/analyze to select a model",
    }


@app.post("/api/analyze", tags=["Analysis"])
async def analyze_leaf(
    image: UploadFile = File(..., description="JPEG/PNG leaf image to analyse"),
    model: Optional[str] = None,
) -> StreamingResponse:
    """
    Analyse an uploaded tea leaf image.

    This endpoint:
      1. Runs the selected segmentation model (default: EfficientNet-B3+SCSE).
      2. Computes an operational risk report.
      3. Passes the risk to the LangGraph agentic pipeline.
      4. Streams progress and final result as Server-Sent Events (SSE).

    The SSE stream emits events with this shape:
    ```json
    {
      "type": "trace" | "result" | "error",
      "step": 1,
      "label": "Extracting Features",
      "data": { ... }
    }
    ```
    The final ``result`` event contains all analysis fields plus base64-encoded
    original and overlay images.

    Parameters
    ----------
    image : UploadFile
        The uploaded leaf image (JPEG or PNG, any resolution).
    model : str, optional
        Model to use: 'sea_unet', 'resnet34_unet', or 'efficientnet_scse'.
        Default is 'efficientnet_scse' (SOTA).
    """
    # Validate model name if provided
    model_name = model if model else DEFAULT_MODEL
    if model_name not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Valid options: {valid}"
        )
    
    # Validate MIME type early to give a clean error before the stream starts
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Expected an image file, got: {image.content_type}"
        )

    file_bytes = await image.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    return StreamingResponse(
        _analysis_stream(file_bytes, model_name),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering if behind proxy
        },
    )


@app.get("/api/classes", tags=["Metadata"])
async def get_classes() -> Dict[str, Any]:
    """Return the list of tea disease classes the model was trained on."""
    model_info = get_current_model_info()
    return {
        "num_classes": NUM_CLASSES,
        "classes":     CLASS_NAMES,
        "model":       model_info.get("description", ""),
        "current_model": model_info.get("name", DEFAULT_MODEL),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
