# TeaVision AI — Architecture Diagrams

> **Project:** BSc Data Science Capstone — NIBM, 2026
> **Author:** Manula Fernando
> **Last Updated:** February 2026

This document contains all formal architecture diagrams required for the technical report.

---

## 1. System Architecture Diagram

The high-level system architecture showing all major components and their interactions.

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[React + Vite UI<br/>Glassmorphism Design]
        UPLOAD[Image Upload Panel]
        MASK[Mask Visualization]
        ADVICE[Agent Advice Panel]
        TRACE[Reasoning Trace]
        SETTINGS[Settings Panel]
    end
    
    subgraph "API Layer"
        FASTAPI[FastAPI Server<br/>Port 8000]
        SSE[SSE Streaming<br/>Real-time Updates]
        CORS[CORS Middleware]
    end
    
    subgraph "Vision Pipeline"
        PREPROCESS[Image Preprocessing<br/>256×256, Normalize]
        SEGMODEL[EfficientNet-B3 + SCSE<br/>U-Net Segmentation]
        CLASSMODEL[EfficientNet-B4<br/>8-Class Classification]
        POSTPROCESS[Post-processing<br/>Overlay, Contours]
    end
    
    subgraph "Agent Pipeline"
        LANGGRAPH[LangGraph<br/>State Machine]
        AGENT1[Evidence Retrieval<br/>Agent]
        AGENT2[Risk & Contraindication<br/>Agent]
        AGENT3[Validation & Citation<br/>Agent]
    end
    
    subgraph "Knowledge Layer"
        CHROMA[(ChromaDB<br/>Vector Store)]
        GUIDELINES[Tea Agronomy<br/>Guidelines]
        TREATMENTS[Treatment<br/>Protocols]
        CONTRA[Contraindications<br/>& Safety]
    end
    
    subgraph "LLM Layer"
        OLLAMA[Ollama Server<br/>Port 11434]
        QWEN[qwen2:1.5b<br/>Fast Model]
        PHI3[phi3<br/>Fallback]
    end
    
    UI --> FASTAPI
    UPLOAD --> FASTAPI
    FASTAPI --> SSE --> UI
    FASTAPI --> PREPROCESS
    PREPROCESS --> SEGMODEL
    SEGMODEL -->|severity ≥ 5%| CLASSMODEL
    SEGMODEL --> POSTPROCESS
    CLASSMODEL --> POSTPROCESS
    POSTPROCESS --> LANGGRAPH
    
    LANGGRAPH --> AGENT1
    AGENT1 --> AGENT2
    AGENT2 --> AGENT3
    
    AGENT1 --> CHROMA
    AGENT2 --> CHROMA
    AGENT3 --> OLLAMA
    
    CHROMA --> GUIDELINES
    CHROMA --> TREATMENTS
    CHROMA --> CONTRA
    
    OLLAMA --> QWEN
    OLLAMA --> PHI3
    
    AGENT3 --> SSE
    
    MASK --> UI
    ADVICE --> UI
    TRACE --> UI
    SETTINGS --> UI
```

---

## 2. Data Flow Diagram

Shows the complete data flow from image upload to treatment recommendation.

```mermaid
flowchart LR
    subgraph Input
        IMG["Tea Leaf Image (RGB)"]
    end
    
    subgraph Preprocessing
        RESIZE["Resize 256x256"]
        NORM["Normalize ImageNet"]
        TENSOR["Convert to Tensor"]
    end
    
    subgraph Segmentation
        UNET["EfficientNet-B3 + SCSE"]
        SIGMOID["Sigmoid Activation"]
        THRESH["Threshold 0.5"]
        MASK["Binary Disease Mask"]
        SEVERITY["Calculate Severity"]
    end
    
    subgraph Decision
        CHECK{"Severity >= 5%?"}
    end
    
    subgraph Classification
        RESIZE512["Resize 512x512"]
        EFFNET["EfficientNet-B4"]
        SOFTMAX["Softmax 8 Classes"]
        DISEASE["Disease Class"]
    end
    
    subgraph RiskAnalysis
        FP["FP Cost ~LKR 2,500"]
        FN["FN Cost ~LKR 30,000"]
        RISK["Composite Risk Score"]
        TIER["Risk Tier"]
    end
    
    subgraph AgentPipeline
        RAG["ChromaDB RAG Query"]
        LLM["Ollama LLM Synthesis"]
        CITE["Citation Extraction"]
    end
    
    subgraph Output
        OVERLAY["Mask Overlay"]
        ADVICE["Treatment Recommendation"]
        CITATIONS["TRI Citations"]
        COSTS["FP/FN Cost Analysis"]
    end
    
    IMG --> RESIZE --> NORM --> TENSOR
    TENSOR --> UNET --> SIGMOID --> THRESH --> MASK
    MASK --> SEVERITY
    SEVERITY --> CHECK
    CHECK -->|Yes| RESIZE512 --> EFFNET --> SOFTMAX --> DISEASE
    CHECK -->|No| HEALTHY["Healthy Classification"]
    
    DISEASE --> FP
    DISEASE --> FN
    FP --> RISK
    FN --> RISK
    RISK --> TIER
    
    TIER --> RAG --> LLM --> CITE
    
    MASK --> OVERLAY
    CITE --> ADVICE
    CITE --> CITATIONS
    RISK --> COSTS
```

---

## 3. ML Pipeline Diagram

Detailed machine learning training and inference pipeline.

```mermaid
flowchart TB
    subgraph "Data Pipeline"
        RAW[Raw Dataset<br/>885 Images, 8 Classes]
        SPLIT[Stratified Split<br/>70/15/15]
        TRAIN_SET[Training Set<br/>619 Images]
        VAL_SET[Validation Set<br/>133 Images]
        TEST_SET[Test Set<br/>133 Images]
    end
    
    subgraph "Mask Generation"
        GRABCUT[GrabCut<br/>Algorithm]
        HSV[Disease-Specific<br/>HSV Windows]
        PSEUDO[Pseudo<br/>Segmentation Masks]
    end
    
    subgraph "Augmentation Pipeline"
        BRIGHT[RandomBrightnessContrast]
        HUE[HueSaturationValue]
        ELASTIC[ElasticTransform]
        FLIP[HFlip + VFlip]
        ROTATE[RandomRotate90]
        CUTOUT[CoarseDropout]
    end
    
    subgraph "Model Architecture"
        ENCODER[EfficientNet-B3<br/>ImageNet Pretrained]
        DECODER[U-Net Decoder<br/>Skip Connections]
        SCSE[SCSE Attention<br/>Channel + Spatial]
        HEAD[Segmentation Head<br/>1 Channel Output]
    end
    
    subgraph "Training Loop"
        ADAMW[AdamW Optimizer<br/>lr=1e-4, wd=1e-4]
        ONECYCLE[OneCycleLR<br/>Scheduler]
        TVERSKY[Focal Tversky Loss<br/>α=0.3, β=0.7, γ=0.75]
        CLIP[Gradient Clipping<br/>max_norm=1.0]
        EARLY[Early Stopping<br/>patience=15]
    end
    
    subgraph "Evaluation Metrics"
        DICE[Dice Coefficient<br/>0.7845]
        IOU[IoU / Jaccard<br/>0.6511]
        SENS[Sensitivity<br/>0.9129]
        SPEC[Specificity<br/>0.9784]
    end
    
    subgraph "Model Export"
        CKPT[Best Checkpoint<br/>.pth File]
        DEPLOY[Production<br/>Deployment]
    end
    
    RAW --> SPLIT
    SPLIT --> TRAIN_SET
    SPLIT --> VAL_SET
    SPLIT --> TEST_SET
    
    TRAIN_SET --> GRABCUT --> HSV --> PSEUDO
    
    PSEUDO --> BRIGHT --> HUE --> ELASTIC --> FLIP --> ROTATE --> CUTOUT
    
    CUTOUT --> ENCODER --> DECODER --> SCSE --> HEAD
    
    HEAD --> ADAMW
    ADAMW --> ONECYCLE --> TVERSKY --> CLIP --> EARLY
    
    EARLY --> DICE
    EARLY --> IOU
    EARLY --> SENS
    EARLY --> SPEC
    
    DICE --> CKPT --> DEPLOY
```

---

## 4. LangGraph Agent Interaction Diagram

Detailed view of the multi-agent orchestration system.

```mermaid
stateDiagram-v2
    [*] --> InputState
    
    state InputState {
        direction LR
        disease_class: str
        risk_tier: GREEN/AMBER/RED/CRITICAL
        severity_pct: float
        fp_cost_lkr: int
        fn_cost_lkr: int
    }
    note right of InputState : Risk Report from Vision Model
    
    InputState --> Agent1
    
    state Agent1 {
        [*] --> QueryGuidelines
        QueryGuidelines: Query ChromaDB
        QueryGuidelines: 'tea_agronomy_guidelines'
        QueryGuidelines --> RetrievePassages
        RetrievePassages: Retrieve Top-4
        RetrievePassages: Relevant Passages
        RetrievePassages --> ExtractCitations
        ExtractCitations: Extract TRI
        ExtractCitations: Source Citations
        ExtractCitations --> [*]
    }
    
    Agent1 --> Agent2: Passages + Citations
    
    state Agent2 {
        [*] --> QueryTreatment
        QueryTreatment: Query ChromaDB
        QueryTreatment: 'tea_treatment_protocols'
        QueryTreatment --> QueryContra
        QueryContra: Query ChromaDB
        QueryContra: 'tea_contraindications'
        QueryContra --> AnalyzeRisk
        AnalyzeRisk: Analyze MRL Limits
        AnalyzeRisk: PHI Days, Eco Buffers
        AnalyzeRisk --> [*]
    }
    
    Agent2 --> Agent3: Treatment + Contraindications
    
    state Agent3 {
        [*] --> CheckOllama
        CheckOllama: Check Ollama
        CheckOllama: Availability
        CheckOllama --> LLMSynth: Available
        CheckOllama --> RuleBased: Unavailable
        
        LLMSynth: Ollama LLM
        LLMSynth: qwen2:1.5b Synthesis
        
        RuleBased: Rule-Based
        RuleBased: Fallback Response
        
        LLMSynth --> FormatOutput
        RuleBased --> FormatOutput
        
        FormatOutput: Format 4-Section
        FormatOutput: Treatment Plan
        FormatOutput --> SetUncertainty
        SetUncertainty: Set Uncertainty
        SetUncertainty: Flag if Needed
        SetUncertainty --> [*]
    }
    
    Agent3 --> OutputState
    
    state OutputState {
        direction LR
        treatment_plan: str
        citations: List_str
        uncertainty_flag: str
        narrative_summary: str
        agent_trace: List_dict
    }
    note right of OutputState : Final Recommendation
    
    OutputState --> [*]
```

---

## 5. Component Interaction Diagram

Shows how all software components interact at runtime.

```mermaid
sequenceDiagram
    participant User
    participant Frontend as React Frontend
    participant API as FastAPI Server
    participant Vision as Vision Pipeline
    participant Agent as LangGraph Agents
    participant ChromaDB as ChromaDB
    participant Ollama as Ollama LLM
    
    User->>Frontend: Upload tea leaf image
    Frontend->>API: POST /api/analyze (multipart)
    
    API->>API: Preprocess image (256×256)
    API->>Vision: Run EfficientNet-B3+SCSE
    Vision-->>API: Binary mask + severity %
    
    alt Severity ≥ 5%
        API->>Vision: Run EfficientNet-B4 classifier
        Vision-->>API: Disease class (8 classes)
    else Severity < 5%
        API->>API: Classify as "Healthy"
    end
    
    API->>API: Calculate operational risk (FP/FN)
    API-->>Frontend: SSE: "Running Inference" ✓
    
    API->>Agent: run_agent_pipeline(risk_report)
    
    Agent->>ChromaDB: Query guidelines collection
    ChromaDB-->>Agent: Top-4 relevant passages
    API-->>Frontend: SSE: "Evidence Retrieved" ✓
    
    Agent->>ChromaDB: Query treatment protocols
    Agent->>ChromaDB: Query contraindications
    ChromaDB-->>Agent: Treatment + safety info
    API-->>Frontend: SSE: "Contraindications Checked" ✓
    
    Agent->>Ollama: Generate treatment synthesis
    Ollama-->>Agent: Structured recommendation
    API-->>Frontend: SSE: "Validation Complete" ✓
    
    Agent-->>API: Final treatment recommendation
    API-->>Frontend: SSE: Complete result payload
    
    Frontend->>Frontend: Render mask overlay
    Frontend->>Frontend: Display treatment advice
    Frontend->>Frontend: Show risk tier badge
    
    Frontend-->>User: Complete analysis displayed
```

---

## 6. Deployment Architecture Diagram

Production deployment configuration.

```mermaid
graph TB
    subgraph "Client Layer"
        BROWSER[Web Browser<br/>Chrome/Firefox/Safari]
        MOBILE[Mobile Browser<br/>Responsive Design]
    end
    
    subgraph "Frontend Deployment"
        VITE[Vite Dev Server<br/>Port 5173]
        STATIC[Static Build<br/>dist/]
    end
    
    subgraph "API Server"
        UVICORN[Uvicorn ASGI<br/>Port 8000]
        FASTAPI_APP[FastAPI Application]
        PYTORCH[PyTorch Models<br/>CPU/GPU]
    end
    
    subgraph "Model Checkpoints"
        SEG_MODEL[efficientnet_scse_best.pth<br/>48 MB]
        CLS_MODEL[EfficientNetB4_512_model.pth<br/>76 MB]
    end
    
    subgraph "Vector Database"
        CHROMA_DB[(ChromaDB<br/>Persistent Storage)]
        EMBED[all-MiniLM-L6-v2<br/>Embedding Model]
    end
    
    subgraph "LLM Server"
        OLLAMA_SRV[Ollama Server<br/>Port 11434]
        MODEL_CACHE[Model Cache<br/>~/.ollama/models]
    end
    
    subgraph "Storage"
        LOGS[JSON Logs<br/>backend/logs/]
        OUTPUTS[Model Outputs<br/>backend/outputs/]
        UPLOADS[Temp Uploads<br/>In-Memory]
    end
    
    BROWSER --> VITE
    MOBILE --> VITE
    VITE --> STATIC
    
    VITE -->|API Proxy| UVICORN
    UVICORN --> FASTAPI_APP
    FASTAPI_APP --> PYTORCH
    
    PYTORCH --> SEG_MODEL
    PYTORCH --> CLS_MODEL
    
    FASTAPI_APP --> CHROMA_DB
    CHROMA_DB --> EMBED
    
    FASTAPI_APP --> OLLAMA_SRV
    OLLAMA_SRV --> MODEL_CACHE
    
    FASTAPI_APP --> LOGS
    FASTAPI_APP --> OUTPUTS
    FASTAPI_APP --> UPLOADS
```

---

## 7. Class Diagram — Core Domain Models

```mermaid
classDiagram
    class AgentState {
        +str disease_class
        +str risk_tier
        +float severity_pct
        +int fp_cost_lkr
        +int fn_cost_lkr
        +str recommended_action
        +List~str~ retrieved_passages
        +List~str~ citations
        +str treatment_plan
        +str uncertainty_flag
        +List~dict~ agent_trace
    }
    
    class OperationalRiskReport {
        +str disease_class
        +int fp_count
        +int fn_count
        +int fp_cost_lkr
        +int fn_cost_lkr
        +float composite_score
        +str risk_tier
        +str recommended_action
        +str explanation
        +calculate_risk()
    }
    
    class TreatmentRecommendation {
        +str treatment_plan
        +List~str~ citations
        +str uncertainty_flag
        +str narrative_summary
        +str kb_source
    }
    
    class SEAUNet {
        +nn.Module encoder
        +nn.Module decoder
        +nn.Module segmentation_head
        +forward(x) Tensor
    }
    
    class EfficientNetClassifier {
        +str model_name
        +int num_classes
        +forward(x) Tensor
    }
    
    class ChromaDBStore {
        +Collection guidelines
        +Collection treatments
        +Collection contraindications
        +query_guidelines(disease, tier)
        +query_treatment_protocol(disease)
        +query_contraindications(disease)
    }
    
    AgentState --> TreatmentRecommendation : produces
    OperationalRiskReport --> AgentState : initializes
    SEAUNet --> OperationalRiskReport : generates
    EfficientNetClassifier --> OperationalRiskReport : classifies
    ChromaDBStore --> AgentState : provides knowledge
```

---

## Diagram Rendering Notes

These diagrams are written in **Mermaid** syntax and can be rendered:

1. **GitHub**: Automatically renders in README.md and markdown files
2. **VS Code**: Use "Markdown Preview Mermaid Support" extension
3. **Online**: Use [Mermaid Live Editor](https://mermaid.live)
4. **Report**: Export as PNG/SVG for PDF inclusion

### Export Commands (using Mermaid CLI)

```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Export to PNG
mmdc -i ARCHITECTURE_DIAGRAMS.md -o diagrams/ -e png

# Export to SVG (vector, better for reports)
mmdc -i ARCHITECTURE_DIAGRAMS.md -o diagrams/ -e svg
```

---

*Generated: February 2026*
