/**
 * useAnalysis.js — Custom hook for the full analysis pipeline via SSE
 * ====================================================================
 * BSc Data Science Capstone — NIBM, 2026
 * Author : Manula Fernando
 *
 * This hook manages the Server-Sent Events (SSE) connection to the FastAPI
 * /api/analyze endpoint.  It exposes:
 *   - submitImage(file, model)  : start the streaming analysis with selected model
 *   - reset()                   : clear all state
 *   - status                    : "idle" | "streaming" | "done" | "error"
 *   - traceSteps                : array of { step, label, message, type }
 *   - currentStep               : index of the currently active trace step
 *   - result                    : the final AnalysisResult object (or null)
 *   - error                     : error message string (or null)
 *   - selectedModel             : currently selected model name
 *   - setSelectedModel          : function to change model selection
 *   - availableModels           : list of available models from API
 *
 * I use the native Fetch API + ReadableStream rather than EventSource because
 * EventSource cannot POST form data.  The response body is an SSE stream that
 * I parse line-by-line.
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Use environment variable for API URL in production, fallback to /api for dev proxy
const API_BASE = import.meta.env.VITE_API_URL || '/api';

// Default model from ablation study
const DEFAULT_MODEL = 'efficientnet_scse';

// Available models (will be fetched from API)
const INITIAL_MODELS = {
  sea_unet: {
    description: 'SEA-UNet (Baseline)',
    dice: 0.7536,
    sensitivity: 0.8493,
  },
  resnet34_unet: {
    description: 'ResNet34-UNet (Transfer Learning)',
    dice: 0.7741,
    sensitivity: 0.8726,
  },
  efficientnet_scse: {
    description: 'EfficientNet-B3 + SCSE (SOTA)',
    dice: 0.7845,
    sensitivity: 0.9129,
  },
};

const AGENT_STEPS = [
  { step: 1, label: 'Extracting Features' },
  { step: 2, label: 'Running Model Inference' },
  { step: 3, label: 'Computing Operational Risk' },
  { step: 4, label: 'Querying Evidence Base' },
  { step: 5, label: 'Checking Contraindications' },
  { step: 6, label: 'Validating & Citing' },
  { step: 7, label: 'Analysis Complete' },
];

export function useAnalysis() {
  const [status, setStatus]               = useState('idle');   // idle | streaming | done | error
  const [traceSteps, setTraceSteps]       = useState([]);
  const [currentStep, setCurrentStep]     = useState(0);
  const [result, setResult]               = useState(null);
  const [segmentationResult, setSegmentationResult] = useState(null);  // Early segmentation result
  const [agentStepData, setAgentStepData] = useState({});  // Real-time agent findings { 1: {...}, 2: {...}, 3: {...} }
  const [error, setError]                 = useState(null);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [availableModels, setAvailableModels] = useState(INITIAL_MODELS);
  const abortRef                          = useRef(null);

  // Fetch available models on mount
  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then(res => res.json())
      .then(data => {
        if (data.models) {
          setAvailableModels(data.models);
        }
      })
      .catch(err => console.warn('Could not fetch models:', err));
  }, []);

  const reset = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    setStatus('idle');
    setTraceSteps([]);
    setCurrentStep(0);
    setResult(null);
    setSegmentationResult(null);
    setAgentStepData({});
    setError(null);
  }, []);

  const submitImage = useCallback(async (file, model = null) => {
    reset();
    setStatus('streaming');

    const controller  = new AbortController();
    abortRef.current  = controller;

    const formData = new FormData();
    formData.append('image', file);

    // Use provided model or fall back to selectedModel state
    const modelToUse = model || selectedModel;

    try {
      const response = await fetch(`${API_BASE}/analyze?model=${encodeURIComponent(modelToUse)}`, {
        method: 'POST',
        body:   formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        const txt = await response.text();
        throw new Error(`Server error ${response.status}: ${txt}`);
      }

      // Read the SSE stream line-by-line
      const reader  = response.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();   // keep incomplete last line in buffer

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;

          let event;
          try { event = JSON.parse(jsonStr); }
          catch { continue; }

          const { type, step, label, data } = event;

          if (type === 'trace') {
            setCurrentStep(step);
            setTraceSteps(prev => {
              // Upsert: update existing step or append new one
              const idx = prev.findIndex(s => s.step === step && s.label === label);
              const entry = { step, label, message: data?.message || '', type };
              return idx >= 0
                ? prev.map((s, i) => i === idx ? entry : s)
                : [...prev, entry];
            });
          } else if (type === 'segmentation') {
            // Early segmentation result - show immediately in UI
            setCurrentStep(step);
            setSegmentationResult({
              disease_class: data.disease_class,
              classification_method: data.classification_method,
              severity_pct: data.severity_pct,
              original_image_b64: data.original_image_b64,
              overlay_image_b64: data.overlay_image_b64,
              model_name: data.model_name,
              model_description: data.model_description,
            });
            setTraceSteps(prev => {
              const entry = { step, label: 'Segmentation Complete', message: data?.message || '', type: 'trace' };
              const idx = prev.findIndex(s => s.step === step);
              return idx >= 0 ? prev.map((s, i) => i === idx ? entry : s) : [...prev, entry];
            });
          } else if (type === 'agent_step') {
            // Real-time agent findings - store by agent number
            const agentNum = data.agent_num;
            setCurrentStep(step);
            setAgentStepData(prev => ({
              ...prev,
              [agentNum]: { ...data, timestamp: Date.now() }
            }));
            // Also update trace steps for progress visualization
            const agentLabels = {
              1: "Evidence Retrieval",
              2: "Risk Analysis",
              3: "Treatment Synthesis"
            };
            setTraceSteps(prev => {
              const entry = { 
                step, 
                label: label || agentLabels[agentNum] || `Agent ${agentNum}`, 
                message: data.status === 'thinking' 
                  ? `Agent ${agentNum} is thinking...` 
                  : `Agent ${agentNum} complete (${data.time_ms || 0}ms)`,
                type: 'trace',
                agentData: data
              };
              const idx = prev.findIndex(s => s.step === step);
              return idx >= 0 ? prev.map((s, i) => i === idx ? entry : s) : [...prev, entry];
            });
          } else if (type === 'result') {
            // Add trace entry for final step so it shows as complete
            setTraceSteps(prev => {
              const entry = { step: 7, label: 'Analysis Complete', message: 'Pipeline finished', type: 'trace' };
              const idx = prev.findIndex(s => s.step === 7);
              return idx >= 0 ? prev.map((s, i) => i === idx ? entry : s) : [...prev, entry];
            });
            setResult({ ...data, step, label });
            setCurrentStep(7);
            setStatus('done');
          } else if (type === 'error') {
            throw new Error(data?.message || 'Unknown analysis error');
          }
        }
      }

      if (status !== 'done') setStatus('done');

    } catch (err) {
      if (err.name === 'AbortError') return;
      setError(err.message);
      setStatus('error');
    }
  }, [reset, selectedModel]);

  return {
    status,
    traceSteps,
    currentStep,
    result,
    segmentationResult,  // Early segmentation data for immediate UI update
    agentStepData,       // Real-time agent findings { 1: {...}, 2: {...}, 3: {...} }
    error,
    submitImage,
    reset,
    agentSteps: AGENT_STEPS,
    selectedModel,
    setSelectedModel,
    availableModels,
  };
}
