/**
 * App.jsx — TeaVision AI · Disease Analysis Command Center
 * =========================================================
 * Main layout: 3-panel glassmorphism design with SSE streaming.
 *
 * Layout map (≥ lg breakpoint):
 *   ┌───────────────────────────────────────┐
 *   │          HEADER  (full width)          │
 *   ├─────────────┬──────────────┬───────────┤
 *   │  Upload     │  Segmentation│  Advisory  │
 *   │  Panel      │  Panel       │  Panel     │
 *   ├─────────────┴──────────────┴───────────┤
 *   │         Reasoning Trace (full width)   │
 *   └───────────────────────────────────────┘
 */

import { useState } from 'react';
import { useAnalysis } from './hooks/useAnalysis';
import { ImageUploadPanel }       from './components/ImageUploadPanel';
import { MaskVisualizationPanel } from './components/MaskVisualizationPanel';
import { AgentAdvicePanel }       from './components/AgentAdvicePanel';
import { ReasoningTrace }         from './components/ReasoningTrace';
import { SettingsPanel, useSettings } from './components/SettingsPanel';

export default function App() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settings = useSettings();
  
  const {
    status,
    traceSteps,
    currentStep,
    result,
    segmentationResult,
    agentStepData,
    error,
    submitImage,
    reset,
    agentSteps,
    selectedModel,
    setSelectedModel,
    availableModels,
  } = useAnalysis();

  const isLoading = status === 'streaming';
  const hasResult = status === 'done' && !!result;
  
  // Use early segmentation result while agent is still processing
  const displayData = result || segmentationResult;
  const hasSegmentation = !!segmentationResult || !!result;

  // Get model display name for current selection
  const modelInfo = availableModels[selectedModel] || {};
  const modelDisplayName = modelInfo.description || selectedModel;

  return (
    <div className="app-container min-h-screen flex flex-col relative">
      {/* Background Image Layer - Tea Plantation */}
      <div 
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: 'url(/bg_img/bg_img.jpeg)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          opacity: settings.bgOpacity / 100,
          filter: 'brightness(1.1) saturate(1.3) contrast(1.05)',
          zIndex: -2,
        }}
      />
      
      {/* Light Gradient Overlay for depth - very subtle */}
      <div 
        className="fixed inset-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse 120% 80% at 20% 10%, rgba(20, 100, 60, 0.1) 0%, transparent 60%),
            radial-gradient(ellipse 100% 60% at 85% 90%, rgba(34, 197, 94, 0.08) 0%, transparent 55%),
            linear-gradient(180deg, rgba(0,0,0,0.15) 0%, transparent 40%, transparent 60%, rgba(0,0,0,0.15) 100%)
          `,
          zIndex: -1,
        }}
      />

      {/* Settings Panel Modal */}
      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />

      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="header-glass relative z-20 flex items-center justify-between px-8 py-5 border-b border-white/10"
        style={{ 
          background: 'rgba(10, 15, 25, 0.4)',
          backdropFilter: 'blur(20px) saturate(180%)',
          WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        }}>
        <div className="flex items-center gap-4">
          {/* Logo mark */}
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{
              background: 'linear-gradient(135deg,rgba(14,165,233,0.25),rgba(74,222,128,0.15))',
              border: '1px solid rgba(74,222,128,0.35)',
              boxShadow: '0 0 20px rgba(74,222,128,0.2)',
            }}
          >
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
              <path d="M11 3C6 5 3 10 5 15 C7 19 11 20 13 19 C17 18 21 13 19 7 C17 3 11 3 11 3Z" fill="none" stroke="#4ade80" strokeWidth="1.2" />
              <line x1="11" y1="3" x2="10" y2="19" stroke="#4ade80" strokeWidth="0.8" strokeOpacity="0.7" />
              <line x1="11" y1="8" x2="7" y2="12" stroke="#4ade80" strokeWidth="0.7" strokeOpacity="0.5" />
              <line x1="11" y1="12" x2="7" y2="16" stroke="#4ade80" strokeWidth="0.7" strokeOpacity="0.5" />
              <line x1="11" y1="8" x2="15" y2="11" stroke="#4ade80" strokeWidth="0.7" strokeOpacity="0.5" />
            </svg>
          </div>

          <div>
            <h1
              className="text-xl font-bold tracking-tight"
              style={{
                background: 'linear-gradient(95deg,#4ade80 0%,#38bdf8 60%,#818cf8 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              TeaVision AI
            </h1>
            <p className="text-xs text-slate-300 tracking-widest uppercase">
              Disease Analysis Command Center
            </p>
          </div>
        </div>

        {/* Model selector + Status badge */}
        <div className="flex items-center gap-4">
          {/* Model selector dropdown */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-300 font-mono">Model:</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isLoading}
              className="bg-slate-800/50 border border-slate-600/50 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:ring-1 focus:ring-sky-400/50 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ minWidth: '220px' }}
            >
              {Object.entries(availableModels).map(([key, model]) => (
                <option key={key} value={key}>
                  {model.description || key}
                </option>
              ))}
            </select>
          </div>

          {/* Settings button */}
          <button
            onClick={() => setSettingsOpen(true)}
            className="p-2 rounded-lg border border-slate-600/50 text-slate-400 hover:text-slate-200 hover:border-slate-500/50 transition-colors"
            title="Settings"
          >
            <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
            </svg>
          </button>

          {/* Status badge */}
          <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
            <span
              className={`w-2 h-2 rounded-full inline-block ${
                isLoading ? 'bg-sky-400 animate-pulse' :
                hasResult  ? 'bg-green-400' :
                error      ? 'bg-red-400'   : 'bg-slate-600'
              }`}
            />
            {isLoading ? 'Analysing…' : hasResult ? 'Analysis complete' : error ? 'Error occurred' : 'Ready'}
          </div>
        </div>
      </header>

      {/* ── Main panels ─────────────────────────────────────────── */}
      <main className="relative z-10 flex-1 px-6 py-6 flex flex-col gap-5">

        {/* Three-column panel row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-stretch">
          {/* 1 — Image upload */}
          <ImageUploadPanel
            onSubmit={submitImage}
            isLoading={isLoading}
            onReset={reset}
            hasResult={hasResult}
          />

          {/* 2 — Segmentation visualisation (shows immediately when ready) */}
          <MaskVisualizationPanel
            originalB64={displayData?.original_image_b64}
            overlayB64={displayData?.overlay_image_b64}
            diseaseClass={displayData?.disease_class}
            severityPct={displayData?.severity_pct}
            isLoading={isLoading && !hasSegmentation}
            modelName={displayData?.model_name || selectedModel}
            modelDescription={displayData?.model_description || modelDisplayName}
            classificationMethod={displayData?.classification_method}
          />

          {/* 3 — Agent advisory */}
          <AgentAdvicePanel
            result={result}
            status={status}
            agentStepData={agentStepData}
          />
        </div>

        {/* Reasoning trace — full width, visible from first streaming event */}
        <ReasoningTrace
          agentSteps={agentSteps}
          traceSteps={traceSteps}
          status={status}
          currentStep={currentStep}
        />

        {/* Error banner */}
        {error && (
          <div className="glass glow-red flex items-start gap-3 p-4 animate-fade-up">
            <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 1.944A11.954 11.954 0 012.166 5C2.056 5.649 2 6.319 2 7c0 5.225 3.34 9.67 8 11.317C14.66 16.67 18 12.225 18 7c0-.682-.057-1.35-.166-2.001A11.954 11.954 0 0110 1.944zM11 14a1 1 0 11-2 0 1 1 0 012 0zm0-7a1 1 0 10-2 0v3a1 1 0 102 0V7z" clipRule="evenodd" />
            </svg>
            <div>
              <p className="text-red-400 text-sm font-semibold mb-0.5">Analysis Error</p>
              <p className="text-red-300/80 text-xs font-mono">{error}</p>
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer className="relative z-10 text-center py-4 text-xs text-slate-700 border-t border-white/5">
        <span className="text-slate-400">TeaVision AI v2.1</span>
        {' · '}
        <span className="text-sky-700">2-Stage Pipeline:</span>
        {' '}
        <span className="text-slate-300">EfficientNet-B3+SCSE (Segmentation) + EfficientNet-B4 512px (Classification)</span>
        {' · '}
        <span className="text-green-700">LangGraph Agentic RAG</span>
        {' · '}
        <span className="text-slate-400">NIBM BSc Data Science Capstone 2025/26</span>
      </footer>
    </div>
  );
}
