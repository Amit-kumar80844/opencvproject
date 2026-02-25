/**
 * MaskVisualizationPanel.jsx — Premium Segmentation Display
 * ==========================================================
 * Large side-by-side original vs disease overlay visualization.
 * Features enhanced glass styling and prominent disease indicators.
 */

export function MaskVisualizationPanel({
  originalB64,
  overlayB64,
  diseaseClass,
  severityPct,
  isLoading,
  modelName,
  modelDescription,
  classificationMethod,
}) {
  const hasResult = originalB64 && overlayB64;
  const displayModel = modelDescription || modelName || 'Model';

  // Severity color gradient
  const getSeverityColor = (pct) => {
    if (pct === undefined || pct === null) return { color: '#64748b', bg: 'rgba(100,116,139,0.15)' };
    if (pct > 60) return { color: '#ef4444', bg: 'rgba(239,68,68,0.15)', label: 'Critical' };
    if (pct > 40) return { color: '#f97316', bg: 'rgba(249,115,22,0.15)', label: 'High' };
    if (pct > 20) return { color: '#fbbf24', bg: 'rgba(251,191,36,0.15)', label: 'Moderate' };
    if (pct > 5) return { color: '#22c55e', bg: 'rgba(34,197,94,0.15)', label: 'Low' };
    return { color: '#22d3ee', bg: 'rgba(34,211,238,0.15)', label: 'Minimal' };
  };

  const severityInfo = getSeverityColor(severityPct);

  return (
    <div className="glass glow-green flex flex-col gap-4 p-5 relative z-10 visualization-panel h-full">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="header-accent w-1.5 h-7 rounded-full" 
          style={{ background: 'linear-gradient(180deg, #22c55e, #14b8a6)', boxShadow: '0 0 12px rgba(34,197,94,0.7)' }} />
        <div className="flex-1">
          <h2 className="text-sm font-semibold tracking-widest uppercase text-slate-200">
            Segmentation Output
          </h2>
          <p className="text-[0.65rem] text-slate-300 mt-0.5">AI Disease Detection Results</p>
        </div>
        {/* Pipeline indicator */}
        <div className="flex items-center gap-1.5 text-[0.6rem] text-slate-300 font-mono">
          <span className="px-2 py-1 rounded bg-green-500/10 border border-green-500/25 text-green-400">U-Net</span>
          <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
          <span className="px-2 py-1 rounded bg-sky-500/10 border border-sky-500/25 text-sky-400">EfficientNet-B4</span>
        </div>
      </div>

      {/* Image viewport */}
      {isLoading && !hasResult ? (
        /* Enhanced loading state */
        <div className="loading-viewport flex flex-col items-center justify-center min-h-[400px] rounded-2xl overflow-hidden"
          style={{ background: 'linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))' }}>
          <div className="relative">
            {/* Pulsing ring */}
            <div className="absolute inset-0 rounded-full animate-ping-slow border-2 border-green-400/30\" style={{ width: '80px', height: '80px', margin: 'auto' }} />
            <svg className="w-16 h-16 text-green-400 animate-spin" style={{ animationDuration: '2s' }} viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeOpacity="0.2" />
              <path d="M12 2 A10 10 0 0 1 22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </div>
          <p className="text-slate-300 text-base mt-5 font-medium">Running {displayModel}...</p>
          <p className="text-slate-300 text-sm mt-1">Analyzing disease patterns</p>
        </div>
      ) : hasResult ? (
        /* Images stacked vertically for better visibility */
        <div className="image-viewport">
          <div className="flex flex-col gap-5">
            {/* Original Image */}
            <div className="image-frame relative">
              <p className="text-xs text-slate-300 tracking-widest uppercase mb-2 font-semibold flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-400"></span>
                Original Leaf Image
              </p>
              <div className="image-container relative rounded-2xl overflow-hidden group">
                <img
                  src={originalB64}
                  alt="Original leaf"
                  className="w-full rounded-2xl object-contain"
                  style={{ maxHeight: '280px', minHeight: '180px' }}
                />
                <div className="image-border absolute inset-0 rounded-2xl pointer-events-none"
                  style={{ 
                    border: '1px solid rgba(56,189,248,0.3)',
                    boxShadow: 'inset 0 0 30px rgba(0,0,0,0.2)'
                  }} />
              </div>
            </div>

            {/* Disease Overlay */}
            <div className="image-frame relative">
              <p className="text-xs text-slate-300 tracking-widest uppercase mb-2 font-semibold flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse"></span>
                Disease Detection Overlay
              </p>
              <div className="image-container relative rounded-2xl overflow-hidden">
                <img
                  src={overlayB64}
                  alt="Disease overlay"
                  className="w-full rounded-2xl object-contain"
                  style={{ maxHeight: '280px', minHeight: '180px' }}
                />
                <div className="image-border absolute inset-0 rounded-2xl pointer-events-none"
                  style={{ 
                    border: '1px solid rgba(34,197,94,0.4)',
                    boxShadow: 'inset 0 0 40px rgba(34,197,94,0.1), 0 0 40px rgba(34,197,94,0.08)'
                  }} />
                {/* Disease indicator overlay */}
                {diseaseClass && diseaseClass !== 'healthy' && (
                  <div className="absolute top-3 right-3 px-3 py-1.5 rounded-lg text-xs font-bold uppercase"
                    style={{
                      background: 'rgba(239,68,68,0.9)',
                      color: 'white',
                      boxShadow: '0 0 15px rgba(239,68,68,0.5)',
                    }}>
                    Affected Area
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Disease Classification Stats */}
          <div className="stats-row mt-5 pt-4 border-t border-white/10">
            <div className="grid grid-cols-3 gap-3">
              {/* Disease Class */}
              <div className="stat-card flex flex-col items-center justify-center p-4 rounded-xl"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)' }}>
                <p className="text-xs text-slate-200 uppercase tracking-wider mb-2 font-medium">Predicted Class</p>
                <p className="text-xl font-bold capitalize text-center leading-tight" style={{ color: diseaseClass === 'healthy' ? '#22c55e' : '#f97316' }}>
                  {diseaseClass ? diseaseClass.replace(/_/g, ' ') : '—'}
                </p>
                {classificationMethod && (
                  <span className="mt-2 px-2.5 py-1 rounded text-xs bg-sky-500/20 text-sky-300 border border-sky-500/30 font-medium">
                    via {classificationMethod}
                  </span>
                )}
              </div>

              {/* Severity */}
              <div className="stat-card flex flex-col items-center justify-center p-4 rounded-xl"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)' }}>
                <p className="text-xs text-slate-200 uppercase tracking-wider mb-2 font-medium">Severity</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-3xl font-bold" style={{ color: severityInfo.color }}>
                    {typeof severityPct === 'number' ? severityPct.toFixed(1) : '—'}
                  </span>
                  <span className="text-base text-slate-300 font-medium">%</span>
                </div>
                {severityInfo.label && (
                  <span className="mt-2 px-2.5 py-1 rounded text-xs font-bold uppercase"
                    style={{ background: severityInfo.bg, color: severityInfo.color, border: `1px solid ${severityInfo.color}50` }}>
                    {severityInfo.label}
                  </span>
                )}
              </div>

              {/* Severity Bar */}
              <div className="stat-card flex flex-col justify-center p-4 rounded-xl"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)' }}>
                <p className="text-xs text-slate-200 uppercase tracking-wider mb-3 font-medium">Affected Area</p>
                <div className="severity-bar h-4 rounded-full bg-slate-700/50 overflow-hidden relative">
                  <div
                    className="h-full rounded-full transition-all duration-1000"
                    style={{
                      width: `${Math.min(100, severityPct ?? 0)}%`,
                      background: `linear-gradient(90deg, ${severityInfo.color}cc, ${severityInfo.color})`,
                      boxShadow: `0 0 15px ${severityInfo.color}80`,
                    }}
                  />
                  {/* Tick marks */}
                  <div className="absolute inset-0 flex justify-between px-0.5">
                    {[25, 50, 75].map(tick => (
                      <div key={tick} className="w-px h-full bg-white/15" style={{ marginLeft: `${tick}%` }} />
                    ))}
                  </div>
                </div>
                <div className="flex justify-between mt-2 text-xs text-slate-200 font-medium">
                  <span>0%</span>
                  <span>50%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Skeleton placeholder - vertical layout */
        <div className="placeholder-viewport flex flex-col items-center justify-center min-h-[400px] rounded-2xl p-6"
          style={{ background: 'linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))' }}>
          <div className="flex flex-col gap-4 w-full opacity-40">
            {[0, 1].map((i) => (
              <div
                key={i}
                className="rounded-2xl animate-shimmer"
                style={{
                  height: '160px',
                  background: 'linear-gradient(90deg,rgba(255,255,255,0.02) 25%,rgba(255,255,255,0.06) 50%,rgba(255,255,255,0.02) 75%)',
                  backgroundSize: '200% 100%',
                  border: '1px solid rgba(255,255,255,0.08)',
                }}
              />
            ))}
          </div>
          <div className="flex flex-col items-center gap-3 mt-6">
            <svg width="56" height="56" viewBox="0 0 48 48" fill="none" className="text-slate-300">
              <rect x="4" y="4" width="40" height="40" rx="8" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 4" />
              <path d="M17 31l6-8 5 6 6-10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="16" cy="18" r="3" stroke="currentColor" strokeWidth="1.5" />
            </svg>
            <p className="text-slate-200 text-sm text-center font-medium">
              Upload and analyse a leaf to see<br />segmentation results
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
