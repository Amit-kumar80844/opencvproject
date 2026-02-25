/**
 * ReasoningTrace.jsx — Glowing 7-step agent progress stepper
 * ============================================================
 * Each step transitions idle → active (pulsing) → done (checkmark).
 */

import { useMemo } from 'react';

const STATUS_ICON = {
  idle:   <span className="w-3 h-3 rounded-full bg-slate-700 border border-slate-600 flex-shrink-0" />,
  active: (
    <span className="trace-step active flex-shrink-0">
      <svg className="animate-spin-slow w-4 h-4 text-sky-400" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeOpacity="0.2" />
        <path d="M12 2 A10 10 0 0 1 22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </span>
  ),
  done: (
    <span className="trace-step done flex-shrink-0">
      <svg className="w-4 h-4 text-green-400" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414L8.414 15 3.293 9.879a1 1 0 111.414-1.414L8.414 12.172l6.879-6.879a1 1 0 011.414 0z" clipRule="evenodd" />
      </svg>
    </span>
  ),
  skipped: (
    <span className="trace-step skipped flex-shrink-0">
      <svg className="w-4 h-4 text-slate-500" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-11a.75.75 0 00-1.5 0v4.59L7.3 9.24a.75.75 0 00-1.1 1.02l3.25 3.5a.75.75 0 001.1 0l3.25-3.5a.75.75 0 10-1.1-1.02l-1.95 2.1V7z" clipRule="evenodd" />
      </svg>
    </span>
  ),
  error: (
    <span className="trace-step error flex-shrink-0">
      <svg className="w-4 h-4 text-red-400" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
      </svg>
    </span>
  ),
};

export function ReasoningTrace({ agentSteps = [], traceSteps = [], status, currentStep = 0, logs = [] }) {
  // Build a map of step number (1-indexed) to trace data
  const traceMap = useMemo(() => {
    const map = {};
    for (const t of traceSteps) {
      map[t.step] = t;
    }
    return map;
  }, [traceSteps]);

  // Find the max traced step to detect skipped steps
  const maxTracedStep = useMemo(() => {
    return Math.max(0, ...traceSteps.map(t => t.step));
  }, [traceSteps]);

  if (status === 'idle') return null;

  return (
    <div className="glass flex flex-col gap-4 p-6" style={{ borderColor: 'rgba(148,163,184,0.18)' }}>
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-2 h-6 rounded-full" style={{ background: '#94a3b8', boxShadow: '0 0 8px rgba(148,163,184,0.5)' }} />
        <h2 className="text-sm font-semibold tracking-widest uppercase text-slate-300">
          Reasoning Trace
        </h2>
        {status === 'streaming' && (
          <span className="ml-auto text-xs text-sky-400 font-mono tracking-wide animate-pulse">
            ● LIVE
          </span>
        )}
        {status === 'done' && (
          <span className="ml-auto text-xs text-green-400 font-mono tracking-wide">
            ✓ COMPLETE
          </span>
        )}
        {status === 'error' && (
          <span className="ml-auto text-xs text-red-400 font-mono tracking-wide">
            ✗ ERROR
          </span>
        )}
      </div>

      {/* Step list */}
      <ol className="flex flex-col gap-0">
        {agentSteps.map((stepDef, idx) => {
          // Steps are 1-indexed from backend
          const stepNum = stepDef.step;
          const traceStep = traceMap[stepNum];
          
          // Derive status from currentStep and trace data
          let stepStatus = 'idle';
          if (status === 'error' && currentStep === stepNum) {
            stepStatus = 'error';
          } else if (traceStep) {
            // Step has trace data
            if (currentStep > stepNum || status === 'done') {
              stepStatus = 'done';
            } else if (currentStep === stepNum) {
              stepStatus = 'active';
            }
          } else if (maxTracedStep > stepNum) {
            // No trace for this step, but later steps were traced = SKIPPED
            stepStatus = 'skipped';
          }

          return (
            <li key={idx} className="flex gap-4 relative">
              {/* Connector line */}
              {idx < agentSteps.length - 1 && (
                <div
                  className="absolute left-[15px] top-6 w-px"
                  style={{
                    height: 'calc(100% - 4px)',
                    background: stepStatus === 'done'
                      ? 'linear-gradient(180deg,rgba(74,222,128,0.5),rgba(74,222,128,0.1))'
                      : stepStatus === 'skipped'
                      ? 'linear-gradient(180deg,rgba(100,116,139,0.3),rgba(100,116,139,0.1))'
                      : 'rgba(255,255,255,0.07)',
                  }}
                />
              )}

              {/* Icon column */}
              <div className="flex-shrink-0 w-8 flex justify-center pt-3">
                {STATUS_ICON[stepStatus] ?? STATUS_ICON.idle}
              </div>

              {/* Content */}
              <div
                className="flex-1 pb-5 min-w-0"
                style={{ opacity: stepStatus === 'idle' ? 0.4 : stepStatus === 'skipped' ? 0.5 : 1, transition: 'opacity 0.3s' }}
              >
                <p
                  className="text-sm font-medium"
                  style={{
                    color: stepStatus === 'done'
                      ? '#4ade80'
                      : stepStatus === 'active'
                      ? '#38bdf8'
                      : stepStatus === 'skipped'
                      ? '#64748b'
                      : '#94a3b8',
                    textShadow: stepStatus === 'done'
                      ? '0 0 8px rgba(74,222,128,0.5)'
                      : stepStatus === 'active'
                      ? '0 0 8px rgba(56,189,248,0.5)'
                      : 'none',
                    transition: 'color 0.3s, text-shadow 0.3s',
                    textDecoration: stepStatus === 'skipped' ? 'line-through' : 'none',
                  }}
                >
                  {stepDef.label}{stepStatus === 'skipped' ? ' (fast mode)' : ''}
                </p>

                {/* Streaming message */}
                {traceStep?.message && (
                  <p className="text-xs text-slate-300 mt-0.5 font-mono truncate">
                    {traceStep.message}
                  </p>
                )}

                {/* Active pulse bar */}
                {stepStatus === 'active' && (
                  <div className="mt-1.5 h-1 w-full rounded-full overflow-hidden bg-white/5">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: '60%',
                        background: 'linear-gradient(90deg,transparent,#38bdf8,transparent)',
                        animation: 'shimmer 1.5s infinite',
                        backgroundSize: '200% 100%',
                      }}
                    />
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
