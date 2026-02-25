/**
 * LogConsole.jsx — Real-time analysis log viewer
 * ===============================================
 * Shows timestamped log entries for debugging and transparency.
 */

import { useEffect, useRef } from 'react';

const LOG_COLORS = {
  info:    'text-slate-300',
  trace:   'text-sky-400',
  success: 'text-green-400',
  error:   'text-red-400',
  warn:    'text-amber-400',
};

export function LogConsole({ logs = [], isExpanded = false, onToggle }) {
  const scrollRef = useRef(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  if (logs.length === 0) return null;

  return (
    <div className="glass flex flex-col" style={{ borderColor: 'rgba(148,163,184,0.18)' }}>
      {/* Header */}
      <button
        onClick={onToggle}
        className="flex items-center gap-3 p-4 hover:bg-white/5 transition-colors w-full text-left"
      >
        <div className="w-2 h-6 rounded-full" style={{ background: '#818cf8', boxShadow: '0 0 8px rgba(129,140,248,0.5)' }} />
        <h2 className="text-sm font-semibold tracking-widest uppercase text-slate-300">
          Live Console
        </h2>
        <span className="ml-auto text-xs text-slate-300 font-mono">
          {logs.length} entries
        </span>
        <svg
          className={`w-4 h-4 text-slate-300 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
        </svg>
      </button>

      {/* Log entries */}
      {isExpanded && (
        <div
          ref={scrollRef}
          className="max-h-48 overflow-y-auto p-4 pt-0 font-mono text-xs space-y-1"
          style={{ scrollBehavior: 'smooth' }}
        >
          {logs.map((log, idx) => (
            <div key={idx} className="flex gap-3">
              <span className="text-slate-400 flex-shrink-0">{log.timestamp}</span>
              <span className={LOG_COLORS[log.type] || 'text-slate-300'}>
                {log.message}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
