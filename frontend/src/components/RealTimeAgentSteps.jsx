/**
 * RealTimeAgentSteps.jsx — Animated real-time agent activity display
 * ===================================================================
 * Shows each agent's work as it happens with progressive reveal animations.
 */

import { useState, useEffect, useMemo } from 'react';

const AGENT_CONFIG = {
  1: {
    name: "Evidence Retrieval",
    icon: "🔍",
    color: "#38bdf8",
    bgColor: "rgba(56, 189, 248, 0.1)",
    borderColor: "rgba(56, 189, 248, 0.3)",
    description: "RAG Vector Search",
  },
  2: {
    name: "Risk Analysis",
    icon: "⚠️",
    color: "#fbbf24",
    bgColor: "rgba(251, 191, 36, 0.1)",
    borderColor: "rgba(251, 191, 36, 0.3)",
    description: "Contraindication Check",
  },
  3: {
    name: "Treatment Synthesis",
    icon: "🤖",
    color: "#4ade80",
    bgColor: "rgba(74, 222, 128, 0.1)",
    borderColor: "rgba(74, 222, 128, 0.3)",
    description: "LLM Generation",
  },
};

export function RealTimeAgentSteps({ agentStepData, status }) {
  // Use useMemo to derive visible agents from props (no state needed)
  const visibleAgents = useMemo(() => {
    return Object.keys(agentStepData)
      .map(Number)
      .filter(num => agentStepData[num]?.status === 'complete')
      .sort((a, b) => a - b);
  }, [agentStepData]);

  const isStreaming = status === 'streaming';
  
  // Show thinking indicator for the next agent
  const nextAgent = visibleAgents.length > 0 
    ? Math.max(...visibleAgents) + 1 
    : 1;
  const showThinking = isStreaming && nextAgent <= 3 && !visibleAgents.includes(nextAgent);

  if (!isStreaming && visibleAgents.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-3 mb-4">
      <div className="flex items-center gap-2 mb-1">
        <div className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" />
        <span className="text-xs font-medium text-sky-400 uppercase tracking-wider">
          Agentic Pipeline
        </span>
        {isStreaming && (
          <span className="ml-auto text-xs text-slate-300 font-mono">
            {visibleAgents.length}/3 agents
          </span>
        )}
      </div>
      
      {/* Agent steps */}
      <div className="flex flex-col gap-2">
        {[1, 2, 3].map((agentNum) => {
          const config = AGENT_CONFIG[agentNum];
          const data = agentStepData[agentNum];
          const isComplete = data?.status === 'complete';
          const isThinking = data?.status === 'thinking' || (showThinking && agentNum === nextAgent);
          const isWaiting = !data && !isThinking;
          
          return (
            <AgentStepCard
              key={agentNum}
              agentNum={agentNum}
              config={config}
              data={data}
              isComplete={isComplete}
              isThinking={isThinking}
              isWaiting={isWaiting}
              isStreaming={isStreaming}
            />
          );
        })}
      </div>
    </div>
  );
}

function AgentStepCard({ agentNum, config, data, isComplete, isThinking, isWaiting, isStreaming }) {
  const [expanded, setExpanded] = useState(true);
  
  // Auto-collapse when next agent starts
  useEffect(() => {
    if (isComplete && agentNum < 3) {
      const timer = setTimeout(() => setExpanded(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [isComplete, agentNum]);

  return (
    <div
      className={`rounded-lg border transition-all duration-500 overflow-hidden ${
        isComplete ? 'animate-fade-in' : ''
      }`}
      style={{
        background: isComplete ? config.bgColor : isThinking ? 'rgba(255,255,255,0.02)' : 'transparent',
        borderColor: isComplete ? config.borderColor : isThinking ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.05)',
        opacity: isWaiting && isStreaming ? 0.4 : 1,
      }}
    >
      {/* Header */}
      <div 
        className="flex items-center gap-2 p-2.5 cursor-pointer"
        onClick={() => isComplete && setExpanded(!expanded)}
      >
        {/* Status icon */}
        <div className="w-6 h-6 flex items-center justify-center text-sm">
          {isComplete ? (
            <span>{config.icon}</span>
          ) : isThinking ? (
            <div className="w-4 h-4 border-2 border-sky-400 border-t-transparent rounded-full animate-spin" />
          ) : (
            <div className="w-3 h-3 rounded-full bg-slate-700" />
          )}
        </div>
        
        {/* Agent info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span 
              className="text-xs font-semibold"
              style={{ color: isComplete ? config.color : isThinking ? '#e2e8f0' : '#94a3b8' }}
            >
              Agent {agentNum}: {config.name}
            </span>
            {isComplete && data?.time_ms !== undefined && (
              <span className="text-xs text-slate-300 font-mono">
                {data.time_ms}ms
              </span>
            )}
          </div>
          <span className="text-xs text-slate-300">{config.description}</span>
        </div>
        
        {/* Expand/collapse for complete agents */}
        {isComplete && (
          <svg 
            className={`w-4 h-4 text-slate-300 transition-transform ${expanded ? 'rotate-180' : ''}`}
            viewBox="0 0 20 20" 
            fill="currentColor"
          >
            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        )}
      </div>
      
      {/* Expanded content */}
      {isComplete && expanded && (
        <div 
          className="px-3 pb-3 animate-fade-in"
          style={{ borderTop: `1px solid ${config.borderColor}` }}
        >
          <AgentResult agentNum={agentNum} data={data} />
        </div>
      )}
      
      {/* Thinking indicator */}
      {isThinking && (
        <div className="px-3 pb-3 flex items-center gap-2 text-xs text-slate-300">
          <span className="animate-pulse">Processing</span>
          <span className="flex gap-1">
            <span className="animate-bounce" style={{ animationDelay: '0ms' }}>.</span>
            <span className="animate-bounce" style={{ animationDelay: '150ms' }}>.</span>
            <span className="animate-bounce" style={{ animationDelay: '300ms' }}>.</span>
          </span>
        </div>
      )}
    </div>
  );
}

function AgentResult({ agentNum, data }) {
  if (agentNum === 1) {
    // Evidence Retrieval - show passages
    const passages = data?.passages || [];
    return (
      <div className="mt-2 space-y-2">
        <div className="text-xs text-slate-200 font-medium">
          Found {data?.count || 0} relevant passages:
        </div>
        {passages.slice(0, 2).map((p, i) => (
          <div 
            key={i}
            className="rounded-md p-2 text-xs"
            style={{ background: 'rgba(0,0,0,0.2)' }}
          >
            <p className="text-slate-300 line-clamp-2">{p.text}</p>
            {p.source && (
              <p className="text-sky-500 mt-1 text-[10px]">
                Source: {p.source}
              </p>
            )}
          </div>
        ))}
      </div>
    );
  }
  
  if (agentNum === 2) {
    // Risk Analysis - show contraindications and risk
    return (
      <div className="mt-2 space-y-2">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <span className="text-xs text-slate-300">Risk:</span>
            <span 
              className={`text-xs font-bold px-2 py-0.5 rounded ${
                data?.risk_tier === 'GREEN' ? 'bg-green-500/20 text-green-400' :
                data?.risk_tier === 'AMBER' ? 'bg-amber-500/20 text-amber-400' :
                data?.risk_tier === 'RED' ? 'bg-red-500/20 text-red-400' :
                'bg-red-600/20 text-red-300'
              }`}
            >
              {data?.risk_tier}
            </span>
          </div>
          <span className="text-xs text-slate-300">
            Score: <span className="text-white font-mono">{data?.risk_score?.toFixed(1)}/100</span>
          </span>
        </div>
        {data?.contraindications && (
          <div 
            className="rounded-md p-2 text-xs"
            style={{ background: 'rgba(0,0,0,0.2)' }}
          >
            <p className="text-amber-300/90 line-clamp-3">{data.contraindications}</p>
          </div>
        )}
      </div>
    );
  }
  
  if (agentNum === 3) {
    // Treatment Synthesis - show narrative preview
    return (
      <div className="mt-2 space-y-2">
        <div className="flex items-center gap-2 text-xs text-slate-300">
          <span>Model:</span>
          <span className="font-mono text-green-400">{data?.model}</span>
        </div>
        {data?.narrative_preview && (
          <div 
            className="rounded-md p-2 text-xs"
            style={{ background: 'rgba(0,0,0,0.2)' }}
          >
            <p className="text-slate-300 line-clamp-4 whitespace-pre-wrap">{data.narrative_preview}</p>
          </div>
        )}
      </div>
    );
  }
  
  return null;
}
