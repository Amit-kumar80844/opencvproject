/**
 * AgentAdvicePanel.jsx — LangGraph advisory output with user-friendly display
 * ===========================================================================
 * Redesigned for human readability with clear, actionable insights
 */

import { RealTimeAgentSteps } from './RealTimeAgentSteps';

const RISK_GLOW = {
  GREEN:    'glow-green',
  AMBER:    'glow-amber',
  RED:      'glow-red',
  CRITICAL: 'glow-critical',
};

const RISK_LABELS = {
  GREEN:    { label: 'Healthy', description: 'Your tea plants look good', icon: '✓', color: '#4ade80' },
  AMBER:    { label: 'Attention Needed', description: 'Minor issues detected', icon: '⚠', color: '#fbbf24' },
  RED:      { label: 'Urgent Action', description: 'Significant disease detected', icon: '!', color: '#f87171' },
  CRITICAL: { label: 'Emergency', description: 'Severe outbreak detected', icon: '‼', color: '#ef4444' },
};

const PLACEHOLDER_ADVICE = `Upload a tea leaf image and click "Analyse Leaf" to receive AI-powered disease detection and treatment recommendations.`;

export function AgentAdvicePanel({ result, status, agentStepData = {} }) {
  const hasResult = !!result;
  const tier      = result?.risk_tier ?? 'GREEN';
  const glowClass = RISK_GLOW[tier] ?? 'glow-green';
  const riskInfo  = RISK_LABELS[tier] ?? RISK_LABELS.GREEN;
  
  const isStreaming = status === 'streaming';
  const hasAgentData = Object.keys(agentStepData).length > 0;

  return (
    <div className={`glass ${glowClass} flex flex-col gap-4 p-5 relative z-10 advisory-panel h-full`}>
      {/* Header with Risk Status */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div
            className="w-2 h-12 rounded-full"
            style={{
              background: riskInfo.color,
              boxShadow: `0 0 15px ${riskInfo.color}80`,
            }}
          />
          <div>
            <h2 className="text-lg font-bold tracking-wide text-white">
              Treatment Recommendations
            </h2>
            <p className="text-xs text-slate-300">AI-Powered Agronomic Advisory</p>
          </div>
        </div>
        {hasResult && (
          <div 
            className="risk-status-card"
            style={{ 
              borderColor: riskInfo.color,
              background: `${riskInfo.color}15`,
            }}
          >
            <span 
              className="risk-icon"
              style={{ 
                background: riskInfo.color,
                boxShadow: `0 0 12px ${riskInfo.color}60`,
              }}
            >
              {riskInfo.icon}
            </span>
            <div className="risk-text">
              <span className="risk-label" style={{ color: riskInfo.color }}>{riskInfo.label}</span>
              <span className="risk-desc">{riskInfo.description}</span>
            </div>
          </div>
        )}
      </div>

      {/* Real-time agent activity */}
      {(isStreaming || hasAgentData) && !hasResult && (
        <RealTimeAgentSteps 
          agentStepData={agentStepData} 
          status={status} 
        />
      )}

      {/* Simplified Risk Overview - Human Friendly */}
      {hasResult && (
        <div className="risk-overview">
          <div className="risk-gauge">
            <RiskGauge score={result.severity_pct ?? 0} tier={tier} />
          </div>
          <div className="risk-summary">
            <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
              <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              What This Means For You
            </h3>
            <RiskExplanation tier={tier} score={result.severity_pct ?? 0} />
          </div>
        </div>
      )}

      {/* Cost Impact - Clear Language */}
      {hasResult && (
        <div className="cost-impact-section">
          <div className="cost-card fp-card">
            <div className="cost-header">
              <span className="cost-icon">💰</span>
              <span className="cost-title">Treatment Cost</span>
            </div>
            <div className="cost-value" style={{ color: '#38bdf8' }}>
              Rs. {formatCost(result.fp_cost_lkr)}
            </div>
            <div className="cost-bar">
              <div className="cost-bar-fill fp-fill" style={{ width: `${Math.min(100, (result.fp_cost_lkr || 0) / 50 * 100)}%` }}></div>
            </div>
            <div className="cost-description">
              Estimated cost if treatment applied to healthy areas
            </div>
          </div>
          <div className="cost-card fn-card">
            <div className="cost-header">
              <span className="cost-icon">🌿</span>
              <span className="cost-title">Potential Loss</span>
            </div>
            <div className="cost-value" style={{ color: '#fbbf24' }}>
              Rs. {formatCost(result.fn_cost_lkr)}
            </div>
            <div className="cost-bar">
              <div className="cost-bar-fill fn-fill" style={{ width: `${Math.min(100, (result.fn_cost_lkr || 0) / 200 * 100)}%` }}></div>
            </div>
            <div className="cost-description">
              Estimated loss if disease spreads untreated
            </div>
          </div>
        </div>
      )}

      {/* Uncertainty flag */}
      {result?.uncertainty_flag === 'LOW_CONFIDENCE' && (
        <div className="uncertainty-banner">
          <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
          </svg>
          <div>
            <strong>Verification Recommended</strong>
            <p>The AI had some uncertainty. Please verify with a field inspection.</p>
          </div>
        </div>
      )}

      {/* Treatment Recommendations - Beautiful Cards */}
      <div className="treatment-section">
        {status === 'streaming' && !result?.narrative_summary
          ? <StreamingPlaceholder />
          : hasResult && result?.narrative_summary
            ? <FormattedTreatment text={result.narrative_summary} diseaseClass={result.disease_class} />
            : <PlaceholderCard />}
      </div>

      {/* Evidence Sources */}
      {hasResult && result.citations?.length > 0 && (
        <div className="evidence-section">
          <p className="evidence-header">
            <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6zm2 10a1 1 0 10-2 0v3a1 1 0 102 0v-3zm2-3a1 1 0 011 1v5a1 1 0 11-2 0v-5a1 1 0 011-1zm4-1a1 1 0 10-2 0v7a1 1 0 102 0V8z" clipRule="evenodd" />
            </svg>
            <span>Sources: Tea Research Institute Guidelines</span>
          </p>
          <div className="evidence-list">
            {result.citations.slice(0, 3).map((cit, i) => (
              <span key={i} className="evidence-tag">
                📄 {cit.source || cit}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Helper Functions ── */

function formatCost(cost) {
  if (cost === undefined || cost === null) return '—';
  // Scale up for more meaningful per-hectare values
  const scaledCost = Math.round(cost * 150); // Project to field-scale estimate
  if (scaledCost >= 100000) {
    return (scaledCost / 100000).toFixed(1) + ' Lakh';
  }
  if (scaledCost >= 1000) {
    return (scaledCost / 1000).toFixed(1) + 'K';
  }
  return scaledCost.toLocaleString('en-LK');
}

/* ── Risk Gauge Component ── */
function RiskGauge({ score, tier }) {
  const normalizedScore = Math.min(100, Math.max(0, score));
  const colors = {
    GREEN: '#4ade80',
    AMBER: '#fbbf24',
    RED: '#f87171',
    CRITICAL: '#ef4444',
  };
  const color = colors[tier] ?? colors.GREEN;
  
  // Calculate the arc progress
  const circumference = 157; // Approximate arc length
  const progress = (normalizedScore / 100) * circumference;
  
  return (
    <div className="gauge-container">
      <svg viewBox="0 0 120 80" className="gauge-svg">
        {/* Background arc */}
        <path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Progress arc */}
        <path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${progress} ${circumference}`}
          style={{ 
            filter: `drop-shadow(0 0 8px ${color})`,
            transition: 'stroke-dasharray 0.5s ease',
          }}
        />
        {/* Score display with % */}
        <text x="60" y="52" textAnchor="middle" fill={color} fontSize="26" fontWeight="bold" style={{ textShadow: `0 0 10px ${color}` }}>
          {Math.round(normalizedScore)}%
        </text>
        <text x="60" y="70" textAnchor="middle" fill="#cbd5e1" fontSize="10" fontWeight="500" letterSpacing="0.5">
          LEAF AFFECTED
        </text>
      </svg>
      
      {/* Scale labels */}
      <div className="gauge-labels">
        <span style={{ color: '#4ade80' }}>Low</span>
        <span style={{ color: '#fbbf24' }}>Medium</span>
        <span style={{ color: '#f87171' }}>High</span>
      </div>
    </div>
  );
}

/* ── Risk Explanation ── */
function RiskExplanation({ tier, score }) {
  const explanations = {
    GREEN: (
      <div className="explanation-content">
        <p className="text-slate-300 text-sm leading-relaxed">
          <span className="text-green-400 font-semibold">✓ Good news!</span> Your tea leaves appear healthy with minimal disease indicators. 
          Continue regular monitoring every 3-4 days as a preventive measure.
        </p>
        <div className="recommendation-quick">
          <span className="quick-label">Recommended:</span>
          <span className="quick-action green">Routine monitoring only</span>
        </div>
      </div>
    ),
    AMBER: (
      <div className="explanation-content">
        <p className="text-slate-300 text-sm leading-relaxed">
          <span className="text-amber-400 font-semibold">⚠ Early signs detected.</span> Some disease markers were found, but it's still manageable. 
          Taking action now can prevent significant damage.
        </p>
        <div className="recommendation-quick">
          <span className="quick-label">Recommended:</span>
          <span className="quick-action amber">Treatment within 24-48 hours</span>
        </div>
      </div>
    ),
    RED: (
      <div className="explanation-content">
        <p className="text-slate-300 text-sm leading-relaxed">
          <span className="text-red-400 font-semibold">! Action required!</span> Significant disease presence detected. 
          Apply treatment within the next 12-24 hours to prevent spread.
        </p>
        <div className="recommendation-quick">
          <span className="quick-label">Recommended:</span>
          <span className="quick-action red">Urgent treatment needed</span>
        </div>
      </div>
    ),
    CRITICAL: (
      <div className="explanation-content">
        <p className="text-slate-300 text-sm leading-relaxed">
          <span className="text-red-500 font-semibold">‼ Emergency!</span> Severe disease outbreak detected. 
          Implement emergency treatment immediately and isolate affected plants.
        </p>
        <div className="recommendation-quick">
          <span className="quick-label">Recommended:</span>
          <span className="quick-action critical">Immediate intervention + contact TRI</span>
        </div>
      </div>
    ),
  };
  
  return explanations[tier] ?? explanations.GREEN;
}

/* ── Streaming Placeholder ── */
function StreamingPlaceholder() {
  return (
    <div className="streaming-indicator">
      <div className="streaming-dots">
        <span></span><span></span><span></span>
      </div>
      <p>AI agents are analyzing your image...</p>
      <p className="streaming-sub">Generating personalized treatment recommendations</p>
    </div>
  );
}

/* ── Placeholder Card ── */
function PlaceholderCard() {
  return (
    <div className="placeholder-card">
      <div className="placeholder-icon">🍃</div>
      <p className="placeholder-title">Ready to Analyze</p>
      <p className="placeholder-text">{PLACEHOLDER_ADVICE}</p>
    </div>
  );
}

/* ── Formatted Treatment Recommendations ── */
function FormattedTreatment({ text, diseaseClass }) {
  const SECTION_STYLES = {
    'IMMEDIATE': { 
      icon: '⚡', 
      color: '#fbbf24', 
      bg: 'rgba(251,191,36,0.08)',
      border: 'rgba(251,191,36,0.25)',
      title: 'Immediate Action',
      priority: 1
    },
    'TREATMENT': { 
      icon: '💊', 
      color: '#4ade80', 
      bg: 'rgba(74,222,128,0.08)',
      border: 'rgba(74,222,128,0.25)',
      title: 'Treatment Plan',
      priority: 2
    },
    'MONITORING': { 
      icon: '👁', 
      color: '#38bdf8', 
      bg: 'rgba(56,189,248,0.08)',
      border: 'rgba(56,189,248,0.25)',
      title: 'Follow-Up Care',
      priority: 3
    },
    'PRECAUTIONS': { 
      icon: '🛡️', 
      color: '#a78bfa', 
      bg: 'rgba(167,139,250,0.08)',
      border: 'rgba(167,139,250,0.25)',
      title: 'Safety & Prevention',
      priority: 4
    },
  };

  // Parse sections
  const sections = [];
  const lines = text.split('\n');
  let currentSection = null;
  let currentContent = [];

  for (const line of lines) {
    const headerMatch = line.match(/^(?:Section\s*\d+[\.\:]\s*)?(?:\d+[\.\)]\s*)?(IMMEDIATE\s*ACTION|TREATMENT|MONITORING|PRECAUTIONS)\s*:?\s*(.*)$/i);
    
    if (headerMatch) {
      if (currentSection) {
        sections.push({ key: currentSection, content: currentContent.join('\n').trim() });
      }
      currentSection = headerMatch[1].toUpperCase().replace(/\s+ACTION/, '');
      currentContent = headerMatch[2] ? [headerMatch[2]] : [];
    } else if (line.trim() && !line.match(/^Section\s*\d+/i)) {
      currentContent.push(line);
    }
  }
  
  if (currentSection) {
    sections.push({ key: currentSection, content: currentContent.join('\n').trim() });
  }

  // If no sections parsed, create a default treatment card
  if (sections.length === 0) {
    return (
      <div className="treatment-card" style={{ background: 'rgba(74,222,128,0.08)', borderColor: 'rgba(74,222,128,0.25)' }}>
        <div className="treatment-card-header" style={{ color: '#4ade80' }}>
          <span className="treatment-icon">💊</span>
          <span className="treatment-title">Treatment Recommendation</span>
        </div>
        <div className="treatment-card-content">
          {text.split('\n').filter(l => l.trim()).map((line, i) => (
            <p key={i}>{line}</p>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="treatment-cards">
      {sections.map((section, i) => {
        const style = SECTION_STYLES[section.key] || SECTION_STYLES.TREATMENT;
        const contentLines = section.content.split('\n').filter(l => l.trim());
        
        return (
          <div 
            key={i} 
            className="treatment-card"
            style={{ 
              background: style.bg, 
              borderColor: style.border,
              animationDelay: `${i * 0.1}s`,
            }}
          >
            <div className="treatment-card-header" style={{ color: style.color }}>
              <span className="treatment-icon">{style.icon}</span>
              <span className="treatment-title">{style.title}</span>
              <span className="treatment-step">Step {style.priority}</span>
            </div>
            <div className="treatment-card-content">
              {contentLines.map((line, j) => {
                // Handle bullet points or dashes
                if (line.trim().startsWith('-') || line.trim().startsWith('•') || line.trim().startsWith('*')) {
                  return (
                    <div key={j} className="treatment-bullet">
                      <span className="bullet-dot" style={{ background: style.color }}></span>
                      <span>{line.replace(/^[-•*]\s*/, '').trim()}</span>
                    </div>
                  );
                }
                return <p key={j}>{line}</p>;
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
