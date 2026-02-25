/**
 * SettingsPanel.jsx — UI Customization Settings
 * ==============================================
 * Allows users to adjust glass transparency, background visibility,
 * toggle animations, and customize color themes.
 */

import { useState, useEffect } from 'react';

const STORAGE_KEY = 'teavision_settings';

const DEFAULT_SETTINGS = {
  glassOpacity: 20,          // 0-90 (percentage)
  bgOpacity: 85,             // 0-100 (background image visibility)
  animationsEnabled: true,
  colorTheme: 'green',       // green, blue, amber, gold
  showCostHelp: true,
};

const COLOR_THEMES = {
  green: {
    name: 'Tea Garden',
    primary: '#22c55e',
    glow: 'rgba(34, 197, 94, 0.8)',
  },
  blue: {
    name: 'Ocean Mist',
    primary: '#38bdf8',
    glow: 'rgba(56, 189, 248, 0.8)',
  },
  amber: {
    name: 'Sunset Gold',
    primary: '#fbbf24',
    glow: 'rgba(251, 191, 36, 0.8)',
  },
  gold: {
    name: 'Ceylon Tea',
    primary: '#c9a227',
    glow: 'rgba(201, 162, 39, 0.8)',
  },
};

export function SettingsPanel({ isOpen, onClose }) {
  const [settings, setSettings] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  // Apply settings to CSS variables
  useEffect(() => {
    const root = document.documentElement;
    
    // Glass panel opacity (0-90 maps to 0.00-0.25 alpha for true transparency)
    // Lower glassOpacity = more transparent panels
    const glassAlpha = (settings.glassOpacity / 90) * 0.25;
    const glassAlphaHover = Math.min(glassAlpha + 0.04, 0.30);
    const borderAlpha = 0.08 + (settings.glassOpacity / 90) * 0.20;
    const borderAlphaHover = borderAlpha + 0.08;
    const blurAmount = 35 - (settings.glassOpacity / 90) * 15; // More blur when more transparent
    
    root.style.setProperty('--glass-alpha', glassAlpha.toFixed(3));
    root.style.setProperty('--glass-alpha-hover', glassAlphaHover.toFixed(3));
    root.style.setProperty('--glass-border-alpha', borderAlpha.toFixed(3));
    root.style.setProperty('--glass-border-alpha-hover', borderAlphaHover.toFixed(3));
    root.style.setProperty('--glass-blur', `${blurAmount}px`);
    
    // Background image opacity (0-100%)
    root.style.setProperty('--bg-opacity', (settings.bgOpacity / 100).toFixed(2));
    
    // Color theme
    const theme = COLOR_THEMES[settings.colorTheme] || COLOR_THEMES.green;
    root.style.setProperty('--glow-primary', theme.primary);
    root.style.setProperty('--glow-green', theme.primary);
    
    // Animations
    if (!settings.animationsEnabled) {
      root.style.setProperty('--animation-duration', '0s');
      document.body.classList.add('reduce-motion');
    } else {
      root.style.setProperty('--animation-duration', '0.3s');
      document.body.classList.remove('reduce-motion');
    }
    
    // Save to localStorage and dispatch custom event for same-window updates
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    window.dispatchEvent(new CustomEvent('teavision-settings-change', { detail: settings }));
  }, [settings]);

  const updateSetting = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const resetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Panel */}
      <div className="glass glow-blue relative w-full max-w-md p-6 animate-fade-up">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div
              className="w-2 h-6 rounded-full"
              style={{
                background: 'var(--glow-blue)',
                boxShadow: '0 0 10px rgba(56,189,248,0.8)',
              }}
            />
            <h2 className="text-sm font-semibold tracking-widest uppercase text-slate-300">
              Settings
            </h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 transition-colors"
          >
            <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>

        {/* Settings content */}
        <div className="flex flex-col gap-6">
          
          {/* Background Opacity */}
          <div className="setting-group flex flex-col gap-3 p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
            <div className="flex items-center justify-between">
              <label className="text-sm text-slate-200 font-medium flex items-center gap-2">
                <span className="text-lg">🍃</span> Background Visibility
              </label>
              <span className="text-sm font-bold text-green-400 font-mono bg-green-500/15 px-2 py-0.5 rounded">{settings.bgOpacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={settings.bgOpacity}
              onChange={(e) => updateSetting('bgOpacity', parseInt(e.target.value))}
              className="settings-slider"
              style={{
                background: `linear-gradient(to right, #22c55e 0%, #22c55e ${settings.bgOpacity}%, rgba(255,255,255,0.1) ${settings.bgOpacity}%, rgba(255,255,255,0.1) 100%)`
              }}
            />
            <p className="text-xs text-slate-400">
              Adjust the tea plantation background (0% = hidden, 100% = visible)
            </p>
          </div>

          {/* Glass Panel Opacity */}
          <div className="setting-group flex flex-col gap-3 p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
            <div className="flex items-center justify-between">
              <label className="text-sm text-slate-200 font-medium">Panel Transparency</label>
              <span className="text-sm font-bold text-sky-400 font-mono bg-sky-500/15 px-2 py-0.5 rounded">{settings.glassOpacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="90"
              step="5"
              value={settings.glassOpacity}
              onChange={(e) => updateSetting('glassOpacity', parseInt(e.target.value))}
              className="settings-slider"
              style={{
                background: `linear-gradient(to right, #38bdf8 0%, #38bdf8 ${(settings.glassOpacity / 90) * 100}%, rgba(255,255,255,0.1) ${(settings.glassOpacity / 90) * 100}%, rgba(255,255,255,0.1) 100%)`
              }}
            />
            <p className="text-xs text-slate-400">
              Glass panel opacity (0% = transparent, 90% = solid)
            </p>
          </div>

          {/* Animations Toggle */}
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm text-slate-300">Animations</label>
              <p className="text-xs text-slate-500">Enable smooth transitions and loading animations</p>
            </div>
            <button
              onClick={() => updateSetting('animationsEnabled', !settings.animationsEnabled)}
              className={`toggle-switch relative w-14 h-7 rounded-full transition-all duration-200 ${
                settings.animationsEnabled 
                  ? 'bg-green-500/25 border-2 border-green-500/60 shadow-[0_0_12px_rgba(34,197,94,0.3)]' 
                  : 'bg-slate-800/60 border-2 border-slate-600/50'
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full transition-all duration-200 ${
                  settings.animationsEnabled 
                    ? 'translate-x-7 bg-gradient-to-br from-green-400 to-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' 
                    : 'translate-x-0 bg-slate-400'
                }`}
              />
            </button>
          </div>

          {/* Color Theme */}
          <div className="flex flex-col gap-3">
            <label className="text-base text-slate-200 font-semibold">Color Theme</label>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(COLOR_THEMES).map(([key, theme]) => (
                <button
                  key={key}
                  onClick={() => updateSetting('colorTheme', key)}
                  className="theme-btn px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200"
                  style={{
                    borderWidth: '2px',
                    borderStyle: 'solid',
                    borderColor: settings.colorTheme === key ? theme.primary : 'rgba(100,116,139,0.3)',
                    background: settings.colorTheme === key 
                      ? `linear-gradient(135deg, ${theme.primary}25, ${theme.primary}10)`
                      : 'rgba(15,23,42,0.5)',
                    color: settings.colorTheme === key ? theme.primary : '#cbd5e1',
                    boxShadow: settings.colorTheme === key 
                      ? `0 0 20px ${theme.primary}40, inset 0 1px 0 rgba(255,255,255,0.1)` 
                      : 'none',
                    transform: settings.colorTheme === key ? 'scale(1.02)' : 'scale(1)',
                  }}
                >
                  {theme.name}
                </button>
              ))}
            </div>
          </div>

          {/* Show Cost Help */}
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm text-slate-300">Show Cost Explanations</label>
              <p className="text-xs text-slate-500">Display help icons for FP/FN cost metrics</p>
            </div>
            <button
              onClick={() => updateSetting('showCostHelp', !settings.showCostHelp)}
              className={`toggle-switch relative w-14 h-7 rounded-full transition-all duration-200 ${
                settings.showCostHelp 
                  ? 'bg-green-500/25 border-2 border-green-500/60 shadow-[0_0_12px_rgba(34,197,94,0.3)]' 
                  : 'bg-slate-800/60 border-2 border-slate-600/50'
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full transition-all duration-200 ${
                  settings.showCostHelp 
                    ? 'translate-x-7 bg-gradient-to-br from-green-400 to-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' 
                    : 'translate-x-0 bg-slate-400'
                }`}
              />
            </button>
          </div>

          {/* Reset Button */}
          <button
            onClick={resetSettings}
            className="mt-4 w-full px-4 py-3 rounded-xl text-sm text-slate-400 font-medium bg-slate-800/50 border border-slate-600/40 hover:border-slate-500/60 hover:text-slate-200 hover:bg-slate-700/50 transition-all duration-200 active:scale-[0.98]"
          >
            Reset to Defaults
          </button>
        </div>
      </div>
    </div>
  );
}

// Hook to access settings from other components
export function useSettings() {
  const [settings, setSettings] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  useEffect(() => {
    // Listen for custom settings change event (same window)
    const handleSettingsChange = (e) => {
      if (e.detail) {
        setSettings(e.detail);
      }
    };
    
    // Listen for storage events (other windows/tabs)
    const handleStorage = () => {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(saved) });
      }
    };
    
    window.addEventListener('teavision-settings-change', handleSettingsChange);
    window.addEventListener('storage', handleStorage);
    
    return () => {
      window.removeEventListener('teavision-settings-change', handleSettingsChange);
      window.removeEventListener('storage', handleStorage);
    };
  }, []);

  return settings;
}
