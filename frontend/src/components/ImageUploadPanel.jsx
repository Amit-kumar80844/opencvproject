/**
 * ImageUploadPanel.jsx — Premium Glassmorphism Upload Interface
 * ==============================================================
 * Sri Lankan tea-themed file upload with drag-and-drop.
 * Features large preview, elegant animations, and professional styling.
 */

import { useState, useRef, useCallback } from 'react';

const VALID_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];

export function ImageUploadPanel({ onSubmit, isLoading, onReset, hasResult }) {
  const [dragOver, setDragOver]     = useState(false);
  const [preview, setPreview]       = useState(null);
  const [fileName, setFileName]     = useState('');
  const [fileError, setFileError]   = useState('');
  const fileRef                     = useRef(null);
  const fileObjRef                  = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!VALID_TYPES.includes(file.type)) {
      setFileError('Please upload a JPEG or PNG image.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setFileError('Image must be smaller than 10 MB.');
      return;
    }
    setFileError('');
    setFileName(file.name);
    fileObjRef.current = file;
    const url = URL.createObjectURL(file);
    setPreview(url);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files?.[0]);
  }, [handleFile]);

  const onInputChange = useCallback((e) => {
    handleFile(e.target.files?.[0]);
  }, [handleFile]);

  const handleAnalyse = () => {
    if (fileObjRef.current) onSubmit(fileObjRef.current);
  };

  const handleReset = () => {
    setPreview(null);
    setFileName('');
    setFileError('');
    fileObjRef.current = null;
    if (fileRef.current) fileRef.current.value = '';
    onReset();
  };

  return (
    <div className="glass glow-blue flex flex-col gap-3 p-4 relative z-10 upload-panel h-full">
      {/* Header with Sri Lankan tea accent */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="header-accent w-1.5 h-7 rounded-full" style={{ background: 'linear-gradient(180deg, #38bdf8, #22c55e)', boxShadow: '0 0 12px rgba(56,189,248,0.7)' }} />
        <div className="flex-1">
          <h2 className="text-sm font-semibold tracking-widest uppercase text-slate-200">
            Leaf Image Upload
          </h2>
          <p className="text-xs text-slate-300 mt-0.5">Sri Lankan Tea Disease Analysis</p>
        </div>
        {/* Ceylon Tea Badge */}
        <div className="ceylon-badge px-2.5 py-1 rounded-full text-[0.6rem] font-bold uppercase tracking-wider"
          style={{
            background: 'linear-gradient(135deg, rgba(201,162,39,0.2), rgba(34,197,94,0.15))',
            border: '1px solid rgba(201,162,39,0.4)',
            color: '#c9a227',
          }}>
          🇱🇰 Ceylon
        </div>
      </div>

      {/* Drop zone - fills available space */}
      <div
        className={`upload-drop-zone flex-1 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${dragOver ? 'drag-active' : ''} ${preview ? 'has-preview' : ''}`}
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && fileRef.current?.click()}
      >
        {preview ? (
          <div className="preview-container relative w-full h-full overflow-hidden rounded-xl">
            <img
              src={preview}
              alt="Selected leaf"
              className="preview-image w-full h-full object-contain rounded-xl"
            />
            <div className="preview-overlay absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity duration-300"
              style={{ background: 'rgba(0,0,0,0.4)' }}>
              <span className="text-base text-white font-medium bg-black/40 px-4 py-2 rounded-lg backdrop-blur-sm">Click to change</span>
            </div>
            {/* Image frame glow */}
            <div className="absolute inset-0 rounded-xl pointer-events-none"
              style={{
                boxShadow: 'inset 0 0 50px rgba(56,189,248,0.2), 0 0 30px rgba(34,197,94,0.1)',
                border: '2px solid rgba(56,189,248,0.4)',
              }}
            />
          </div>
        ) : (
          <div className="upload-placeholder flex flex-col items-center gap-4 py-6 text-center">
            {/* Animated leaf icon */}
            <div className="leaf-icon-wrapper relative">
              <svg width="72" height="72" viewBox="0 0 72 72" className="leaf-icon">
                <defs>
                  <linearGradient id="leafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#38bdf8" />
                    <stop offset="100%" stopColor="#22c55e" />
                  </linearGradient>
                </defs>
                <circle cx="36" cy="36" r="34" fill="rgba(56,189,248,0.08)" stroke="url(#leafGrad)" strokeWidth="1" strokeOpacity="0.3" />
                <path d="M36 14 C20 20 14 36 20 50 C26 60 34 62 40 60 C50 56 58 44 54 26 C50 16 36 14 36 14Z" 
                  fill="none" stroke="url(#leafGrad)" strokeWidth="2" strokeLinecap="round" />
                <path d="M36 14 L34 60" stroke="url(#leafGrad)" strokeWidth="1.2" strokeOpacity="0.6" />
                <path d="M36 26 L28 36" stroke="url(#leafGrad)" strokeWidth="1" strokeOpacity="0.4" />
                <path d="M36 34 L26 46" stroke="url(#leafGrad)" strokeWidth="1" strokeOpacity="0.4" />
                <path d="M36 26 L46 34" stroke="url(#leafGrad)" strokeWidth="1" strokeOpacity="0.4" />
                <path d="M36 36 L46 44" stroke="url(#leafGrad)" strokeWidth="1" strokeOpacity="0.4" />
              </svg>
              {/* Animated ring */}
              <div className="absolute inset-0 animate-ping-slow rounded-full border border-sky-400/30" />
            </div>
            
            <div>
              <p className="text-white text-base font-medium mb-1">
                Drop a tea leaf image here
              </p>
              <p className="text-slate-300 text-sm">
                or <span className="text-sky-400 hover:text-sky-300 cursor-pointer font-medium">browse files</span>
              </p>
            </div>
            
            <div className="flex items-center gap-3 text-xs text-slate-300">
              <span className="px-2 py-1 rounded-md bg-white/10 border border-white/20">JPEG</span>
              <span className="px-2 py-1 rounded-md bg-white/10 border border-white/20">PNG</span>
              <span className="px-2 py-1 rounded-md bg-white/10 border border-white/20">WebP</span>
              <span className="text-slate-300">• max 10 MB</span>
            </div>
          </div>
        )}
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onInputChange}
        />
      </div>

      {fileError && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/30">
          <svg className="w-4 h-4 text-red-400" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <p className="text-xs text-red-400">{fileError}</p>
        </div>
      )}

      {fileName && !fileError && (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-sky-500/10 border border-sky-500/25 shrink-0">
          <svg className="w-3.5 h-3.5 text-sky-400" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
          </svg>
          <p className="text-xs text-slate-200 font-mono truncate flex-1">{fileName}</p>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 shrink-0">
        <button
          className="primary-btn flex-1 flex items-center justify-center gap-2 py-2.5"
          onClick={handleAnalyse}
          disabled={!preview || isLoading}
        >
          {isLoading ? (
            <>
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeOpacity="0.3" />
                <path d="M12 2 A10 10 0 0 1 22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <span>Analysing...</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
              </svg>
              <span>Analyse Leaf</span>
            </>
          )}
        </button>

        {(preview || hasResult) && (
          <button
            className="reset-btn flex items-center justify-center gap-2"
            onClick={handleReset}
            disabled={isLoading}
          >
            <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
            </svg>
            Reset
          </button>
        )}
      </div>
    </div>
  );
}
