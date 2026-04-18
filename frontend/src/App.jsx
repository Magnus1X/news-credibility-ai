import React, { useState } from 'react';
import axios from 'axios';
import {
  Search, BrainCircuit, CheckCircle2, CircleDashed, Loader2,
  Info, Zap, ShieldCheck, AlertTriangle, BookOpen, Download,
  ChevronDown, ChevronUp, FileText, BarChart2, Shield, Link2
} from 'lucide-react';
import './index.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';

const STEPS_M1 = [
  'Analyzing text input...',
  'Extracting linguistic features...',
  'Running logistic regression model...',
  'Synthesizing credibility score...',
];

const STEPS_M2 = [
  'Step 1 — ML model prediction...',
  'Step 2 — Risk signal analysis...',
  'Step 3 — Retrieving fact-check sources...',
  'Step 4 — Evaluating uncertainty...',
  'Step 5 — Generating credibility report...',
];

function resolveVerdict(label, uncertain) {
  if (uncertain) return { display: 'Uncertain', cls: 'uncertain' };
  if (label === 'Real News') return { display: 'Real News', cls: 'real' };
  return { display: 'Fake News', cls: 'fake' };
}

function ConfidenceBar({ value, color }) {
  return (
    <div className="conf-bar-track">
      <div className="conf-bar-fill" style={{ width: `${value}%`, background: color }} />
    </div>
  );
}

function RiskBadge({ score }) {
  const level = score >= 70 ? 'high' : score >= 35 ? 'medium' : 'low';
  const labels = { high: 'High Risk', medium: 'Medium Risk', low: 'Low Risk' };
  return <span className={`risk-badge risk-${level}`}>{labels[level]} · {score}/100</span>;
}

function Section({ icon: Icon, title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="report-section">
      <button className="section-header" onClick={() => setOpen(o => !o)}>
        <span className="section-title">
          <Icon size={15} className="section-icon" />
          {title}
        </span>
        {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>
      {open && <div className="section-body">{children}</div>}
    </div>
  );
}

function ReliabilityWarning({ wordCount, inputSource }) {
  return (
    <div className="banner banner-warning">
      <AlertTriangle size={15} className="banner-icon" />
      <div>
        <strong>Result may be unreliable — only {wordCount} words detected.</strong>
        <br />
        This model was trained on full news articles (200+ words). Short snippets lack sufficient signal for accurate classification.
        {inputSource === 'text' && (
          <><br /><span className="banner-tip">Tip: Paste just the URL — the system will auto-scrape the full article.</span></>
        )}
      </div>
    </div>
  );
}

function UncertainWarning({ confidence }) {
  return (
    <div className="banner banner-info">
      <Info size={15} className="banner-icon" />
      <div>
        <strong>Low confidence ({confidence}%) — result is uncertain.</strong>
        <br />
        This model was trained on US wire-service news (Reuters / AP). Content from other domains such as
        Indian politics, regional news, or opinion pieces may not be classified accurately.
        Treat this as a signal, not a definitive verdict.
      </div>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState('m2');
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeStepIndex, setActiveStepIndex] = useState(-1);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);

  const steps = mode === 'm1' ? STEPS_M1 : STEPS_M2;

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    setIsLoading(true);
    setResult(null);
    setError(null);
    setActiveStepIndex(0);

    const stepInterval = setInterval(() => {
      setActiveStepIndex(prev => {
        if (prev < steps.length - 1) return prev + 1;
        clearInterval(stepInterval);
        return prev;
      });
    }, mode === 'm1' ? 800 : 1000);

    try {
      const isUrl = inputValue.trim().startsWith('http');
      const payload = isUrl ? { url: inputValue.trim() } : { text: inputValue.trim() };
      const endpoint = mode === 'm1' ? `${API_BASE}/predict` : `${API_BASE}/analyze`;
      const response = await axios.post(endpoint, payload);

      setTimeout(() => {
        clearInterval(stepInterval);
        setActiveStepIndex(steps.length);
        setResult(response.data);
        setIsLoading(false);
      }, steps.length * (mode === 'm1' ? 800 : 1000) + 400);

    } catch (err) {
      clearInterval(stepInterval);
      setIsLoading(false);
      setActiveStepIndex(-1);
      setError(err.response?.data?.detail || 'An error occurred while analyzing the text.');
    }
  };

  const handleDownloadPdf = async () => {
    setPdfLoading(true);
    try {
      const isUrl = inputValue.trim().startsWith('http');
      const payload = isUrl ? { url: inputValue.trim() } : { text: inputValue.trim() };
      const response = await axios.post(`${API_BASE}/analyze/pdf`, payload, { responseType: 'blob' });
      const blobUrl = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = 'credibility_report.pdf';
      a.click();
      window.URL.revokeObjectURL(blobUrl);
    } catch {
      setError('PDF generation failed. Make sure the backend server is running.');
    } finally {
      setPdfLoading(false);
    }
  };

  const reset = () => { setResult(null); setInputValue(''); setError(null); };

  const m1Verdict = result?.prediction ? resolveVerdict(result.prediction, result.uncertain) : null;
  const m2Verdict = result?.prediction?.label ? resolveVerdict(result.prediction.label, result.uncertain) : null;
  const confidence = mode === 'm1' ? result?.confidence_score : result?.prediction?.confidence;

  return (
    <div className="app-container">

      {/* Header */}
      <div className="logo-container" onClick={reset}>
        <BrainCircuit className="logo-icon" />
        <div className="logo-text">
          <h1>News Credibility AI</h1>
          <p>AI Research Assistant</p>
        </div>
      </div>

      {/* Mode Toggle */}
      <div className="mode-toggle">
        <button className={`mode-btn ${mode === 'm1' ? 'active' : ''}`} onClick={() => { setMode('m1'); reset(); }}>
          <Zap size={14} /> Quick Analysis
        </button>
        <button className={`mode-btn ${mode === 'm2' ? 'active' : ''}`} onClick={() => { setMode('m2'); reset(); }}>
          <ShieldCheck size={14} /> Deep Analysis
        </button>
      </div>
      <p className="mode-desc">
        {mode === 'm1'
          ? 'Fast ML prediction — label + confidence score'
          : '5-step agentic pipeline — risk analysis, RAG retrieval, structured report + PDF export'}
      </p>

      {/* Search */}
      <form className="search-container" onSubmit={handleSearch}>
        <Search className="search-icon" size={18} />
        <input
          type="text"
          className="search-input"
          placeholder="Paste full article text, or a URL to auto-scrape..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" className="search-button" disabled={!inputValue.trim() || isLoading}>
          {isLoading ? <Loader2 className="animate-spin" size={18} /> : 'Analyze'}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="error-card">
          <AlertTriangle size={15} /> {error}
        </div>
      )}

      {/* Steps */}
      {isLoading && (
        <div className="steps-container">
          {steps.map((stepText, idx) => {
            const isCompleted = idx < activeStepIndex;
            const isActive = idx === activeStepIndex;
            return (
              <div key={idx} className={`step-item ${isCompleted ? 'completed' : isActive ? 'active' : 'pending'}`}>
                {isCompleted ? <CheckCircle2 className="step-icon" size={18} />
                  : isActive ? <Loader2 className="step-icon animate-spin" size={18} />
                  : <CircleDashed className="step-icon" size={18} />}
                <span>{stepText}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* ── MILESTONE 1 RESULT ── */}
      {result && !isLoading && mode === 'm1' && (
        <div className="results-container">
          {result.reliable === false && (
            <ReliabilityWarning wordCount={result.word_count} inputSource={result.input_source} />
          )}
          {result.uncertain && result.reliable !== false && (
            <UncertainWarning confidence={result.confidence_score} />
          )}
          <div className="result-card">
            <div className="result-header">
              <span className="result-title">Credibility Result</span>
              <span className="confidence-badge">{confidence}% Confidence</span>
            </div>
            <div className="result-content">
              <span className="result-label-pre">This article is predicted as:</span>
              <div className={`prediction-text ${m1Verdict.cls}`}>{m1Verdict.display}</div>
              <p className="result-message">{result.message}</p>
            </div>
          </div>
          <div className="section-group-label">
            <Info size={14} /> Analysis Details
          </div>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-title">Input Source</div>
              <div className="metric-value">{result.input_source === 'url' ? 'Auto-scraped from URL' : 'Direct Text Input'}</div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Word Count</div>
              <div className="metric-value">{result.word_count} words · {result.text_length?.toLocaleString()} characters</div>
            </div>
          </div>
        </div>
      )}

      {/* ── MILESTONE 2 RESULT ── */}
      {result && !isLoading && mode === 'm2' && (
        <div className="results-container">
          {result.reliable === false && (
            <ReliabilityWarning wordCount={result.word_count} inputSource={result.input_source} />
          )}
          {result.uncertain && result.reliable !== false && (
            <UncertainWarning confidence={result.prediction.confidence} />
          )}

          {/* Verdict card */}
          <div className="result-card">
            <div className="result-header">
              <span className="result-title">Credibility Verdict</span>
              <div className="result-badges">
                <RiskBadge score={result.risk_analysis.risk_score} />
                <span className="confidence-badge">{confidence}% Confidence</span>
              </div>
            </div>
            <div className="result-content">
              <div className={`prediction-text ${m2Verdict.cls}`}>{m2Verdict.display}</div>
              <div className="prob-bars">
                <div className="prob-row">
                  <span>Real</span>
                  <ConfidenceBar value={result.prediction.real_probability} color="var(--success)" />
                  <span>{result.prediction.real_probability}%</span>
                </div>
                <div className="prob-row">
                  <span>Fake</span>
                  <ConfidenceBar value={result.prediction.fake_probability} color="var(--danger)" />
                  <span>{result.prediction.fake_probability}%</span>
                </div>
              </div>
              <div className="top-features">
                <span className="features-label">Top signals</span>
                {result.prediction.top_features.map(f => (
                  <span key={f} className="feature-chip">{f}</span>
                ))}
              </div>
            </div>
          </div>

          <Section icon={FileText} title="Summary">
            <p className="report-text">{result.report.summary}</p>
          </Section>

          <div className="two-col">
            <Section icon={AlertTriangle} title="Risk Factors">
              {result.risk_analysis.risk_factors.length === 0
                ? <p className="report-text muted">No risk factors detected.</p>
                : result.risk_analysis.risk_factors.map((r, i) => (
                    <div key={i} className="list-item danger">
                      <AlertTriangle size={13} /> {r}
                    </div>
                  ))}
            </Section>
            <Section icon={Shield} title="Credibility Indicators">
              {result.risk_analysis.credibility_indicators.length === 0
                ? <p className="report-text muted">No credibility indicators found.</p>
                : result.risk_analysis.credibility_indicators.map((c, i) => (
                    <div key={i} className="list-item success">
                      <CheckCircle2 size={13} /> {c}
                    </div>
                  ))}
            </Section>
          </div>

          <Section icon={Link2} title="Cross-Source Verification">
            <p className="report-text">{result.report.cross_source_verification}</p>
            {result.retrieved_sources.length > 0 && (
              <div className="sources-list">
                {result.retrieved_sources.map((s, i) => (
                  <div key={i} className="source-chip">
                    <BookOpen size={12} />
                    <span className="source-name">{s.source}</span>
                    <span className="source-title">{s.title}</span>
                    <span className="source-score">{(s.relevance * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}
          </Section>

          <Section icon={BarChart2} title="Confidence Assessment">
            <p className="report-text">{result.report.confidence_assessment}</p>
            <div className="meta-row">
              <span className="meta-chip">Tier <b>{result.prediction.confidence_tier}</b></span>
              <span className="meta-chip">Words <b>{result.word_count}</b></span>
              <span className="meta-chip">Steps <b>{result.pipeline_steps.length}</b></span>
              <span className="meta-chip">Engine <b>{result.used_llm ? 'Mistral-7B' : 'Rule-based'}</b></span>
            </div>
          </Section>

          <div className="disclaimer-card">
            <Info size={13} className="disclaimer-icon" />
            <span>{result.report.disclaimer}</span>
          </div>

          <button className="pdf-btn" onClick={handleDownloadPdf} disabled={pdfLoading}>
            {pdfLoading
              ? <><Loader2 size={15} className="animate-spin" /> Generating PDF...</>
              : <><Download size={15} /> Download PDF Report</>}
          </button>
        </div>
      )}
    </div>
  );
}
