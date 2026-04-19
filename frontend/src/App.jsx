import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  ScanSearch, Search, CheckCircle2, CircleDashed, Loader2,
  Info, Zap, ShieldCheck, AlertTriangle, BookOpen, Download,
  ChevronDown, ChevronUp, FileText, BarChart2, Shield, Link2,
  Moon, Sun, Copy, Check, Clock, Trash2, Type, Globe,
  TrendingUp, Activity, Cpu, Database, FlaskConical, Eye
} from 'lucide-react';
import './index.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';

const STEPS_M1 = [
  'Analyzing text input',
  'Extracting linguistic features',
  'Running logistic regression model',
  'Synthesizing credibility score',
];
const STEPS_M2 = [
  'ML model prediction',
  'Risk signal analysis',
  'Retrieving fact-check sources',
  'Evaluating uncertainty',
  'Generating credibility report',
];

function resolveVerdict(label, uncertain) {
  if (uncertain) return { display: 'Uncertain', cls: 'uncertain' };
  if (label === 'Real News') return { display: 'Real News', cls: 'real' };
  return { display: 'Fake News', cls: 'fake' };
}

function ProbBar({ value, color }) {
  return (
    <div className="bar-track">
      <div className="bar-fill" style={{ width: `${value}%`, background: color }} />
    </div>
  );
}

function RiskTag({ score }) {
  const level = score >= 70 ? 'high' : score >= 35 ? 'medium' : 'low';
  const labels = { high: 'High Risk', medium: 'Medium Risk', low: 'Low Risk' };
  return <span className={`risk-tag risk-${level}`}>{labels[level]} · {score}/100</span>;
}

function Section({ icon: Icon, title, children, defaultOpen = true, badge }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="section">
      <button className="section-toggle" onClick={() => setOpen(o => !o)}>
        <span className="section-toggle-left">
          <Icon size={13} />
          {title}
          {badge !== undefined && <span className="section-badge">{badge}</span>}
        </span>
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>
      {open && <div className="section-content">{children}</div>}
    </div>
  );
}

function CopyBtn({ text }) {
  const [ok, setOk] = useState(false);
  const handle = () => {
    navigator.clipboard.writeText(text);
    setOk(true);
    setTimeout(() => setOk(false), 2000);
  };
  return (
    <button className={`copy-btn ${ok ? 'ok' : ''}`} onClick={handle}>
      {ok ? <><Check size={11} />Copied</> : <><Copy size={11} />Copy</>}
    </button>
  );
}

function WarnBanner({ wordCount, inputSource }) {
  return (
    <div className="banner warn">
      <AlertTriangle size={14} className="banner-icon" />
      <div>
        <strong>Only {wordCount} words — result may be unreliable.</strong>
        <span> Trained on full articles (200+ words).</span>
        {inputSource === 'text' && <span className="banner-tip"> Try pasting the URL instead.</span>}
      </div>
    </div>
  );
}

function UncertainBanner({ confidence }) {
  return (
    <div className="banner info">
      <Info size={14} className="banner-icon" />
      <div>
        <strong>Low confidence ({confidence}%) — treat as a signal, not a verdict.</strong>
        <span> Trained on US wire-service news. Regional or opinion content may not classify accurately.</span>
      </div>
    </div>
  );
}

/* ── Gauge bar ── */
function GaugeBar({ value, max = 100, color }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="gauge-track">
      <div className="gauge-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

/* ── Stat grid item ── */
function Stat({ label, value, sub }) {
  return (
    <div className="stat-item">
      <p className="stat-label">{label}</p>
      <p className="stat-value">{value}</p>
      {sub && <p className="stat-sub">{sub}</p>}
    </div>
  );
}

export default function App() {
  const [dark, setDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);
  const [mode, setMode] = useState('m2');
  const [tab, setTab] = useState('text');
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(-1);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem('nca_h') || '[]'); } catch { return []; }
  });
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  }, [dark]);

  const steps = mode === 'm1' ? STEPS_M1 : STEPS_M2;
  const delay = mode === 'm1' ? 800 : 1000;

  const analyze = async (e) => {
    e?.preventDefault();
    const val = input.trim();
    if (!val) return;
    setLoading(true); setResult(null); setError(null); setStep(0);
    const iv = setInterval(() => setStep(p => p < steps.length - 1 ? p + 1 : p), delay);
    try {
      const isUrl = tab === 'url' || val.startsWith('http');
      const payload = isUrl ? { url: val } : { text: val };
      const endpoint = mode === 'm1' ? `${API_BASE}/predict` : `${API_BASE}/analyze`;
      const { data } = await axios.post(endpoint, payload);
      setTimeout(() => {
        clearInterval(iv);
        setStep(steps.length);
        setResult(data);
        setLoading(false);
        const label = mode === 'm1' ? data.prediction : data.prediction?.label;
        const conf = mode === 'm1' ? data.confidence_score : data.prediction?.confidence;
        const entry = {
          id: Date.now(),
          snip: val.slice(0, 55) + (val.length > 55 ? '…' : ''),
          label, conf,
          cls: resolveVerdict(label, data.uncertain).cls,
        };
        setHistory(prev => {
          const next = [entry, ...prev].slice(0, 5);
          localStorage.setItem('nca_h', JSON.stringify(next));
          return next;
        });
      }, steps.length * delay + 400);
    } catch (err) {
      clearInterval(iv);
      setLoading(false); setStep(-1);
      setError(err.response?.data?.detail || 'Something went wrong. Is the backend running?');
    }
  };

  const downloadPdf = async () => {
    setPdfLoading(true);
    try {
      const val = input.trim();
      const isUrl = tab === 'url' || val.startsWith('http');
      const payload = isUrl ? { url: val } : { text: val };
      const res = await axios.post(`${API_BASE}/analyze/pdf`, payload, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }));
      Object.assign(document.createElement('a'), { href: url, download: 'credibility_report.pdf' }).click();
      window.URL.revokeObjectURL(url);
    } catch {
      setError('PDF generation failed.');
    } finally {
      setPdfLoading(false);
    }
  };

  const reset = () => { setResult(null); setInput(''); setError(null); setStep(-1); };
  const wordCount = input.trim() ? input.trim().split(/\s+/).filter(Boolean).length : 0;
  const m1v = result && mode === 'm1' ? resolveVerdict(result.prediction, result.uncertain) : null;
  const m2v = result?.prediction?.label ? resolveVerdict(result.prediction.label, result.uncertain) : null;
  const conf = mode === 'm1' ? result?.confidence_score : result?.prediction?.confidence;

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <button className="brand" onClick={reset}>
            <div className="brand-icon"><ScanSearch size={15} strokeWidth={2.2} /></div>
            <span className="brand-name">Credibility AI</span>
          </button>
          <div className="header-actions">
            {history.length > 0 && (
              <button className="icon-btn" onClick={() => setShowHistory(s => !s)} title="History">
                <Clock size={15} />
              </button>
            )}
            <button className="icon-btn" onClick={() => setDark(d => !d)} title="Toggle theme">
              {dark ? <Sun size={15} /> : <Moon size={15} />}
            </button>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main">

        {/* Hero */}
        <div className="hero">
          <p className="hero-label">AI-powered fact analysis</p>
          <h1>Is this news real or fake?</h1>
          <p className="hero-sub">Paste article text or a URL — get a full credibility breakdown in seconds.</p>
        </div>

        {/* Mode */}
        <div className="mode-row">
          <div className="mode-pills">
            <button className={`mode-pill ${mode === 'm1' ? 'active' : ''}`} onClick={() => { setMode('m1'); reset(); }}>
              <Zap size={12} />Quick
            </button>
            <button className={`mode-pill ${mode === 'm2' ? 'active' : ''}`} onClick={() => { setMode('m2'); reset(); }}>
              <ShieldCheck size={12} />Deep Analysis
            </button>
          </div>
          <p className="mode-desc">
            {mode === 'm1' ? 'Fast ML prediction — label + confidence' : '5-step pipeline — risk signals, RAG retrieval, full structured report'}
          </p>
        </div>

        {/* History */}
        {showHistory && history.length > 0 && (
          <div className="history-box">
            <div className="history-box-head">
              <span>Recent</span>
              <button onClick={() => { setHistory([]); localStorage.removeItem('nca_h'); }}>Clear</button>
            </div>
            {history.map(h => (
              <div key={h.id} className="history-item" onClick={() => { setInput(h.snip.replace('…', '')); setShowHistory(false); setResult(null); }}>
                <span className={`h-dot ${h.cls}`} />
                <span className="h-snip">{h.snip}</span>
                <span className={`h-verdict ${h.cls}`}>{h.label}</span>
                <span className="h-conf">{h.conf}%</span>
              </div>
            ))}
          </div>
        )}

        {/* Input */}
        <form className="input-box" onSubmit={analyze}>
          <div className="input-tabs">
            <button type="button" className={`itab ${tab === 'text' ? 'active' : ''}`} onClick={() => setTab('text')}>
              <Type size={11} />Article text
            </button>
            <button type="button" className={`itab ${tab === 'url' ? 'active' : ''}`} onClick={() => setTab('url')}>
              <Globe size={11} />URL
            </button>
          </div>
          <div className="input-area">
            {tab === 'url'
              ? <input className="text-field" type="text" placeholder="https://example.com/article" value={input} onChange={e => setInput(e.target.value)} disabled={loading} />
              : <textarea className="text-field area" placeholder="Paste the full article text here…" value={input} onChange={e => setInput(e.target.value)} disabled={loading} />
            }
          </div>
          <div className="input-foot">
            <span className="wc">{tab === 'text' && input.trim() ? `${wordCount} words · ${input.length} chars` : ''}</span>
            <button type="submit" className="analyze-btn" disabled={!input.trim() || loading}>
              {loading ? <><Loader2 size={14} className="spin" />Analyzing…</> : <><Search size={14} />Analyze</>}
            </button>
          </div>
        </form>

        {/* Error */}
        {error && <div className="error-msg"><AlertTriangle size={14} />{error}</div>}

        {/* Steps */}
        {loading && (
          <div className="steps-box">
            <p className="steps-label">Analyzing</p>
            {steps.map((text, i) => {
              const done = i < step, active = i === step;
              return (
                <div key={i} className={`step ${done ? 'done' : active ? 'active' : 'idle'}`}>
                  {done ? <CheckCircle2 size={14} /> : active ? <Loader2 size={14} className="spin" /> : <CircleDashed size={14} />}
                  <span>{text}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* ── M1 Result ── */}
        {result && !loading && mode === 'm1' && (
          <div className="result-stack">
            {result.reliable === false && <WarnBanner wordCount={result.word_count} inputSource={result.input_source} />}
            {result.uncertain && result.reliable !== false && <UncertainBanner confidence={result.confidence_score} />}

            {/* Verdict */}
            <div className={`verdict-card ${m1v.cls}`}>
              <div className="vc-top">
                <div className="vc-left">
                  <p className="vc-label">Credibility result</p>
                  <p className={`vc-verdict ${m1v.cls}`}>{m1v.display}</p>
                  <p className="vc-msg">{result.message}</p>
                </div>
                <div className="vc-right">
                  <div className={`conf-circle ${m1v.cls}`}>
                    <span className="conf-num">{conf}%</span>
                    <span className="conf-word">confidence</span>
                  </div>
                  <CopyBtn text={`${m1v.display} — ${conf}% confidence`} />
                </div>
              </div>
            </div>

            {/* Stats */}
            <div className="stat-grid">
              <Stat label="Input source" value={result.input_source === 'url' ? 'URL (scraped)' : 'Direct text'} />
              <Stat label="Word count" value={`${result.word_count} words`} sub={`${result.text_length?.toLocaleString()} characters`} />
              <Stat label="Reliability" value={result.reliable ? 'Sufficient' : 'Low'} sub={result.reliable ? '≥ 80 words' : '< 80 words'} />
              <Stat label="Certainty" value={result.uncertain ? 'Uncertain' : 'Confident'} sub={`${conf}% model score`} />
            </div>
          </div>
        )}

        {/* ── M2 Result ── */}
        {result && !loading && mode === 'm2' && (() => {
          const ra = result.risk_analysis;
          const pred = result.prediction;
          return (
            <div className="result-stack">
              {result.reliable === false && <WarnBanner wordCount={result.word_count} inputSource={result.input_source} />}
              {result.uncertain && result.reliable !== false && <UncertainBanner confidence={pred.confidence} />}

              {/* ── 1. Verdict ── */}
              <div className={`verdict-card ${m2v.cls}`}>
                <div className="vc-top">
                  <div className="vc-left">
                    <p className="vc-label">Credibility verdict</p>
                    <p className={`vc-verdict ${m2v.cls}`}>{m2v.display}</p>
                    <div className="vc-badges">
                      <RiskTag score={ra.risk_score} />
                      <span className={`tier-tag tier-${pred.confidence_tier}`}>{pred.confidence_tier} confidence</span>
                      <CopyBtn text={`${m2v.display} — ${conf}% confidence`} />
                    </div>
                  </div>
                  <div className="vc-right">
                    <div className={`conf-circle ${m2v.cls}`}>
                      <span className="conf-num">{conf}%</span>
                      <span className="conf-word">confidence</span>
                    </div>
                  </div>
                </div>

                {/* Probability bars */}
                <div className="vc-body">
                  <p className="vc-body-label">Model probability breakdown</p>
                  <div className="prob-block">
                    <div className="prob-row">
                      <span className="prob-lbl">Real</span>
                      <ProbBar value={pred.real_probability} color="var(--green)" />
                      <span className="prob-pct">{pred.real_probability}%</span>
                    </div>
                    <div className="prob-row">
                      <span className="prob-lbl">Fake</span>
                      <ProbBar value={pred.fake_probability} color="var(--red)" />
                      <span className="prob-pct">{pred.fake_probability}%</span>
                    </div>
                  </div>
                  <div className="signals">
                    <span className="signals-label">Top TF-IDF signals</span>
                    {pred.top_features.map(f => <span key={f} className="signal">{f}</span>)}
                  </div>
                </div>
              </div>

              {/* ── 2. Summary ── */}
              <Section icon={FileText} title="Summary">
                <p className="body-text">{result.report.summary}</p>
              </Section>

              {/* ── 3. Risk & Credibility ── */}
              <div className="two-col">
                <Section icon={AlertTriangle} title="Risk factors" badge={ra.risk_factors.length}>
                  {ra.risk_factors.length === 0
                    ? <p className="body-text muted">No risk factors detected.</p>
                    : ra.risk_factors.map((r, i) => {
                        const tag = r.match(/^\[(.+?)\]/)?.[1];
                        const text = r.replace(/^\[.+?\]\s*/, '');
                        return (
                          <div key={i} className="insight-row bad">
                            <AlertTriangle size={12} />
                            <div>
                              {tag && <span className="insight-tag">{tag}</span>}
                              <span>{text}</span>
                            </div>
                          </div>
                        );
                      })}
                </Section>
                <Section icon={Shield} title="Credibility indicators" badge={ra.credibility_indicators.length}>
                  {ra.credibility_indicators.length === 0
                    ? <p className="body-text muted">None found.</p>
                    : ra.credibility_indicators.map((c, i) => {
                        const tag = c.match(/^\[(.+?)\]/)?.[1];
                        const text = c.replace(/^\[.+?\]\s*/, '');
                        return (
                          <div key={i} className="insight-row good">
                            <CheckCircle2 size={12} />
                            <div>
                              {tag && <span className="insight-tag">{tag}</span>}
                              <span>{text}</span>
                            </div>
                          </div>
                        );
                      })}
                </Section>
              </div>

              {/* ── 4. Heuristic signals ── */}
              <Section icon={Activity} title="Heuristic signal breakdown">
                <div className="signal-grid">
                  <div className="signal-meter">
                    <div className="signal-meter-head">
                      <span>Clickbait patterns</span>
                      <span className={ra.clickbait_hits > 0 ? 'meter-val bad' : 'meter-val good'}>{ra.clickbait_hits} hit{ra.clickbait_hits !== 1 ? 's' : ''}</span>
                    </div>
                    <GaugeBar value={ra.clickbait_hits} max={5} color={ra.clickbait_hits > 0 ? 'var(--red)' : 'var(--green)'} />
                  </div>
                  <div className="signal-meter">
                    <div className="signal-meter-head">
                      <span>Emotional language</span>
                      <span className={ra.emotional_hits >= 3 ? 'meter-val bad' : 'meter-val good'}>{ra.emotional_hits} word{ra.emotional_hits !== 1 ? 's' : ''}</span>
                    </div>
                    <GaugeBar value={ra.emotional_hits} max={10} color={ra.emotional_hits >= 3 ? 'var(--amber)' : 'var(--green)'} />
                  </div>
                  <div className="signal-meter">
                    <div className="signal-meter-head">
                      <span>Source citations</span>
                      <span className={ra.credibility_hits > 0 ? 'meter-val good' : 'meter-val bad'}>{ra.credibility_hits} found</span>
                    </div>
                    <GaugeBar value={ra.credibility_hits} max={5} color={ra.credibility_hits > 0 ? 'var(--green)' : 'var(--red)'} />
                  </div>
                  <div className="signal-meter">
                    <div className="signal-meter-head">
                      <span>ALL-CAPS ratio</span>
                      <span className={ra.caps_ratio > 0.1 ? 'meter-val bad' : 'meter-val good'}>{(ra.caps_ratio * 100).toFixed(1)}%</span>
                    </div>
                    <GaugeBar value={ra.caps_ratio * 100} max={30} color={ra.caps_ratio > 0.1 ? 'var(--amber)' : 'var(--green)'} />
                  </div>
                </div>
              </Section>

              {/* ── 5. Cross-source verification ── */}
              <Section icon={Link2} title="Cross-source verification">
                <p className="body-text">{result.report.cross_source_verification}</p>
                {result.retrieved_sources.length > 0 && (
                  <div className="source-list">
                    {result.retrieved_sources.map((s, i) => (
                      <div key={i} className="source-item">
                        <BookOpen size={12} />
                        <span className="src-name">{s.source}</span>
                        <span className="src-title">{s.title}</span>
                        <span className="src-rel">{(s.relevance * 100).toFixed(0)}% match</span>
                      </div>
                    ))}
                  </div>
                )}
              </Section>

              {/* ── 6. Confidence assessment ── */}
              <Section icon={BarChart2} title="Confidence assessment">
                <p className="body-text">{result.report.confidence_assessment}</p>
                <div className="stat-grid" style={{ marginTop: '0.85rem' }}>
                  <Stat label="Confidence tier" value={pred.confidence_tier.charAt(0).toUpperCase() + pred.confidence_tier.slice(1)} sub={`${conf}% score`} />
                  <Stat label="Word count" value={`${result.word_count} words`} sub={result.reliable ? 'Sufficient length' : 'Too short'} />
                  <Stat label="Pipeline steps" value={result.pipeline_steps?.length} sub="completed" />
                  <Stat label="Report engine" value={result.used_llm ? 'Mistral-7B' : 'Rule-based'} sub={result.used_llm ? 'LLM generated' : 'Deterministic'} />
                </div>
              </Section>

              {/* ── 7. Pipeline trace ── */}
              <Section icon={Cpu} title="Pipeline trace" defaultOpen={false}>
                <div className="pipeline-trace">
                  {result.pipeline_steps?.map((s, i) => (
                    <div key={i} className="trace-row">
                      <span className="trace-num">{i + 1}</span>
                      <span className="trace-step">{s.replace(/_/g, ' ')}</span>
                      <CheckCircle2 size={13} className="trace-check" />
                    </div>
                  ))}
                </div>
                <div className="trace-meta">
                  <span>Source: <strong>{result.input_source === 'url' ? 'URL (scraped)' : 'Direct text'}</strong></span>
                  <span>Length: <strong>{result.text_length?.toLocaleString()} chars</strong></span>
                  {result.timestamp && <span>Time: <strong>{new Date(result.timestamp).toLocaleTimeString()}</strong></span>}
                </div>
              </Section>

              {/* Disclaimer */}
              <p className="disclaimer"><Info size={11} />{result.report.disclaimer}</p>

              {/* PDF */}
              <button className="pdf-btn" onClick={downloadPdf} disabled={pdfLoading}>
                {pdfLoading ? <><Loader2 size={14} className="spin" />Generating…</> : <><Download size={14} />Download PDF report</>}
              </button>
            </div>
          );
        })()}

      </main>
    </div>
  );
}
