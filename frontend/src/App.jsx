import React, { useState } from 'react';
import axios from 'axios';
import { Search, BrainCircuit, CheckCircle2, CircleDashed, Loader2, Info } from 'lucide-react';
import './index.css';

const STEPS = [
  'Analyzing text input...',
  'Extracting linguistic features...',
  'Running logistic regression model...',
  'Synthesizing final credibility score...'
];

export default function App() {
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeStepIndex, setActiveStepIndex] = useState(-1);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    setIsLoading(true);
    setResult(null);
    setError(null);
    setActiveStepIndex(0);

    // Simulate progress through steps for UX (each step ~800ms)
    const stepInterval = setInterval(() => {
      setActiveStepIndex(prev => {
        if (prev < STEPS.length - 1) return prev + 1;
        clearInterval(stepInterval);
        return prev;
      });
    }, 800);

    try {
      const isUrl = inputValue.startsWith('http');
      const payload = isUrl ? { url: inputValue } : { text: inputValue };

      const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${API_BASE_URL}/predict`, payload);

      // Artificial delay to ensure steps finish nicely
      setTimeout(() => {
        clearInterval(stepInterval);
        setActiveStepIndex(STEPS.length);
        setResult(response.data);
        setIsLoading(false);
      }, STEPS.length * 800 + 500);

    } catch (err) {
      clearInterval(stepInterval);
      setIsLoading(false);
      setActiveStepIndex(-1);
      setError(err.response?.data?.detail || 'An error occurred while analyzing the text.');
    }
  };

  const isReal = result?.prediction === 'Real News';

  return (
    <div className="app-container">
      <div className="logo-container" onClick={() => { setResult(null); setInputValue(''); }}>
        <BrainCircuit className="logo-icon" />
        <div className="logo-text">
          <h1>News Credibility AI</h1>
          <p>AI Research Assistant</p>
        </div>
      </div>

      <form className="search-container" onSubmit={handleSearch}>
        <Search className="search-icon" size={20} />
        <input
          type="text"
          className="search-input"
          placeholder="Enter news text or URL to analyze..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          disabled={isLoading}
        />
        <button
          type="submit"
          className="search-button"
          disabled={!inputValue.trim() || isLoading}
        >
          {isLoading ? <Loader2 className="animate-spin" size={20} /> : 'Analyze'}
        </button>
      </form>

      {error && (
        <div className="results-container" style={{ margin: 0, padding: 0 }}>
          <div className="metric-card" style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}>
            <p>Error: {error}</p>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="steps-container">
          {STEPS.map((stepText, idx) => {
            const isCompleted = idx < activeStepIndex;
            const isActive = idx === activeStepIndex;
            return (
              <div key={idx} className={`step-item ${isCompleted ? 'completed' : isActive ? 'active' : 'pending'}`}>
                {isCompleted ? (
                  <CheckCircle2 className="step-icon" size={20} />
                ) : isActive ? (
                  <Loader2 className="step-icon animate-spin" size={20} />
                ) : (
                  <CircleDashed className="step-icon" size={20} />
                )}
                <span>{stepText}</span>
              </div>
            );
          })}
        </div>
      )}

      {result && !isLoading && (
        <div className="results-container">
          <div className="result-card">
            <div className="result-header">
              <span className="result-title">Credibility Result</span>
              <span className="confidence-badge">{result.confidence_score}% Confidence</span>
            </div>
            <div className="result-content">
              <span>According to our rigorous NLP analysis, this article is predicted as:</span>
              <div className={`prediction-text ${isReal ? 'real' : 'fake'}`}>
                {result.prediction}
              </div>
              <p style={{ marginTop: '1rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                {result.message}
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem' }}>
            <Info size={16} color="var(--text-main)" />
            <span style={{ fontWeight: 600, fontSize: '1.1rem' }}>Analysis Details</span>
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-title">Input Source</div>
              <div className="metric-value">
                {result.input_source === 'url' ? 'Extracted from Web URL' : 'Direct Text Input'}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Text Length</div>
              <div className="metric-value">
                {result.text_length.toLocaleString()} characters processed
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
