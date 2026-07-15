// src/components/ModerationResult.jsx

import ScoreBar from "./ScoreBar";

const LABEL_CONFIG = {
  ALLOW: { text: "Allowed", tone: "allow" },
  FLAG: { text: "Needs review", tone: "flag" },
  BLOCK: { text: "Blocked", tone: "block" },
};

export default function ModerationResult({ result }) {
  if (!result) return null;

  const config = LABEL_CONFIG[result.label] || LABEL_CONFIG.FLAG;

  return (
    <section className={`result-card ${config.tone}`} aria-live="polite">
      <header className="result-header">
        <div>
          <p className="result-kicker">Decision</p>
          <h2>{config.text}</h2>
        </div>
        <span className={`decision-pill ${config.tone}`}>{result.label}</span>
      </header>

      <dl className="result-meta">
        <div>
          <dt>Confidence</dt>
          <dd>{(result.confidence * 100).toFixed(2)}%</dd>
        </div>
        <div>
          <dt>Latency</dt>
          <dd>{result.latency_ms}ms</dd>
        </div>
      </dl>

      <div className="score-section">
        <p className="subheading">Class scores</p>
        <ScoreBar label="ALLOW" score={result.scores.ALLOW} />
        <ScoreBar label="FLAG" score={result.scores.FLAG} />
        <ScoreBar label="BLOCK" score={result.scores.BLOCK} />
      </div>
    </section>
  );
}
