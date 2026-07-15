// src/components/ScoreBar.jsx

const LABEL_CLASS = {
  ALLOW: "allow",
  FLAG: "flag",
  BLOCK: "block",
};

export default function ScoreBar({ label, score }) {
  const pct = (score * 100).toFixed(1);

  return (
    <div className="score-row">
      <div className="score-label">
        <span>{label}</span>
        <strong>{pct}%</strong>
      </div>
      <div className="track">
        <div className={`track-fill ${LABEL_CLASS[label]}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
