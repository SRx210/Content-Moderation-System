// src/components/ModerationForm.jsx

import { useState } from "react";
import { moderateText } from "../api/moderationApi";
import ModerationResult from "./ModerationResult";

export default function ModerationForm() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await moderateText(text);
      setResult(data);
    } catch {
      setError("Failed to reach moderation API. Is the server running?");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSubmit();
  }

  return (
    <div className="moderation-form">
      <textarea
        className="content-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Paste a comment, message, or post for review."
        rows={5}
      />

      <div className="form-actions">
        <span className="character-count">{text.length} characters</span>
        <button
          className="primary-button"
          type="button"
          onClick={handleSubmit}
          disabled={loading || !text.trim()}
        >
          {loading ? "Analyzing..." : "Moderate"}
        </button>
      </div>

      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}

      <ModerationResult result={result} />
    </div>
  );
}
