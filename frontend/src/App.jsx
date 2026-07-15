// src/App.jsx

import "./App.css";
import ModerationForm from "./components/ModerationForm";

export default function App() {

  return (
    <main className="app-shell">
      <section className="app-header" aria-labelledby="page-title">
        <div>
          <p className="eyebrow">Review Console</p>
          <h1 id="page-title">Content Moderation</h1>
          <p className="lede">
            Check user-submitted text and review model decisions.
          </p>
        </div>
      </section>

      <section className="workspace">
        <div className="panel">
          <div className="section-heading">
            <span>Moderation input</span>
            <span>Ctrl + Enter to submit</span>
          </div>
          <ModerationForm />
        </div>
      </section>
    </main>
  );
}
