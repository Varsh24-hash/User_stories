"""
app.py
AI-Driven Backlog Quality Gate — Streamlit Application

Pages:
  1. Upload & Analyze   — ingest backlog, trigger AI analysis
  2. Dashboard          — health overview, scores, issue breakdown
  3. Review & Approve   — before/after AI diffs, human approval gate
  4. MCP Log & Export   — live MCP tool call log, schema viewer, export
"""

import json
import copy
import os
import pathlib
import streamlit as st
import pandas as pd

from analyzer import analyze_backlog, backlog_health_summary
from mcp import create_session, MCPClient, get_tool_schemas

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Backlog Quality Gate",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg:        #0d0f14;
  --surface:   #13161e;
  --surface2:  #1a1e2a;
  --border:    #252a38;
  --accent:    #4f8ef7;
  --accent2:   #7c5cfc;
  --success:   #22c55e;
  --warning:   #f59e0b;
  --danger:    #ef4444;
  --ai:        #a78bfa;
  --text:      #e8eaf0;
  --muted:     #6b7280;
  --font-head: 'Syne', sans-serif;
  --font-body: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  --radius:    10px;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
h1,h2,h3 { font-family: var(--font-head) !important; letter-spacing:-0.02em; }
h1 { font-size:2.2rem !important; font-weight:800 !important; }
h2 { font-size:1.5rem !important; font-weight:700 !important; }
h3 { font-size:1.1rem !important; font-weight:600 !important; }

.qg-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
}
.qg-card:hover { border-color: var(--accent); transition: border-color 0.2s; }

.metric-tile {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.4rem;
  text-align: center;
}
.metric-tile .val { font-family:var(--font-head); font-size:2.6rem; font-weight:800; line-height:1; }
.metric-tile .lbl { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); margin-top:0.3rem; }

.score-badge { display:inline-block; padding:2px 10px; border-radius:20px; font-family:var(--font-mono); font-size:0.78rem; font-weight:500; }
.score-high  { background:rgba(34,197,94,0.15);  color:#22c55e; border:1px solid rgba(34,197,94,0.3); }
.score-mid   { background:rgba(245,158,11,0.15); color:#f59e0b; border:1px solid rgba(245,158,11,0.3); }
.score-low   { background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid rgba(239,68,68,0.3); }

.badge-critical { background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid rgba(239,68,68,0.35); padding:2px 8px; border-radius:20px; font-size:0.72rem; }
.badge-warning  { background:rgba(245,158,11,0.15); color:#f59e0b; border:1px solid rgba(245,158,11,0.35); padding:2px 8px; border-radius:20px; font-size:0.72rem; }
.badge-healthy  { background:rgba(34,197,94,0.15);  color:#22c55e; border:1px solid rgba(34,197,94,0.35); padding:2px 8px; border-radius:20px; font-size:0.72rem; }
.badge-decompose{ background:rgba(124,92,252,0.15); color:#a78bfa; border:1px solid rgba(124,92,252,0.35); padding:2px 8px; border-radius:20px; font-size:0.72rem; }

/* AI badge */
.badge-ai { background:rgba(167,139,250,0.15); color:#a78bfa; border:1px solid rgba(167,139,250,0.4); padding:2px 8px; border-radius:20px; font-size:0.68rem; font-family:var(--font-mono); }
.badge-rule { background:rgba(79,142,247,0.12); color:#4f8ef7; border:1px solid rgba(79,142,247,0.3); padding:2px 8px; border-radius:20px; font-size:0.68rem; font-family:var(--font-mono); }

.diff-before { background:rgba(239,68,68,0.08); border-left:3px solid var(--danger); padding:0.6rem 0.9rem; border-radius:0 6px 6px 0; font-family:var(--font-mono); font-size:0.8rem; color:#fca5a5; }
.diff-after  { background:rgba(34,197,94,0.08);  border-left:3px solid var(--success); padding:0.6rem 0.9rem; border-radius:0 6px 6px 0; font-family:var(--font-mono); font-size:0.8rem; color:#86efac; }
.diff-label  { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); margin-bottom:0.25rem; }

.mcp-log { background:#080a0f; border:1px solid var(--border); border-radius:var(--radius); padding:1rem 1.2rem; font-family:var(--font-mono); font-size:0.78rem; line-height:1.8; max-height:400px; overflow-y:auto; }
.mcp-INFO   { color:#6b7280; }
.mcp-UPDATE { color:#4f8ef7; }
.mcp-CREATE { color:#22c55e; }
.mcp-ERROR  { color:#ef4444; }

.schema-card { background:var(--surface2); border:1px solid var(--border); border-radius:6px; padding:0.9rem 1.1rem; margin-bottom:0.6rem; font-family:var(--font-mono); font-size:0.78rem; }
.schema-tool { color:#a78bfa; font-weight:600; font-size:0.85rem; }
.schema-desc { color:#6b7280; margin:0.2rem 0 0.5rem; font-family:var(--font-body); font-size:0.8rem; }
.schema-prop { color:#4f8ef7; }
.schema-type { color:#f59e0b; }

[data-testid="stButton"] button { background:var(--surface2) !important; border:1px solid var(--border) !important; color:var(--text) !important; font-family:var(--font-body) !important; border-radius:6px !important; transition:all 0.15s !important; }
[data-testid="stButton"] button:hover { border-color:var(--accent) !important; color:var(--accent) !important; }
[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,var(--accent),var(--accent2)) !important; }
hr { border-color:var(--border) !important; margin:1.2rem 0 !important; }

.page-header { border-bottom:1px solid var(--border); padding-bottom:0.8rem; margin-bottom:1.6rem; }
.page-header p { color:var(--muted); font-size:0.9rem; margin-top:0.2rem; }

.child-card { background:var(--surface2); border:1px solid var(--border); border-left:3px solid var(--accent2); border-radius:6px; padding:0.7rem 1rem; margin-top:0.4rem; font-size:0.82rem; }

.status-pending  { color:#f59e0b; font-size:0.75rem; }
.status-approved { color:#22c55e; font-size:0.75rem; }
.status-rejected { color:#ef4444; font-size:0.75rem; }

.ai-notice { background:rgba(167,139,250,0.08); border:1px solid rgba(167,139,250,0.25); border-radius:8px; padding:0.8rem 1.1rem; font-size:0.83rem; color:#c4b5fd; margin-bottom:1rem; }

[data-testid="stFileUploader"] { border:1px dashed var(--border) !important; border-radius:var(--radius) !important; background:var(--surface) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "page":           "Upload & Analyze",
        "raw_stories":    None,
        "analyzed":       None,
        "summary":        None,
        "approvals":      {},
        "mcp_client":     None,
        "mcp_done":       False,
        "export_backlog": None,
        "use_ai":         True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

PAGES = [
    ("🔍", "Upload & Analyze"),
    ("📊", "Dashboard"),
    ("✅", "Review & Approve"),
    ("⚡", "MCP Log & Export"),
]

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.2rem;">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#e8eaf0;letter-spacing:-0.02em;">
        🎯 Backlog Quality Gate
      </div>
      <div style="font-size:0.7rem;color:#6b7280;margin-top:4px;letter-spacing:0.05em;text-transform:uppercase;">
        AI-Driven · MCP-Powered
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    for icon, name in PAGES:
        is_active = st.session_state.page == name
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
            st.session_state.page = name
            st.rerun()

    st.divider()

    # AI toggle
    st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b7280;margin-bottom:0.4rem;">Settings</div>', unsafe_allow_html=True)
    use_ai = st.toggle("🤖 Groq AI Fixes", value=st.session_state.use_ai,
                       help="Uses Groq (llama-3.3-70b-versatile) to generate contextual fixes. Disable for rule-based only (faster, no API cost).")
    if use_ai != st.session_state.use_ai:
        st.session_state.use_ai = use_ai

    # API key check
    api_key = os.environ.get("GROQ_API_KEY", "")
    if use_ai:
        if api_key:
            st.markdown('<div style="font-size:0.75rem;color:#22c55e;margin-top:0.3rem;">✓ GROQ_API_KEY detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.75rem;color:#ef4444;margin-top:0.3rem;">✗ GROQ_API_KEY not set</div>', unsafe_allow_html=True)

    st.divider()

    if st.session_state.summary:
        s = st.session_state.summary
        st.markdown(f"""
        <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b7280;margin-bottom:0.5rem;">Session</div>
        <div style="font-size:0.85rem;color:#e8eaf0;line-height:2.1;">
          📦 {s['total']} stories<br>
          ✅ {s['healthy']} healthy<br>
          ❗ {s['critical_count']} critical<br>
          ⚠️ {s['warning_count']} warnings
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.approvals:
        approved_n = sum(1 for v in st.session_state.approvals.values() if v == "approved")
        pending_n  = sum(1 for v in st.session_state.approvals.values() if v == "pending")
        st.markdown(f"""
        <div style="margin-top:0.8rem;font-size:0.82rem;color:#6b7280;line-height:2.1;">
          🟢 {approved_n} approved<br>🕐 {pending_n} pending
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_badge(score: int) -> str:
    cls = "score-high" if score >= 85 else ("score-mid" if score >= 65 else "score-low")
    return f'<span class="score-badge {cls}">{score}/100</span>'

def ai_badge(ai_generated: bool) -> str:
    if ai_generated:
        return '<span class="badge-ai">✦ Groq AI</span>'
    return '<span class="badge-rule">⚙ Rule-based</span>'

# ===========================================================================
# PAGE 1 — Upload & Analyze
# ===========================================================================

def page_upload():
    st.markdown("""
    <div class="page-header">
      <h1>Upload & Analyze</h1>
      <p>Load your backlog JSON or CSV, then run the quality analysis engine.</p>
    </div>
    """, unsafe_allow_html=True)

    use_ai = st.session_state.use_ai

    if use_ai:
        st.markdown("""
        <div class="ai-notice">
          ✦ <strong>Groq AI mode enabled</strong> — fixes will be generated by Groq (llama-3.3-70b-versatile).
          Each story with issues triggers real API calls for contextual AC generation, title rewriting, epic suggestion, and story decomposition.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("⚙ Rule-based mode — fast deterministic fixes, no API calls.")

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        st.markdown("#### Upload Backlog File")
        uploaded = st.file_uploader("Drag & drop .json or .csv",
                                    type=["json", "csv"],
                                    label_visibility="collapsed")

    with col_sample:
        st.markdown("#### Or Use Sample")
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        if st.button("📋  Load sample_backlog.json", use_container_width=True):
            sample_path = pathlib.Path(__file__).parent / "sample_backlog.json"
            if sample_path.exists():
                raw = json.loads(sample_path.read_text())
                _run_analysis(raw)
            else:
                st.error("sample_backlog.json not found.")

    if uploaded:
        try:
            if uploaded.name.endswith(".json"):
                raw = json.load(uploaded)
            else:
                df = pd.read_csv(uploaded)
                raw = df.to_dict(orient="records")
            _run_analysis(raw)
            st.success(f"✓ Loaded {len(raw)} stories from **{uploaded.name}**")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

    if st.session_state.raw_stories:
        st.divider()
        st.markdown("#### Loaded Stories")
        for s in st.session_state.raw_stories:
            ac_text = (s.get("acceptance_criteria") or "").strip()
            ac_display = ac_text[:80] + "..." if len(ac_text) > 80 else (ac_text or "—")
            st.markdown(f"""
            <div class="qg-card" style="padding:1rem 1.4rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#4f8ef7;">{s.get('id','?')}</span>
                <span style="font-size:0.78rem;color:#6b7280;">{s.get('points','?')} pts · {s.get('epic','—') or '—'}</span>
              </div>
              <div style="font-weight:500;margin:0.35rem 0 0.25rem;font-size:0.92rem;">{s.get('title','')}</div>
              <div style="font-size:0.8rem;color:#6b7280;font-family:'JetBrains Mono',monospace;">{ac_display}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        col_a, col_b, _ = st.columns([1, 1, 2])
        with col_a:
            if st.button("🔍  Run Analysis", type="primary", use_container_width=True):
                _run_analysis(st.session_state.raw_stories)
                st.rerun()
        with col_b:
            if st.session_state.analyzed:
                if st.button("📊  View Dashboard →", use_container_width=True):
                    st.session_state.page = "Dashboard"
                    st.rerun()
    elif not uploaded:
        st.markdown("""
        <div class="qg-card" style="text-align:center;padding:3rem 2rem;border-style:dashed;">
          <div style="font-size:2rem;margin-bottom:0.8rem;">📂</div>
          <div style="font-weight:600;color:#e8eaf0;margin-bottom:0.4rem;">No backlog loaded</div>
          <div style="font-size:0.85rem;color:#6b7280;">Upload a JSON/CSV or click "Load sample" to get started.</div>
        </div>
        """, unsafe_allow_html=True)


def _run_analysis(stories: list):
    use_ai = st.session_state.use_ai
    api_key = os.environ.get("GROQ_API_KEY", "")

    if use_ai and not api_key:
        st.warning("⚠️ GROQ_API_KEY not set. Falling back to rule-based mode.")
        use_ai = False

    st.session_state.raw_stories = copy.deepcopy(stories)

    if use_ai:
        progress_bar = st.progress(0, text="🤖 Groq AI is analyzing your backlog...")
        analyzed = []
        total = len(stories)

        for i, story in enumerate(stories):
            progress_bar.progress(
                (i + 1) / total,
                text=f"🤖 Groq analyzing {story['id']} ({i+1}/{total})..."
            )
            from analyzer import detect_issues, score_story, build_fixes
            issues = detect_issues(story)
            score  = score_story(story)
            fixes  = build_fixes(story, issues) if issues else {"field_fixes": {}, "children": []}
            analyzed.append({
                **story,
                "_score":  score,
                "_issues": issues,
                "_fixes":  fixes,
                "_status": "healthy" if not issues else "needs_review",
            })

        progress_bar.progress(1.0, text="✓ Analysis complete!")
        import time; time.sleep(0.5)
        progress_bar.empty()
    else:
        analyzed = analyze_backlog(stories, use_ai=False)

    st.session_state.analyzed = analyzed
    st.session_state.summary = backlog_health_summary(analyzed)

    approvals = {}
    for s in analyzed:
        if s["_status"] != "healthy":
            approvals[s["id"]] = "pending"
    st.session_state.approvals = approvals
    st.session_state.mcp_client = None
    st.session_state.mcp_done = False
    st.session_state.export_backlog = None

    st.success(f"✓ Analysis complete — {len(analyzed)} stories processed.")

# ===========================================================================
# PAGE 2 — Dashboard
# ===========================================================================

def page_dashboard():
    if not st.session_state.analyzed:
        _no_data_prompt("Dashboard"); return

    summary  = st.session_state.summary
    analyzed = st.session_state.analyzed

    st.markdown("""
    <div class="page-header">
      <h1>Backlog Health Dashboard</h1>
      <p>Quality overview, issue breakdown, and per-story scores.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    cols = st.columns(5)
    kpis = [
        (str(round(summary["avg_score"])), "Health Score",    "#4f8ef7"),
        (str(summary["total"]),            "Total Stories",   "#e8eaf0"),
        (str(summary["healthy"]),          "Healthy",         "#22c55e"),
        (str(summary["critical_count"]),   "Critical Issues", "#ef4444"),
        (str(summary["warning_count"]),    "Warnings",        "#f59e0b"),
    ]
    for col, (val, lbl, color) in zip(cols, kpis):
        col.markdown(f"""
        <div class="metric-tile">
          <div class="val" style="color:{color};">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Issue Breakdown")
        total = summary["total"]
        for label, count in summary["issue_type_counts"].items():
            pct   = count / total if total else 0
            color = "#22c55e" if label == "Healthy stories" else ("#ef4444" if "AC" in label else "#f59e0b")
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
              <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:0.3rem;">
                <span>{label}</span>
                <span style="font-family:'JetBrains Mono',monospace;color:{color};">{count}/{total}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pct)

    with col_right:
        st.markdown("#### Story Quality Scores")
        for sid, sc in summary["scores"].items():
            cls = "score-high" if sc >= 85 else ("score-mid" if sc >= 65 else "score-low")
            bar_color = "#22c55e" if sc >= 85 else ("#f59e0b" if sc >= 65 else "#ef4444")
            st.markdown(f"""
            <div style="margin-bottom:0.7rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#4f8ef7;">{sid}</span>
                <span class="score-badge {cls}">{sc}</span>
              </div>
              <div style="background:#1a1e2a;border-radius:4px;height:6px;overflow:hidden;">
                <div style="width:{sc}%;background:{bar_color};height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Story Detail")
    for story in analyzed:
        issues      = story["_issues"]
        has_children = bool(story["_fixes"].get("children"))
        crit = [i for i in issues if i["severity"] == "critical"]
        warn = [i for i in issues if i["severity"] == "warning"]

        badges = ""
        if not issues:
            badges = '<span class="badge-healthy">● healthy</span>'
        else:
            if crit: badges += f'<span class="badge-critical">● {len(crit)} critical</span> '
            if warn: badges += f'<span class="badge-warning">▲ {len(warn)} warning</span> '
            if has_children: badges += '<span class="badge-decompose">⬡ decompose</span>'

        issue_list = "".join(
            f'<div style="color:#a8b4cc;font-size:0.8rem;margin-top:0.2rem;">• {i["message"]}</div>'
            for i in issues
        )
        st.markdown(f"""
        <div class="qg-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#4f8ef7;">{story['id']}</span>
              <span style="margin-left:0.6rem;font-size:0.78rem;color:#6b7280;">{story.get('points','?')} pts · {story.get('epic','—') or '—'}</span>
            </div>
            <div style="display:flex;gap:0.4rem;align-items:center;">{badges} {score_badge(story['_score'])}</div>
          </div>
          <div style="font-weight:500;margin:0.4rem 0 0.3rem;">{story['title']}</div>
          {issue_list}
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    if st.button("✅  Go to Review & Approve →"):
        st.session_state.page = "Review & Approve"
        st.rerun()

# ===========================================================================
# PAGE 3 — Review & Approve
# ===========================================================================

def page_review():
    if not st.session_state.analyzed:
        _no_data_prompt("Review & Approve"); return

    analyzed  = st.session_state.analyzed
    approvals = st.session_state.approvals
    needs_review = [s for s in analyzed if s["id"] in approvals]

    st.markdown(f"""
    <div class="page-header">
      <h1>Review & Approve</h1>
      <p>{len(needs_review)} stories need attention. Review AI-generated fixes and approve or reject before MCP applies changes.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.use_ai:
        st.markdown("""
        <div class="ai-notice">
          ✦ Fixes marked <strong>Groq AI</strong> were generated by Groq (llama-3.3-70b-versatile) based on the story's full context.
          Fixes marked <strong>Rule-based</strong> are deterministic (e.g. nearest Fibonacci number).
        </div>
        """, unsafe_allow_html=True)

    # Bulk actions
    col_a, col_b, _ = st.columns([1, 1, 4])
    with col_a:
        if st.button("✓  Approve All", use_container_width=True):
            for sid in approvals: st.session_state.approvals[sid] = "approved"
            st.rerun()
    with col_b:
        if st.button("✗  Reject All", use_container_width=True):
            for sid in approvals: st.session_state.approvals[sid] = "rejected"
            st.rerun()

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    for story in needs_review:
        sid    = story["id"]
        status = approvals.get(sid, "pending")
        fixes  = story["_fixes"]
        field_fixes = fixes.get("field_fixes", {})
        children    = fixes.get("children", [])

        status_html = {
            "pending":  '<span class="status-pending">⏳ pending</span>',
            "approved": '<span class="status-approved">✓ approved</span>',
            "rejected": '<span class="status-rejected">✗ rejected</span>',
        }.get(status, "")

        issues  = story["_issues"]
        crit_n  = sum(1 for i in issues if i["severity"] == "critical")
        warn_n  = sum(1 for i in issues if i["severity"] == "warning")

        border_color = "#ef4444" if crit_n else "#f59e0b"
        if status == "approved": border_color = "#22c55e"
        if status == "rejected": border_color = "#6b7280"

        with st.container():
            st.markdown(f"""
            <div class="qg-card" style="border-left:3px solid {border_color};">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#4f8ef7;font-weight:600;">{sid}</span>
                  <span style="margin-left:0.5rem;font-size:0.75rem;color:#6b7280;">{crit_n} critical · {warn_n} warning</span>
                </div>
                <div style="display:flex;gap:0.5rem;align-items:center;">{score_badge(story['_score'])} {status_html}</div>
              </div>
              <div style="font-weight:500;margin:0.5rem 0 0.8rem;font-size:0.95rem;">{story['title']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Field diffs with AI/rule badge
            for field, change in field_fixes.items():
                before_val = str(change["before"]) if change["before"] != "" else "empty"
                after_val  = str(change["after"])
                before_disp = before_val[:200] + "..." if len(before_val) > 200 else before_val
                after_disp  = after_val[:200]  + "..." if len(after_val)  > 200 else after_val
                is_ai = change.get("ai_generated", False)

                st.markdown(f"""
                <div style="margin:0 0 0.8rem 0.2rem;">
                  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;">
                    <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b7280;font-family:'JetBrains Mono',monospace;">· {field}</span>
                    {ai_badge(is_ai)}
                  </div>
                  <div class="diff-before"><span class="diff-label">Before</span>{before_disp}</div>
                  <div style="height:4px"></div>
                  <div class="diff-after"><span class="diff-label">After</span>{after_disp}</div>
                </div>
                """, unsafe_allow_html=True)

            # Child stories
            if children:
                n = len(children)
                st.markdown(f"""
                <div style="margin:0.4rem 0 0.6rem 0.2rem;">
                  <span style="font-size:0.78rem;color:#a78bfa;">⬡ Will create {n} child {'stories' if n > 1 else 'story'} via MCP</span>
                  <span style="margin-left:0.5rem;">{ai_badge(True)}</span>
                </div>
                """, unsafe_allow_html=True)
                for child in children:
                    st.markdown(f"""
                    <div class="child-card">
                      <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#a78bfa;">{child['id']}</span>
                      <span style="margin-left:0.5rem;font-size:0.82rem;font-weight:500;">{child['title']}</span>
                      <span style="float:right;font-size:0.75rem;color:#6b7280;">{child['points']} pts</span>
                      <div style="font-size:0.78rem;color:#6b7280;margin-top:0.3rem;font-family:'JetBrains Mono',monospace;">{child['acceptance_criteria']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            btn1, btn2, _ = st.columns([1, 1, 4])
            with btn1:
                lbl = "✓ Approve & Apply" if status != "approved" else "✓ Approved"
                if st.button(lbl, key=f"approve_{sid}", use_container_width=True,
                             disabled=(status == "approved")):
                    st.session_state.approvals[sid] = "approved"
                    st.rerun()
            with btn2:
                lbl = "✗ Reject" if status != "rejected" else "✗ Rejected"
                if st.button(lbl, key=f"reject_{sid}", use_container_width=True,
                             disabled=(status == "rejected")):
                    st.session_state.approvals[sid] = "rejected"
                    st.rerun()

            st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)

    approved_ids = [sid for sid, v in approvals.items() if v == "approved"]
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if approved_ids:
        st.markdown(f'<div style="font-size:0.88rem;color:#22c55e;margin-bottom:0.6rem;">{len(approved_ids)} stories approved and ready to apply via MCP</div>', unsafe_allow_html=True)
        if st.button("⚡  Apply via MCP", type="primary"):
            _apply_via_mcp(approved_ids)
            st.session_state.page = "MCP Log & Export"
            st.rerun()
    else:
        st.info("Approve at least one story to enable MCP application.")


def _apply_via_mcp(approved_ids: list[str]):
    raw_stories     = st.session_state.raw_stories
    analyzed        = st.session_state.analyzed
    client          = create_session(copy.deepcopy(raw_stories))
    approved_stories = [s for s in analyzed if s["id"] in approved_ids]
    client.apply_approved_fixes(approved_stories)
    st.session_state.mcp_client     = client
    st.session_state.mcp_done       = True
    st.session_state.export_backlog = client.export_backlog()

# ===========================================================================
# PAGE 4 — MCP Log & Export
# ===========================================================================

def page_mcp_log():
    st.markdown("""
    <div class="page-header">
      <h1>MCP Log & Export</h1>
      <p>Live MCP tool call log, protocol schema, and updated backlog export.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── MCP Schema viewer (always visible) ──
    with st.expander("📐 MCP Tool Schemas — what the server exposes", expanded=False):
        for schema in get_tool_schemas():
            props = schema.get("inputSchema", {}).get("properties", {})
            props_html = "".join(
                f'<div style="margin-left:1rem;"><span class="schema-prop">{k}</span>: <span class="schema-type">{v.get("type","any")}</span> — {v.get("description","")}</div>'
                for k, v in props.items()
            )
            st.markdown(f"""
            <div class="schema-card">
              <div class="schema-tool">{schema['name']}</div>
              <div class="schema-desc">{schema['description']}</div>
              {props_html}
            </div>
            """, unsafe_allow_html=True)

    if not st.session_state.mcp_done or not st.session_state.mcp_client:
        st.markdown("""
        <div class="qg-card" style="text-align:center;padding:3rem 2rem;border-style:dashed;margin-top:1rem;">
          <div style="font-size:2rem;margin-bottom:0.8rem;">⚡</div>
          <div style="font-weight:600;color:#e8eaf0;margin-bottom:0.4rem;">No MCP session yet</div>
          <div style="font-size:0.85rem;color:#6b7280;">Go to Review & Approve, approve stories, then click "Apply via MCP".</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("← Go to Review & Approve"):
            st.session_state.page = "Review & Approve"
            st.rerun()
        return

    client: MCPClient = st.session_state.mcp_client
    approvals = st.session_state.approvals
    approved_n = sum(1 for v in approvals.values() if v == "approved")
    rejected_n = sum(1 for v in approvals.values() if v == "rejected")

    # Stats
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="metric-tile"><div class="val" style="color:#22c55e;">{approved_n}</div><div class="lbl">Approved</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-tile"><div class="val" style="color:#ef4444;">{rejected_n}</div><div class="lbl">Rejected</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-tile"><div class="val" style="color:#4f8ef7;">{len(client.list_stories())}</div><div class="lbl">Stories in Backlog</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Log viewer ──
    tabs = st.tabs(["📋 Activity Log", "🔬 Raw Tool Calls", "📦 Export"])

    with tabs[0]:
        log_lines = client.get_log_lines()
        colored   = ""
        for line in log_lines:
            cls = ("mcp-UPDATE" if "[UPDATE]" in line else
                   "mcp-CREATE" if "[CREATE]" in line else
                   "mcp-ERROR"  if "[ERROR]"  in line else "mcp-INFO")
            colored += f'<div class="{cls}">{line}</div>\n'
        st.markdown(f'<div class="mcp-log">{colored}</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("#### Structured tool_call / tool_response pairs")
        for entry in client.get_log():
            if "tool_call" not in entry:
                continue
            tc  = entry["tool_call"]
            tr  = entry["tool_response"]
            ok  = tr["result"].get("ok", True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<div style="font-size:0.75rem;color:#a78bfa;margin-bottom:0.2rem;">→ tool_call</div>', unsafe_allow_html=True)
                st.code(json.dumps(tc, indent=2), language="json")
            with col_b:
                st.markdown(f'<div style="font-size:0.75rem;color:{"#22c55e" if ok else "#ef4444"};margin-bottom:0.2rem;">← tool_response</div>', unsafe_allow_html=True)
                st.code(json.dumps(tr, indent=2), language="json")

    with tabs[2]:
        export_data = st.session_state.export_backlog or []
        col_exp, col_btn = st.columns([3, 1])
        with col_exp:
            st.code(json.dumps(export_data, indent=2, ensure_ascii=False), language="json")
        with col_btn:
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button("⬇ Download JSON", data=json_str,
                               file_name="updated_backlog.json",
                               mime="application/json", use_container_width=True)
            if export_data:
                df = pd.DataFrame(export_data)
                st.download_button("⬇ Download CSV", data=df.to_csv(index=False),
                                   file_name="updated_backlog.csv",
                                   mime="text/csv", use_container_width=True)

    st.divider()
    if st.button("🔄  Start New Analysis"):
        for k in ["raw_stories","analyzed","summary","approvals",
                  "mcp_client","mcp_done","export_backlog"]:
            st.session_state[k] = {} if k == "approvals" else None
        st.session_state.page = "Upload & Analyze"
        st.rerun()

# ===========================================================================
# Utility
# ===========================================================================

def _no_data_prompt(page_name: str):
    st.markdown(f"""
    <div class="page-header"><h1>{page_name}</h1></div>
    <div class="qg-card" style="text-align:center;padding:3rem 2rem;border-style:dashed;">
      <div style="font-size:2rem;margin-bottom:0.8rem;">📂</div>
      <div style="font-weight:600;color:#e8eaf0;margin-bottom:0.4rem;">No data loaded</div>
      <div style="font-size:0.85rem;color:#6b7280;">Upload or load a backlog first.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("← Go to Upload & Analyze"):
        st.session_state.page = "Upload & Analyze"
        st.rerun()

# ===========================================================================
# Router
# ===========================================================================

PAGE_FN = {
    "Upload & Analyze": page_upload,
    "Dashboard":        page_dashboard,
    "Review & Approve": page_review,
    "MCP Log & Export": page_mcp_log,
}
PAGE_FN[st.session_state.page]()