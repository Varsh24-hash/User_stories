"""
analyzer.py
Core quality detection engine + Groq AI fix proposal generator.

Detection      : rule-based  (fast, auditable, zero API cost)
Fix generation : Groq (llama-3.3-70b-versatile) via Groq API  (nuanced, context-aware)

Setup:
    pip install groq
    export GROQ_API_KEY=your_key_here   # free at console.groq.com
"""

import re
import json
import os
from groq import Groq
from typing import Any

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.3-70b-versatile"   # fast, free-tier available

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIBONACCI        = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}
FIBONACCI_SORTED = sorted(FIBONACCI)

GIVEN_WHEN_THEN_RE = re.compile(
    r"\bgiven\b.+\bwhen\b.+\bthen\b", re.IGNORECASE | re.DOTALL
)

MAX_TITLE_WORDS  = 12
MAX_STORY_POINTS = 13

# Severity
CRITICAL = "critical"
WARNING  = "warning"

# Issue codes
ISSUE_MISSING_AC    = "missing_ac"
ISSUE_WEAK_AC       = "weak_ac"
ISSUE_NON_FIBONACCI = "non_fibonacci"
ISSUE_OVERSIZED     = "oversized"
ISSUE_ORPHAN        = "orphan"
ISSUE_LONG_TITLE    = "long_title"

# ---------------------------------------------------------------------------
# Scoring helpers  (pure rules — no LLM)
# ---------------------------------------------------------------------------

def nearest_fibonacci(n: int) -> int:
    if n <= 0:
        return 1
    return min(FIBONACCI_SORTED, key=lambda f: abs(f - n))


def score_ac(ac: str) -> int:
    if not ac or not ac.strip():
        return 0
    if GIVEN_WHEN_THEN_RE.search(ac):
        return 40
    return 15


def score_story(story: dict) -> int:
    score = 100
    score -= (40 - score_ac(story.get("acceptance_criteria", "")))
    pts = story.get("points", 0)
    if pts not in FIBONACCI:
        score -= 15
    if pts > MAX_STORY_POINTS:
        score -= 5
    if len(story.get("title", "").split()) > MAX_TITLE_WORDS:
        score -= 10
    if not (story.get("epic") or "").strip():
        score -= 10
    if pts > MAX_STORY_POINTS:
        score -= 10
    return max(0, min(100, score))

# ---------------------------------------------------------------------------
# Issue detection  (pure rules — no LLM)
# ---------------------------------------------------------------------------

def detect_issues(story: dict) -> list[dict]:
    issues = []
    ac    = story.get("acceptance_criteria", "") or ""
    pts   = story.get("points", 0)
    title = story.get("title", "") or ""
    epic  = story.get("epic", "") or ""

    if not ac.strip():
        issues.append({"code": ISSUE_MISSING_AC, "severity": CRITICAL,
                        "message": "No acceptance criteria defined"})
    elif not GIVEN_WHEN_THEN_RE.search(ac):
        issues.append({"code": ISSUE_WEAK_AC, "severity": WARNING,
                        "message": "Acceptance criteria missing Given/When/Then structure"})

    if pts not in FIBONACCI:
        issues.append({"code": ISSUE_NON_FIBONACCI, "severity": WARNING,
                        "message": f"Story points ({pts}) not in Fibonacci sequence"})

    if pts > MAX_STORY_POINTS:
        issues.append({"code": ISSUE_OVERSIZED, "severity": WARNING,
                        "message": f"High story points ({pts}) — consider decomposing"})

    if len(title.split()) > MAX_TITLE_WORDS:
        issues.append({"code": ISSUE_LONG_TITLE, "severity": WARNING,
                        "message": "Title too long — should be a single concise sentence"})

    if not epic.strip():
        issues.append({"code": ISSUE_ORPHAN, "severity": WARNING,
                        "message": "Story not linked to any epic (orphan)"})

    return issues

# ---------------------------------------------------------------------------
# Groq AI fix generation
# ---------------------------------------------------------------------------

def _call_groq(prompt: str, max_tokens: int = 800) -> str:
    """Single Groq API call. Returns text response."""
    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def ai_generate_ac(story: dict) -> str:
    """
    Groq generates a proper Given/When/Then acceptance criterion
    that is specific, testable, and contextually appropriate for the story.
    """
    prompt = f"""You are a senior agile coach reviewing a product backlog.

Generate ONE well-formed acceptance criterion in Given/When/Then format for this user story.

Story ID    : {story.get('id', '')}
Title       : {story.get('title', '')}
Epic        : {story.get('epic', '') or 'unknown'}
Current AC  : {story.get('acceptance_criteria', '') or 'none provided'}

Rules:
- Use exactly the format: "Given [context], when [action], then [outcome]."
- Be specific and testable — mention concrete UI feedback, error handling, or data outcomes
- If current AC exists but is vague, improve it significantly
- Output ONLY the acceptance criterion — no preamble, no explanation, no quotes

Acceptance criterion:"""

    return _call_groq(prompt, max_tokens=200)


def ai_rewrite_title(story: dict) -> str:
    """
    Groq rewrites a long or vague title into a concise, action-oriented
    user story title of maximum 10 words.
    """
    prompt = f"""You are a senior agile coach.

Rewrite this user story title to be concise and clear (maximum 10 words).
Keep it in plain English. Do not use "As a user..." format.
Focus on the core feature being delivered.

Original title : {story.get('title', '')}
Epic           : {story.get('epic', '') or 'unknown'}

Output ONLY the rewritten title — nothing else."""

    return _call_groq(prompt, max_tokens=60)


def ai_suggest_epic(story: dict) -> str:
    """Groq suggests the most appropriate epic for an orphan story."""
    prompt = f"""You are a senior agile coach organising a product backlog.

Suggest the single most appropriate epic name for this user story.
Output 1-3 words only. Be specific to the domain.

Story title : {story.get('title', '')}
Story points: {story.get('points', '')}

Common epics for reference: Auth, Profile, Admin, Notifications, Data, Discovery, UI/UX, Security, Payments, Onboarding

Output ONLY the epic name — no explanation, no punctuation."""

    return _call_groq(prompt, max_tokens=20)


def ai_decompose_story(story: dict) -> list[dict]:
    """
    Groq decomposes an oversized story into 2-4 independently deliverable
    child stories, each with proper Fibonacci points and Given/When/Then AC.
    """
    prompt = f"""You are a senior agile coach decomposing an oversized user story.

Parent story:
  ID     : {story.get('id', '')}
  Title  : {story.get('title', '')}
  Points : {story.get('points', '')}
  AC     : {story.get('acceptance_criteria', '') or 'none'}
  Epic   : {story.get('epic', '') or 'unknown'}

Decompose this into 2-4 independent, deliverable child stories.

Rules:
- Each child must have a clear, concise title (max 8 words)
- Points must be from Fibonacci: 1, 2, 3, 5, 8, 13
- Each child must have one Given/When/Then acceptance criterion
- Children must be independently deployable slices of the parent
- Total child points should roughly equal the parent story points

Respond ONLY with a valid JSON array — no markdown fences, no explanation:
[
  {{
    "title": "child story title",
    "points": 3,
    "acceptance_criteria": "Given ..., when ..., then ..."
  }}
]"""

    raw = _call_groq(prompt, max_tokens=700)
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        children_raw = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                children_raw = json.loads(match.group())
            except json.JSONDecodeError:
                return _fallback_decompose(story)
        else:
            return _fallback_decompose(story)

    sid  = story.get("id", "US-XXX")
    epic = story.get("epic", "")
    children = []
    for i, child in enumerate(children_raw[:4]):   # cap at 4
        suffix = chr(65 + i)                        # A, B, C, D
        pts = child.get("points", 3)
        if pts not in FIBONACCI:
            pts = nearest_fibonacci(pts)
        children.append({
            "id":                 f"{sid}-{suffix}",
            "title":              child.get("title", f"{story['title']} — Part {suffix}"),
            "points":             pts,
            "acceptance_criteria": child.get("acceptance_criteria", ""),
            "epic":               epic,
            "parent":             sid,
        })
    return children


def _fallback_decompose(story: dict) -> list[dict]:
    """Rule-based fallback if Groq JSON fails to parse."""
    sid   = story.get("id", "US-XXX")
    epic  = story.get("epic", "")
    title = story.get("title", "")
    return [
        {
            "id": f"{sid}-A",
            "title": f"{title[:45]} — Part 1",
            "points": 5,
            "acceptance_criteria": "Given a user, when they complete part 1 of this feature, then the core outcome is confirmed.",
            "epic": epic, "parent": sid,
        },
        {
            "id": f"{sid}-B",
            "title": f"{title[:45]} — Part 2",
            "points": 5,
            "acceptance_criteria": "Given a user, when they complete part 2 of this feature, then the feature is fully functional.",
            "epic": epic, "parent": sid,
        },
    ]

# ---------------------------------------------------------------------------
# Fix builder — calls appropriate Groq function per detected issue
# ---------------------------------------------------------------------------

def build_fixes(story: dict, issues: list[dict]) -> dict:
    """
    For each detected issue, call the appropriate Groq AI function.
    Returns: {field_fixes: {field: {before, after, ai_generated}}, children: [...]}
    """
    fixes: dict[str, Any] = {}
    issue_codes = {i["code"] for i in issues}

    # AC — Groq generates a real GWT criterion
    if ISSUE_MISSING_AC in issue_codes or ISSUE_WEAK_AC in issue_codes:
        fixes["acceptance_criteria"] = {
            "before":       story.get("acceptance_criteria", "") or "",
            "after":        ai_generate_ac(story),
            "ai_generated": True,
        }

    # Points — pure math (nearest Fibonacci), no AI needed
    if ISSUE_NON_FIBONACCI in issue_codes:
        pts = story.get("points", 0)
        fixes["points"] = {
            "before":       pts,
            "after":        nearest_fibonacci(pts),
            "ai_generated": False,
        }

    # Title — Groq rewrites concisely
    if ISSUE_LONG_TITLE in issue_codes:
        fixes["title"] = {
            "before":       story.get("title", ""),
            "after":        ai_rewrite_title(story),
            "ai_generated": True,
        }

    # Epic — Groq suggests from context
    if ISSUE_ORPHAN in issue_codes:
        fixes["epic"] = {
            "before":       "",
            "after":        ai_suggest_epic(story),
            "ai_generated": True,
        }

    # Decomposition — Groq produces fully-formed child stories
    children = []
    if ISSUE_OVERSIZED in issue_codes:
        children = ai_decompose_story(story)

    return {"field_fixes": fixes, "children": children}

# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------

def analyze_backlog(stories: list[dict], use_ai: bool = True) -> list[dict]:
    """
    Analyze a list of story dicts.
    use_ai=True  → call Groq for fixes (requires GROQ_API_KEY)
    use_ai=False → detection + scoring only, no API calls
    """
    results = []
    for story in stories:
        issues = detect_issues(story)
        score  = score_story(story)
        fixes  = (build_fixes(story, issues)
                  if use_ai and issues
                  else {"field_fixes": {}, "children": []})
        results.append({
            **story,
            "_score":  score,
            "_issues": issues,
            "_fixes":  fixes,
            "_status": "healthy" if not issues else "needs_review",
        })
    return results


def backlog_health_summary(analyzed: list[dict]) -> dict:
    total   = len(analyzed)
    healthy = sum(1 for s in analyzed if s["_status"] == "healthy")
    scores  = [s["_score"] for s in analyzed]
    return {
        "total":        total,
        "healthy":      healthy,
        "needs_review": total - healthy,
        "critical_count": sum(
            1 for s in analyzed for i in s["_issues"] if i["severity"] == CRITICAL),
        "warning_count": sum(
            1 for s in analyzed for i in s["_issues"] if i["severity"] == WARNING),
        "avg_score": round(sum(scores) / total, 1) if total else 0,
        "issue_type_counts": {
            "Missing / weak AC": sum(
                1 for s in analyzed for i in s["_issues"]
                if i["code"] in (ISSUE_MISSING_AC, ISSUE_WEAK_AC)),
            "Orphan stories": sum(
                1 for s in analyzed for i in s["_issues"]
                if i["code"] == ISSUE_ORPHAN),
            "Non-Fibonacci / oversized": sum(
                1 for s in analyzed for i in s["_issues"]
                if i["code"] in (ISSUE_NON_FIBONACCI, ISSUE_OVERSIZED)),
            "Needs decomposition": sum(
                1 for s in analyzed if s["_fixes"].get("children")),
            "Healthy stories": healthy,
        },
        "scores": {s["id"]: s["_score"] for s in analyzed},
    }


if __name__ == "__main__":
    import pathlib
    data    = json.loads(pathlib.Path("sample_backlog.json").read_text())
    results = analyze_backlog(data, use_ai=True)
    summary = backlog_health_summary(results)
    print(json.dumps(summary, indent=2))