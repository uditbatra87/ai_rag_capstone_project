# =============================================================================
# intermediate_example.py  —  ASYNC + FULL DATASET + ACCURACY SCORING (Step 3 of 4)
# =============================================================================
#
# LEARNING GOAL
# ─────────────
# Bridge between the simple examples and the full assignment_template.
# Three things are new compared to simple_example_02:
#
#   1. SCALE  — run on all 10 snippets instead of just 1
#               (10 snippets × 4 strategies = 40 concurrent API calls)
#
#   2. GOLDEN ANSWERS  — load golden_set.jsonl, which contains the correct
#               company / role / years for each snippet. This lets us check
#               whether the model's extraction was right.
#
#   3. ACCURACY SCORING  — after all 40 calls complete, compare each
#               extracted field to the golden answer and assign a score 0–3.
#               This is deterministic (no second LLM call needed).
#
# WHAT IS STILL MISSING vs assignment_template:
#   • Cost tracking (prompt + completion token counts)
#   • Latency measurement (wall-clock time per call)
#   • LLM-as-judge scoring (a second model grades the extraction)
#   • JSON and CSV output files
#
# PROGRESSION
# ───────────
#   simple_example_01    (sync client,  for loop,    sequential,  1×4, no scoring)
#   simple_example_02    (async client, for loop,    sequential,  1×4, no scoring)
#   simple_example_03    (async client, gather(),    concurrent,  1×4, no scoring)
#   intermediate_example ← YOU ARE HERE (async, gather(), 10×4, accuracy scoring)
#   assignment_template  (async, 10×4, accuracy + LLM judge + cost + 3 outputs)
#
# OUTPUT
# ──────
#   intermediate_example_report.html  — 3 sections:
#     1. Strategy accuracy summary table
#     2. Per-row scored detail table (✔/✘ per field)
#     3. Full prompt + response trace for every snippet × strategy
#
# HOW TO RUN
# ──────────
#   pip install openai pandas
#   export OPENAI_API_KEY="sk-..."
#   python intermediate_example.py
# =============================================================================

# ── Standard-library imports ─────────────────────────────────────────────────
import asyncio
import json
import os
import re
from pathlib import Path

# ── Third-party imports ───────────────────────────────────────────────────────
import pandas as pd          # used for groupby aggregation in the summary table
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

assert os.environ.get('OPENAI_API_KEY'), (
    'OPENAI_API_KEY is not set. '
    'Run: export OPENAI_API_KEY="sk-..." before executing this script.'
)

client      = AsyncOpenAI()
MODEL       = 'gpt-4o-mini'
TEMPERATURE = 0.0

# scripts/ is one level below the project root, so data/ is at ../data/
DATA_DIR = Path('../data')

# Strategy → hex color for HTML badges (consistent across all scripts)
STRATEGY_COLORS = {
    'zero_shot':  '#6c757d',
    'few_shot':   '#0d6efd',
    'structured': '#198754',
    'cot':        '#fd7e14',
}


# =============================================================================
# STEP 1 — LOAD ALL SNIPPETS AND GOLDEN ANSWERS
# =============================================================================
# Unlike simple_example_01/02, we load the FULL dataset here.
#
# job_snippets.jsonl — one job posting per line
#   Each line: {"id": "job_001", "snippet": "...posting text..."}
#
# golden_set.jsonl   — one correct answer per line (same IDs)
#   Each line: {"id": "job_001", "company": "Acme", "role": "Engineer",
#               "years_experience_required": 3}
#
# We build `golden` as a dict keyed by id so any result row can look up
# its correct answer in O(1): golden[row['snippet_id']]

snippets = [
    json.loads(line)
    for line in (DATA_DIR / 'job_snippets.jsonl').read_text(encoding='utf-8').splitlines()
    if line.strip()
]

golden = {
    row['id']: row
    for row in (
        json.loads(line)
        for line in (DATA_DIR / 'golden_set.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    )
}

print('─' * 60)
print(f'STEP 1 — Data loaded: {len(snippets)} snippets, {len(golden)} golden answers')
print()


# =============================================================================
# STEP 2 — PROMPT STRATEGY FUNCTIONS
# =============================================================================
# Identical in purpose to simple_example_02.
# Each function returns a messages list for one strategy.
# All four strategies target the same three extraction fields.

def prompt_zero_shot(text: str) -> list[dict]:
    """Strategy 1 — zero-shot: ask directly with no examples or persona.

    Minimal prompt. The model relies entirely on its training to interpret
    the task. Works well for clear postings, may struggle with edge cases.
    """
    return [
        {
            'role': 'user',
            'content': (
                'Extract "company", "role", and "years_experience_required" '
                'from this job posting. Return a JSON object. '
                'Use an integer for years, or null if not stated.\n\n'
                f'Job posting:\n{text}'
            ),
        }
    ]


def prompt_few_shot(text: str) -> list[dict]:
    """Strategy 2 — few-shot: provide 3 worked examples covering edge cases.

    Examples teach:
      • Normal integer years (3+ → 3)
      • Zero years ("fresh graduates welcome" → 0)
      • Null years (no requirement stated → null)

    The trailing "Output:" cue nudges the model to reply with JSON directly.
    """
    examples = (
        'Example 1:\n'
        'Posting: "TechCorp is hiring a Backend Engineer. 3+ years Python required."\n'
        'Output: {"company": "TechCorp", "role": "Backend Engineer", "years_experience_required": 3}\n\n'
        'Example 2:\n'
        'Posting: "DataWorks Ltd seeks a Data Scientist. Fresh graduates welcome."\n'
        'Output: {"company": "DataWorks Ltd", "role": "Data Scientist", "years_experience_required": 0}\n\n'
        'Example 3:\n'
        'Posting: "Omega Systems needs a Principal Architect. No years requirement listed."\n'
        'Output: {"company": "Omega Systems", "role": "Principal Architect", "years_experience_required": null}'
    )
    return [
        {
            'role': 'user',
            'content': (
                'Extract "company", "role", and "years_experience_required" from a job posting.\n'
                f'Return JSON with exactly those three keys.\n\n{examples}\n\n'
                f'Now extract from:\n{text}\nOutput:'
            ),
        }
    ]


def prompt_structured(text: str) -> list[dict]:
    """Strategy 3 — structured: system persona + strict output schema.

    A system message sets the model's persona and output rules permanently.
    This strategy is the most explicit about format and edge-case handling,
    which helps consistency across varied input styles.
    """
    return [
        {
            'role': 'system',
            'content': (
                'You are an expert technical recruiter and data extraction specialist.\n\n'
                'Return raw JSON only (no markdown fences, no explanation) with:\n'
                '  "company"                   (string) — hiring company name\n'
                '  "role"                       (string) — exact job title\n'
                '  "years_experience_required"  (integer or null) — '
                'minimum years required; use lower bound if a range; '
                'convert word-numbers to integers; null if not stated.\n\n'
                'Never fabricate information not present in the posting.'
            ),
        },
        {
            'role': 'user',
            'content': f'Extract from this job posting:\n\n{text}',
        },
    ]


def prompt_cot(text: str) -> list[dict]:
    """Strategy 4 — chain-of-thought: numbered reasoning steps before the JSON.

    Forces the model to address each field explicitly before outputting JSON.
    Most useful for postings with implicit or ambiguous years requirements.
    parse_response() handles the reasoning-before-JSON format by finding
    the LAST {...} block in the response.
    """
    return [
        {
            'role': 'user',
            'content': (
                f'Job posting:\n{text}\n\n'
                'Think step by step:\n'
                '1. What is the company name?\n'
                '2. What is the exact job title?\n'
                '3. How many years of experience are required?\n'
                '   (integer; minimum if range; null if not stated)\n'
                '4. Output a JSON object:\n'
                '   {"company": ..., "role": ..., "years_experience_required": ...}\n\n'
                'Show your reasoning for each field, then end with the JSON.'
            ),
        }
    ]


# Registry of all strategies — the only place you need to edit to add a new one.
STRATEGIES = {
    'zero_shot':  prompt_zero_shot,
    'few_shot':   prompt_few_shot,
    'structured': prompt_structured,
    'cot':        prompt_cot,
}


# =============================================================================
# STEP 3 — ASYNC CALL FUNCTION  (one snippet × one strategy)
# =============================================================================

async def call_one(strategy_name: str, snippet: dict) -> dict:
    """Fire one API call for one strategy on one snippet.

    Compared to simple_example_02.call_strategy(), this version also
    captures `snippet_id` in the return dict. This is essential because
    asyncio.gather() returns results in task-creation order, but we need
    to match each result to its golden answer — and the results dict must
    carry the ID to make that lookup possible.

    Args:
        strategy_name: Key into STRATEGIES dict.
        snippet:       Full snippet dict (uses ['snippet'] and ['id']).

    Returns:
        Dict with keys: strategy, snippet_id, messages, raw_response.
    """
    # Build the prompt messages for this strategy (synchronous, instant)
    messages = STRATEGIES[strategy_name](snippet['snippet'])

    # await suspends this coroutine during the network request.
    # Other coroutines (other calls in the gather) run in the meantime.
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )

    return {
        'strategy':     strategy_name,
        'snippet_id':   snippet['id'],               # needed to look up gold answer
        'messages':     messages,                    # saved for the HTML trace
        'raw_response': resp.choices[0].message.content or '',
    }


async def run_all() -> list[dict]:
    """Fire all 40 calls (10 snippets × 4 strategies) concurrently.

    The nested list comprehension creates 40 coroutine objects — one per
    (snippet, strategy) pair. asyncio.gather() launches them all at once.

    Total wall-clock time ≈ slowest single call (not 40× the average).

    Returns:
        List of 40 result dicts. Order matches task-creation order
        (snippet 0 all strategies, snippet 1 all strategies, …).
    """
    tasks = [
        call_one(strategy_name, snip)
        for snip in snippets          # outer loop: 10 snippets
        for strategy_name in STRATEGIES  # inner loop: 4 strategies each
    ]
    return list(await asyncio.gather(*tasks))


# =============================================================================
# STEP 4 — RESPONSE PARSING
# =============================================================================

def parse_response(text: str) -> dict | None:
    """Extract a Python dict from a raw model response.

    Three-step fallback (same pattern as all other scripts):
      1. Strip markdown fences → try json.loads()
      2. Regex-find last {...} block → try json.loads()
         (handles CoT format: "reasoning text ... {json}")
      3. Return None

    Args:
        text: Raw response string from the model.

    Returns:
        dict on success, None on failure.
    """
    # Step 1: strip ``` / ```json fences, parse directly
    cleaned = re.sub(r'```(?:json)?\s*', '', text).strip()
    cleaned = cleaned.replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 2: find last single-depth {...} block
    matches = list(re.finditer(r'\{[^{}]+\}', text, re.DOTALL))
    for match in reversed(matches):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue

    return None


# =============================================================================
# STEP 5 — ACCURACY SCORING
# =============================================================================

def score_accuracy(extracted: dict | None, gold: dict) -> int:
    """Compare extracted fields to the golden answer. Return 0–3.

    One point is awarded per correctly extracted field:
      • company  (string, case-insensitive, whitespace-trimmed)
      • role     (string, case-insensitive, whitespace-trimmed)
      • years_experience_required (integer comparison; null == null)

    Deterministic — no API call needed.

    Args:
        extracted: The dict returned by parse_response(), or None.
        gold:      The golden answer dict from golden[snippet_id].

    Returns:
        Integer 0–3. 3 = all fields correct. 0 = none correct or unparsable.
    """
    # A None extracted (parse failure) scores 0 immediately.
    if extracted is None:
        return 0

    score = 0

    # ── company ──────────────────────────────────────────────────────────────
    # .strip().lower() normalises whitespace and case so "OpenAI " == "openai".
    # The `or ''` guard converts None to '' before calling .strip().
    ext_company  = str(extracted.get('company')  or '').strip().lower()
    gold_company = str(gold.get('company')       or '').strip().lower()
    if ext_company == gold_company:
        score += 1

    # ── role ─────────────────────────────────────────────────────────────────
    ext_role  = str(extracted.get('role')  or '').strip().lower()
    gold_role = str(gold.get('role')       or '').strip().lower()
    if ext_role == gold_role:
        score += 1

    # ── years_experience_required ─────────────────────────────────────────────
    gold_years = gold.get('years_experience_required')
    ext_raw    = extracted.get('years_experience_required')

    if gold_years is None:
        # Correct answer is null — model must also return None/null.
        if ext_raw is None:
            score += 1
    else:
        # Correct answer is an integer.
        # int() cast handles models that return "3" (string) instead of 3.
        # If the cast fails (e.g., model returned "three"), ext_years = None.
        try:
            ext_years = int(ext_raw) if ext_raw is not None else None
        except (ValueError, TypeError):
            ext_years = None
        if ext_years == int(gold_years):
            score += 1

    return score


# =============================================================================
# STEP 6 — HTML HELPER FUNCTIONS
# =============================================================================

def esc(text: str) -> str:
    """HTML-escape &, <, >, " — must be applied to all embedded user text."""
    return (
        text.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
    )


def messages_to_html(msgs: list[dict]) -> str:
    """Render a messages list as color-coded chat bubbles.

    Color scheme (consistent across all scripts):
      system    → yellow  — persistent instructions/persona
      user      → blue    — your prompt
      assistant → green   — model reply
    """
    parts = []
    for msg in msgs:
        role    = msg.get('role', '')
        content = msg.get('content', '')
        if role == 'system':
            bg, label, tc = '#fff3cd', 'SYSTEM',    '#856404'
        elif role == 'user':
            bg, label, tc = '#cfe2ff', 'USER',      '#084298'
        else:
            bg, label, tc = '#d1e7dd', 'ASSISTANT', '#0a3622'
        parts.append(
            f'<div style="background:{bg};border-radius:6px;padding:8px 12px;margin-bottom:6px;">'
            f'<span style="font-size:.7em;font-weight:700;color:{tc};text-transform:uppercase;'
            f'letter-spacing:.06em">{label}</span>'
            f'<pre style="margin:4px 0 0;white-space:pre-wrap;font-size:.82em;'
            f'font-family:Menlo,Consolas,monospace;color:#212529">{esc(content)}</pre>'
            f'</div>'
        )
    return ''.join(parts)


def strategy_badge(name: str) -> str:
    """Return a colored pill <span> for a strategy name."""
    bg = STRATEGY_COLORS.get(name, '#aaa')
    return (
        f'<span style="background:{bg};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:.8em;font-weight:700">{name}</span>'
    )


def acc_color(acc: int) -> str:
    """Return a hex color reflecting the accuracy score level.

    3   → green  (perfect)
    1–2 → orange (partial)
    0   → red    (failed)
    """
    return '#198754' if acc == 3 else ('#fd7e14' if acc >= 1 else '#dc3545')


def field_cell(ext_val, gold_val) -> str:
    """Return an HTML <td> showing extracted value vs gold with a ✔/✘ icon.

    The icon is green ✔ if the values match (case-insensitive, trimmed),
    red ✘ otherwise. The gold value is shown in smaller grey text below.

    Args:
        ext_val:  The extracted value (may be None).
        gold_val: The correct golden value (may be None).

    Returns:
        A complete <td>...</td> HTML string.
    """
    # Display '—' for missing extracted values (None or absent)
    e = str(ext_val  if ext_val  is not None else '—')
    g = str(gold_val if gold_val is not None else 'null')
    match = e.strip().lower() == g.strip().lower()
    icon  = '✔' if match else '✘'
    color = '#198754' if match else '#dc3545'
    return (
        f'<td>'
        f'<span style="color:{color};font-weight:700">{icon}</span> {esc(e)}'
        f'<br><small style="color:#888">gold: {esc(g)}</small>'
        f'</td>'
    )


# =============================================================================
# STEP 7 — BUILD HTML REPORT
# =============================================================================

def build_html_report(scored: list[dict]) -> str:
    """Assemble the complete 3-section HTML report from scored results.

    Section 1 — Strategy Accuracy Summary
        A pandas groupby produces mean accuracy and parse rate per strategy.
        Rendered as a table with inline progress bars. Best strategy is bold.

    Section 2 — Per-Row Scored Detail
        All 40 rows sorted by (snippet_id, strategy).
        Each row shows ✔/✘ per field via field_cell(), plus total accuracy.

    Section 3 — Full Prompt & Response Trace
        Grouped by snippet. For each snippet:
          • Snippet ID badge + gold answer on one line
          • Job posting text in a left-bordered quote block
          • For each strategy: prompt bubbles → raw response → parsed result

    Args:
        scored: List of 40 fully scored dicts (includes 'extracted',
                'parse_success', 'accuracy' keys added in main()).

    Returns:
        Complete HTML document string.
    """

    # ── Section 1: summary table via pandas ──────────────────────────────────
    df = pd.DataFrame(scored)

    # groupby('strategy') groups the 40 rows into 4 groups of 10.
    # agg() computes mean accuracy and mean parse_success per group.
    summary = df.groupby('strategy').agg(
        accuracy=('accuracy', 'mean'),
        parse_success=('parse_success', 'mean'),
    ).round(3)
    summary.columns = ['Accuracy (mean/3)', 'Parse Rate']

    best_acc = summary['Accuracy (mean/3)'].max()
    summary_rows = ''
    for strat, row in summary.iterrows():
        # Convert 0–3 accuracy to a 0–100% width for the CSS bar
        pct  = row['Accuracy (mean/3)'] / 3.0 * 100
        bold = 'font-weight:700;' if row['Accuracy (mean/3)'] == best_acc else ''
        # Green bar for accuracy
        acc_bar = (
            f'<div style="background:#e9ecef;border-radius:4px;height:10px;'
            f'width:120px;display:inline-block;vertical-align:middle">'
            f'<div style="background:#198754;width:{pct:.1f}%;height:100%;border-radius:4px"></div>'
            f'</div> <span style="font-size:.8em;color:#555">{row["Accuracy (mean/3)"]:.3f}</span>'
        )
        # Purple bar for parse rate (0–100%)
        pr_pct  = row['Parse Rate'] * 100
        pr_bar  = (
            f'<div style="background:#e9ecef;border-radius:4px;height:10px;'
            f'width:120px;display:inline-block;vertical-align:middle">'
            f'<div style="background:#6f42c1;width:{pr_pct:.1f}%;height:100%;border-radius:4px"></div>'
            f'</div> <span style="font-size:.8em;color:#555">{row["Parse Rate"]:.3f}</span>'
        )
        summary_rows += (
            f'<tr><td>{strategy_badge(strat)}</td>'
            f'<td style="{bold}">{acc_bar}</td>'
            f'<td>{pr_bar}</td></tr>'
        )

    # ── Section 2: per-row detail table ──────────────────────────────────────
    detail_rows = ''
    for row in sorted(scored, key=lambda r: (r['snippet_id'], r['strategy'])):
        gold = golden[row['snippet_id']]
        ext  = row['extracted'] or {}
        ac   = row['accuracy']
        detail_rows += (
            f'<tr>'
            f'<td style="white-space:nowrap;font-family:monospace;font-size:.82em">'
            f'{esc(row["snippet_id"])}</td>'
            f'<td>{strategy_badge(row["strategy"])}</td>'
            # field_cell() produces ✔/✘ + gold value for each field
            f'{field_cell(ext.get("company"),                   gold.get("company"))}'
            f'{field_cell(ext.get("role"),                      gold.get("role"))}'
            f'{field_cell(ext.get("years_experience_required"), gold.get("years_experience_required"))}'
            f'<td style="text-align:center;color:{acc_color(ac)};font-weight:700">{ac}/3</td>'
            f'</tr>'
        )

    # ── Section 3: full trace grouped by snippet ──────────────────────────────
    # trace_map[snippet_id][strategy_name] = scored_row
    # Grouping lets us show all 4 strategies for one snippet together.
    trace_map: dict[str, dict[str, dict]] = {}
    for row in scored:
        trace_map.setdefault(row['snippet_id'], {})[row['strategy']] = row

    trace_html = ''
    for sid in sorted(trace_map.keys()):
        gold         = golden[sid]
        snippet_text = next(s['snippet'] for s in snippets if s['id'] == sid)

        strat_blocks = ''
        for strat_name in STRATEGIES:
            row = trace_map[sid].get(strat_name)
            if row is None:
                continue  # should not happen, but guard against missing data
            ac      = row['accuracy']
            ext_str = (
                json.dumps(row['extracted'], indent=2)
                if row['extracted'] else 'null (parse failed)'
            )
            strat_blocks += f"""
            <div style="border:1px solid #dee2e6;border-radius:8px;margin-bottom:14px;overflow:hidden;">
              <!-- Strategy header: badge + accuracy score -->
              <div style="background:#343a40;padding:8px 14px;display:flex;align-items:center;gap:12px;">
                {strategy_badge(strat_name)}
                <span style="color:#adb5bd;font-size:.8em;">
                  Accuracy: <strong style="color:{acc_color(ac)}">{ac}/3</strong>
                </span>
              </div>
              <div style="padding:12px 14px;">
                <!-- Prompt sent -->
                <p style="margin:0 0 6px;font-size:.75em;font-weight:700;color:#6c757d;
                   text-transform:uppercase;letter-spacing:.06em">Prompt sent to model</p>
                {messages_to_html(row['messages'])}
                <!-- Raw response -->
                <p style="margin:12px 0 6px;font-size:.75em;font-weight:700;color:#6c757d;
                   text-transform:uppercase;letter-spacing:.06em">Raw model response</p>
                <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:10px 12px;">
                  <pre style="margin:0;white-space:pre-wrap;font-size:.82em;
                       font-family:Menlo,Consolas,monospace">{esc(row['raw_response'])}</pre>
                </div>
                <!-- Parsed result -->
                <p style="margin:10px 0 4px;font-size:.75em;font-weight:700;color:#6c757d;
                   text-transform:uppercase;letter-spacing:.06em">Parsed result</p>
                <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:10px 12px;">
                  <pre style="margin:0;white-space:pre-wrap;font-size:.82em;
                       font-family:Menlo,Consolas,monospace">{esc(ext_str)}</pre>
                </div>
              </div>
            </div>"""

        trace_html += f"""
        <div style="margin-bottom:40px;">
          <!-- Snippet header: ID badge + gold answer summary -->
          <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px;">
            <span style="background:#495057;color:#fff;padding:3px 10px;border-radius:6px;
                  font-size:.8em;font-weight:700;font-family:monospace">{esc(sid)}</span>
            <span style="font-size:.82em;color:#6c757d;">
              Gold: company=<strong>{esc(str(gold['company']))}</strong>
              &nbsp; role=<strong>{esc(str(gold['role']))}</strong>
              &nbsp; years=<strong>{esc(str(gold['years_experience_required']))}</strong>
            </span>
          </div>
          <!-- Job posting text in left-bordered quote block -->
          <div style="background:#f8f9fa;border-left:4px solid #6c757d;padding:8px 12px;
               margin-bottom:12px;border-radius:0 4px 4px 0;">
            <p style="margin:0 0 4px;font-size:.75em;font-weight:700;color:#6c757d;
               text-transform:uppercase;letter-spacing:.06em">Job posting snippet</p>
            <p style="margin:0;font-size:.88em">{esc(snippet_text)}</p>
          </div>
          {strat_blocks}
        </div>"""

    # ── Assemble the complete HTML document ───────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Intermediate Example — Async + Accuracy Scoring</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body  {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
             background: #f8f9fa; color: #212529; margin: 0; padding: 32px; }}
    h1    {{ font-size: 1.5rem; margin-bottom: 4px; }}
    h2    {{ font-size: 1rem; color: #495057; margin: 28px 0 10px;
             border-bottom: 2px solid #dee2e6; padding-bottom: 5px; }}
    .meta {{ font-size:.82em; color:#6c757d; margin-bottom:24px; }}
    .badge {{ background:#198754; color:#fff; padding:2px 10px; border-radius:12px;
              font-size:.8em; font-weight:700; }}
    /* Green left-border callout explaining what's new in this script */
    .callout {{ background:#d1e7dd; border-left:4px solid #198754; padding:10px 14px;
                border-radius:0 6px 6px 0; font-size:.85em; margin-bottom:20px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px;
             overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.08); margin-bottom:32px; }}
    th    {{ background:#343a40; color:#fff; padding:10px 14px; text-align:left;
             font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }}
    td    {{ padding:9px 14px; font-size:.88rem; border-bottom:1px solid #f1f3f5; vertical-align:middle; }}
    tr:last-child td {{ border-bottom:none; }}
    tr:hover td {{ background:#f8f9fa; }}
    /* .toc: inline table-of-contents panel */
    .toc  {{ background:#fff; border-radius:8px; padding:16px 20px;
             box-shadow:0 1px 4px rgba(0,0,0,.08); margin-bottom:32px; display:inline-block; }}
    .toc a {{ color:#0d6efd; text-decoration:none; font-size:.9em; }}
    .toc a:hover {{ text-decoration:underline; }}
    .toc li {{ margin-bottom:4px; }}
  </style>
</head>
<body>
  <h1>Intermediate Example <span class="badge">Async + Accuracy Scoring</span></h1>
  <div class="meta">
    Scope: <strong>{len(snippets)} snippets × {len(STRATEGIES)} strategies
    = {len(snippets)*len(STRATEGIES)} calls</strong>
    &nbsp;|&nbsp; Model: <strong>{MODEL}</strong>
    &nbsp;|&nbsp; Scoring: <strong>deterministic accuracy (no LLM judge)</strong>
  </div>

  <div class="callout">
    <strong>New vs. simple_example_02:</strong> now running on the full dataset,
    matching each result against golden answers, and computing per-field accuracy.
    Cost tracking and LLM-as-judge are the remaining steps in
    <code>assignment_template.py</code>.
  </div>

  <!-- Table of contents with anchor links -->
  <nav class="toc">
    <strong style="font-size:.85em;text-transform:uppercase;letter-spacing:.05em">Contents</strong>
    <ol style="margin:8px 0 0;padding-left:20px;">
      <li><a href="#summary">Strategy Accuracy Summary</a></li>
      <li><a href="#detail">Per-Row Scored Detail</a></li>
      <li><a href="#trace">Full Prompt &amp; Response Trace</a></li>
    </ol>
  </nav>

  <!-- ══════════════════════════════════════════════════ SECTION 1 -->
  <h2 id="summary">1 · Strategy Accuracy Summary</h2>
  <table>
    <thead>
      <tr><th>Strategy</th><th>Accuracy (mean / 3)</th><th>Parse Rate</th></tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>

  <!-- ══════════════════════════════════════════════════ SECTION 2 -->
  <h2 id="detail">2 · Per-Row Scored Detail</h2>
  <table>
    <thead>
      <tr>
        <th>Snippet ID</th><th>Strategy</th>
        <th>Company</th><th>Role</th><th>Years Exp.</th><th>Accuracy</th>
      </tr>
    </thead>
    <tbody>{detail_rows}</tbody>
  </table>

  <!-- ══════════════════════════════════════════════════ SECTION 3 -->
  <h2 id="trace">3 · Full Prompt &amp; Response Trace</h2>
  <p style="font-size:.85em;color:#6c757d;margin-bottom:24px;">
    Every snippet: job posting text, exact prompt per strategy,
    verbatim model response, and parsed result.
  </p>
  {trace_html}
</body>
</html>
"""


# =============================================================================
# ASYNC ENTRY POINT
# =============================================================================

async def main():
    """Orchestrate: run all 40 calls, score, write report."""
    print(
        f'STEP 3 — Running {len(snippets)} × {len(STRATEGIES)} = '
        f'{len(snippets)*len(STRATEGIES)} async calls...'
    )
    raw_results = await run_all()
    print(f'  All {len(raw_results)} calls complete. Scoring...')
    print()

    # Post-processing: parse + score each result.
    # These operations are synchronous and instant — no async needed.
    scored = []
    for row in raw_results:
        gold                = golden[row['snippet_id']]
        extracted           = parse_response(row['raw_response'])
        row['extracted']    = extracted
        row['parse_success'] = extracted is not None   # used in summary table
        row['accuracy']     = score_accuracy(extracted, gold)
        scored.append(row)

    # Quick terminal summary
    df      = pd.DataFrame(scored)
    summary = df.groupby('strategy')['accuracy'].mean().round(3)
    print('STEP 4 — Accuracy by strategy (mean / 3):')
    print(summary.to_string())
    print()

    out_path = Path('intermediate_example_report.html')
    out_path.write_text(build_html_report(scored), encoding='utf-8')
    print(f'STEP 5 — HTML report written')
    print(f'  Open in browser: {out_path.resolve()}')
    print('─' * 60)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    asyncio.run(main())
