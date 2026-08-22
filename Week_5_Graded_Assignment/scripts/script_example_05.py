# Setup
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

# Make sure your OPENAI_API_KEY is set in the environment
assert os.environ.get('OPENAI_API_KEY'), 'Set OPENAI_API_KEY first'

client = AsyncOpenAI()

MODEL = 'gpt-4o-mini'
JUDGE_MODEL = 'gpt-4o'
TEMPERATURE = 0.0

# Cost rates ($ per token) — from W4 cost.py
RATES = {
    'gpt-4o-mini': {'in': 0.15 / 1_000_000, 'out': 0.60 / 1_000_000},
    'gpt-4o':      {'in': 2.50 / 1_000_000, 'out': 10.00 / 1_000_000},
}

# ---------------------------------------------------------------------------
# Results directory + Logging setup
# ---------------------------------------------------------------------------
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # create it if it doesn't exist yet

logger = logging.getLogger('mp1_prompt_lab')
logger.setLevel(logging.DEBUG)

if not logger.handlers:  # avoid duplicate handlers if this module is re-imported
    log_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    file_handler = logging.FileHandler(RESULTS_DIR / 'mp1.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

logger.info('=' * 70)
logger.info('MP1 Prompt Lab run started')
logger.info(f'Model: {MODEL} | Judge model: {JUDGE_MODEL} | Temperature: {TEMPERATURE}')

print('Setup complete.')


# ---------------------------------------------------------------------------
# Step 1 — Load the data
# ---------------------------------------------------------------------------

DATA_DIR = Path('data')   # adjust if your folder layout differs

logger.info(f'Loading snippets from {DATA_DIR / "job_snippets.jsonl"}')
snippets = [json.loads(line) for line in (DATA_DIR / 'job_snippets.jsonl').read_text().splitlines() if line.strip()]

logger.info(f'Loading golden set from {DATA_DIR / "golden_set.jsonl"}')
golden = {row['id']: row for row in (json.loads(line) for line in (DATA_DIR / 'golden_set.jsonl').read_text().splitlines() if line.strip())}

logger.info(f'Loaded {len(snippets)} snippets, {len(golden)} golden entries.')
print(f'Loaded {len(snippets)} snippets, {len(golden)} golden entries.')
print('Sample snippet:', snippets[0])

# ---------------------------------------------------------------------------
# Step 2 — Four prompt strategies
# ---------------------------------------------------------------------------
# Each function accepts a job-posting snippet string and returns a `messages`
# list ready to pass to client.chat.completions.create().
# ---------------------------------------------------------------------------

def prompt_zero_shot(snippet_text: str) -> list[dict]:
    """Strategy 1 — zero-shot. Just ask, no examples, no persona."""
    return [
        {
            'role': 'user',
            'content': (
                f'Extract the following three fields from this job posting and '
                f'return them as a JSON object with exactly these keys: '
                f'"company", "role", "years_experience_required" (integer, or null if not stated).\n\n'
                f'Job posting:\n{snippet_text}'
            ),
        }
    ]


def prompt_few_shot(snippet_text: str) -> list[dict]:
    """Strategy 2 — few-shot. Include 2-3 worked examples in the prompt."""
    examples = """Example 1:
                Job posting: "TechCorp is hiring a Backend Engineer. We need someone with 3+ years of Python experience."
                Output: {"company": "TechCorp", "role": "Backend Engineer", "years_experience_required": 3}

                Example 2:
                Job posting: "DataWorks Ltd is looking for a Data Scientist. Fresh graduates are welcome — no experience required."
                Output: {"company": "DataWorks Ltd", "role": "Data Scientist", "years_experience_required": 0}

                Example 3:
                Job posting: "Omega Systems needs a Principal Architect. We hire on demonstrated impact and do not list a years requirement."
                Output: {"company": "Omega Systems", "role": "Principal Architect", "years_experience_required": null}"""

    return [
        {
            'role': 'user',
            'content': (
                f'Extract "company", "role", and "years_experience_required" from a job posting. '
                f'Return a JSON object with exactly those three keys. '
                f'Use an integer for years (take the minimum if a range is given), or null if no years are stated.\n\n'
                f'{examples}\n\n'
                f'Now extract from this job posting:\n{snippet_text}\n'
                f'Output:'
            ),
        }
    ]


def prompt_structured(snippet_text: str) -> list[dict]:
    """Strategy 3 — structured / role-based. Persona + explicit JSON schema."""
    return [
        {
            'role': 'system',
            'content': (
                'You are an expert technical recruiter and data extraction specialist. '
                'Your job is to read job posting text and extract structured information with high precision.\n\n'
                'Always return a JSON object with exactly these fields:\n'
                '  "company"                  (string)  — the name of the hiring company\n'
                '  "role"                     (string)  — the exact job title\n'
                '  "years_experience_required" (integer or null) — the minimum years of experience required; '
                'use the lower bound if a range is given; use null if no years requirement is stated\n\n'
                'Rules:\n'
                '- Output raw JSON only. No markdown fences, no explanation.\n'
                '- Never fabricate information not present in the text.\n'
                '- If years are written as a word (e.g. "three"), convert to an integer (3).'
            ),
        },
        {
            'role': 'user',
            'content': f'Extract the structured fields from this job posting:\n\n{snippet_text}',
        },
    ]


def prompt_cot(snippet_text: str) -> list[dict]:
    """Strategy 4 — chain-of-thought. Ask the model to reason before answering."""
    return [
        {
            'role': 'user',
            'content': (
                f'Read the following job posting carefully and extract three fields: '
                f'"company", "role", and "years_experience_required".\n\n'
                f'Job posting:\n{snippet_text}\n\n'
                f'Think step by step:\n'
                f'1. Identify the company name.\n'
                f'2. Identify the job title/role.\n'
                f'3. Find any mention of required years of experience. '
                f'If a range is given, take the minimum. '
                f'If years are spelled out as a word, convert to an integer. '
                f'If no years requirement is stated at all, use null.\n'
                f'4. Output a JSON object with exactly these keys: '
                f'"company", "role", "years_experience_required".\n\n'
                f'Show your reasoning, then end your response with the JSON object.'
            ),
        }
    ]


STRATEGIES = {
    'zero_shot':  prompt_zero_shot,
    'few_shot':   prompt_few_shot,
    'structured': prompt_structured,
    'cot':        prompt_cot,
}


# ---------------------------------------------------------------------------
# Step 3 — Async batching
# ---------------------------------------------------------------------------

def parse_response(text: str) -> dict | None:
    """Try to parse a JSON object out of the model's response.

    Handles:
    - Bare JSON
    - JSON wrapped in ```json ... ``` or ``` ... ``` fences
    - JSON embedded anywhere in a longer CoT response
    Returns None if no valid JSON object is found.
    """
    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text).strip()
    cleaned = cleaned.replace('```', '').strip()

    # Try the cleaned text directly first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back: find the last {...} block in the text (handles CoT reasoning + trailing JSON)
    matches = list(re.finditer(r'\{[^{}]+\}', text, re.DOTALL))
    for match in reversed(matches):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue

    return None


async def run_one(strategy_name: str, snippet: dict) -> dict:
    """Run one strategy on one snippet. Returns a dict with all captured fields."""
    logger.debug(f'[{snippet["id"]}] Starting strategy "{strategy_name}"')
    messages = STRATEGIES[strategy_name](snippet['snippet'])

    t_start = time.monotonic()
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )
    except Exception as e:
        logger.error(f'[{snippet["id"]}] Strategy "{strategy_name}" API call FAILED: {type(e).__name__}: {e}')
        raise

    latency_s = time.monotonic() - t_start

    raw_text = resp.choices[0].message.content or ''

    # Cost from token usage
    usage = resp.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    cost_usd = (
        prompt_tokens * RATES[MODEL]['in']
        + completion_tokens * RATES[MODEL]['out']
    )

    parsed = parse_response(raw_text)

    if parsed is None:
        logger.warning(f'[{snippet["id"]}] Strategy "{strategy_name}" — FAILED to parse JSON from response')
    else:
        logger.debug(f'[{snippet["id"]}] Strategy "{strategy_name}" — parsed OK, latency={latency_s:.2f}s, cost=${cost_usd:.5f}')

    return {
        'strategy':    strategy_name,
        'snippet_id':  snippet['id'],
        'raw_response': raw_text,
        'extracted':   parsed,
        'cost_usd':    cost_usd,
        'latency_s':   latency_s,
    }


async def run_all() -> list[dict]:
    """Run all 10 snippets × 4 strategies = 40 calls in parallel."""
    logger.info(f'Dispatching {len(snippets) * len(STRATEGIES)} LLM calls ({len(snippets)} snippets × {len(STRATEGIES)} strategies)')
    tasks = [
        run_one(strategy_name, snippet)
        for snippet in snippets
        for strategy_name in STRATEGIES
    ]
    results = await asyncio.gather(*tasks)
    logger.info(f'All {len(results)} LLM calls completed')
    return results


# ---------------------------------------------------------------------------
# Step 4 — Scoring
# ---------------------------------------------------------------------------

def score_accuracy(extracted: dict | None, gold: dict) -> int:
    """Count how many of the 3 fields match the golden set (0–3).

    String fields: case-insensitive, whitespace-trimmed.
    years_experience_required: compare as integers; null == null.
    """
    if extracted is None:
        return 0

    score = 0

    # --- company ---
    ext_company = str(extracted.get('company', '') or '').strip().lower()
    gold_company = str(gold.get('company', '') or '').strip().lower()
    if ext_company and ext_company == gold_company:
        score += 1

    # --- role ---
    ext_role = str(extracted.get('role', '') or '').strip().lower()
    gold_role = str(gold.get('role', '') or '').strip().lower()
    if ext_role and ext_role == gold_role:
        score += 1

    # --- years_experience_required ---
    gold_years = gold.get('years_experience_required')
    ext_years_raw = extracted.get('years_experience_required')

    if gold_years is None:
        # Correct answer is null — model should return null / None
        if ext_years_raw is None:
            score += 1
    else:
        # Convert to int for comparison (handles string numbers from some models)
        try:
            ext_years = int(ext_years_raw) if ext_years_raw is not None else None
        except (ValueError, TypeError):
            ext_years = None
        if ext_years == int(gold_years):
            score += 1

    return score


async def score_llm_judge(snippet_text: str, extracted: dict | None, gold: dict) -> int:
    """Use gpt-4o as a judge. Returns an integer 1–4.

    Rubric:
      4 — all three fields correct
      3 — two of three correct, no fabricated data
      2 — one of three correct, or a field was fabricated
      1 — none correct, unparsable, or response is null
    """
    extracted_str = json.dumps(extracted) if extracted is not None else 'null (failed to parse)'
    gold_str = json.dumps({
        'company': gold['company'],
        'role': gold['role'],
        'years_experience_required': gold['years_experience_required'],
    })

    prompt = (
        f'You are a strict evaluator. A model was asked to extract three fields from a job posting.\n\n'
        f'Job posting:\n{snippet_text}\n\n'
        f'Golden (correct) answer:\n{gold_str}\n\n'
        f'Model extraction:\n{extracted_str}\n\n'
        f'Score the model extraction on a scale of 1–4 using this rubric:\n'
        f'  4 — all three fields correct\n'
        f'  3 — two of three correct, no fabricated data\n'
        f'  2 — one of three correct, or a field was fabricated\n'
        f'  1 — none correct, failed to parse, or null output\n\n'
        f'Reply with a single integer (1, 2, 3, or 4) and nothing else.'
    )

    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=5,
        )
    except Exception as e:
        logger.error(f'LLM judge call FAILED: {type(e).__name__}: {e} — defaulting score to 1')
        return 1

    raw = (resp.choices[0].message.content or '').strip()

    try:
        score = int(raw[0])
        return max(1, min(4, score))   # clamp to 1–4
    except (ValueError, IndexError):
        logger.warning('LLM judge returned a non-numeric response — defaulting score to 1')
        return 1   # default to lowest if judge response is malformed


# ---------------------------------------------------------------------------
# Main — ties everything together
# ---------------------------------------------------------------------------

async def main():
    # --- Run all 40 calls ---
    logger.info('STEP 3: Running LLM calls')
    print('\nRunning 40 LLM calls (10 snippets × 4 strategies)...')
    results = await run_all()
    print(f'Got {len(results)} results.')
    print('Sample result:', {k: v for k, v in results[0].items() if k != 'raw_response'})

    # --- Score all results ---
    logger.info('STEP 4: Scoring results (deterministic accuracy + LLM judge)')
    print('\nScoring results (accuracy + LLM judge)...')
    scored = []
    judge_tasks = []

    parse_failures = 0

    # Build judge coroutines alongside deterministic scores so we can
    # batch all 40 judge calls in one gather (faster than sequential).
    for row in results:
        gold = golden[row['snippet_id']]
        row['parse_success'] = row['extracted'] is not None
        if not row['parse_success']:
            parse_failures += 1
        row['accuracy'] = score_accuracy(row['extracted'], gold)
        judge_tasks.append(
            score_llm_judge(
                snippet_text=next(s['snippet'] for s in snippets if s['id'] == row['snippet_id']),
                extracted=row['extracted'],
                gold=gold,
            )
        )

    logger.info(f'Deterministic scoring complete. Parse failures: {parse_failures}/{len(results)}')
    logger.info(f'Dispatching {len(judge_tasks)} LLM judge calls')
    print(f'Running {len(judge_tasks)} LLM judge calls in parallel...')
    judge_scores = await asyncio.gather(*judge_tasks)
    logger.info('LLM judge calls complete')

    for row, judge_score in zip(results, judge_scores):
        row['llm_judge_score'] = judge_score
        scored.append(row)

    print(f'Scored {len(scored)} results.')

    # --- Step 5: Comparison table ---
    logger.info('STEP 5: Building comparison table')
    df = pd.DataFrame(scored)

    summary = df.groupby('strategy').agg(
        accuracy=('accuracy', 'mean'),
        parse_success=('parse_success', 'mean'),
        llm_judge_score=('llm_judge_score', 'mean'),
        cost_usd=('cost_usd', 'sum'),
        latency_s=('latency_s', 'median'),
    ).round(3)

    summary.columns = [
        'Accuracy (mean/3)',
        'Parse Rate',
        'Judge Score (mean/4)',
        'Total Cost ($)',
        'Latency p50 (s)',
    ]

    print('\n=== Strategy Comparison ===')
    print(summary.to_string())
    logger.info('Comparison table built successfully')

    # ------------------------------------------------------------------
    # Output 1 — JSON  (raw scored rows, excluding bulky raw_response)
    # ------------------------------------------------------------------
    json_path = RESULTS_DIR / 'mp1_results.json'
    json_path.write_text(
        json.dumps(
            [{k: v for k, v in row.items() if k != 'raw_response'} for row in scored],
            indent=2,
        ),
        encoding='utf-8',
    )
    logger.info(f'Saved JSON results to {json_path}')
    print(f'\nSaved JSON  → {json_path}')

    # ------------------------------------------------------------------
    # Output 2 — CSV  (flat per-row detail, easy to open in Excel)
    # ------------------------------------------------------------------
    csv_df = df.drop(columns=['raw_response'], errors='ignore').copy()
    # Flatten 'extracted' dict into separate columns for readability
    extracted_df = csv_df['extracted'].apply(
        lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series({'company': None, 'role': None, 'years_experience_required': None})
    ).rename(columns=lambda c: f'extracted_{c}')
    csv_df = pd.concat([csv_df.drop(columns=['extracted']), extracted_df], axis=1)

    csv_path = RESULTS_DIR / 'mp1_results.csv'
    csv_df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f'Saved CSV results to {csv_path}')
    print(f'Saved CSV   → {csv_path}')

    # ------------------------------------------------------------------
    # Output 3 — HTML report
    # ------------------------------------------------------------------
    html_path = RESULTS_DIR / 'mp1_report.html'
    html_path.write_text(_build_html_report(summary, scored, snippets, golden), encoding='utf-8')
    logger.info(f'Saved HTML report to {html_path}')
    print(f'Saved HTML  → {html_path}')

    logger.info('MP1 Prompt Lab run finished successfully')
    logger.info('=' * 70)

    return summary


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

def _bar(value: float, max_value: float, color: str) -> str:
    """Return an inline SVG progress bar."""
    pct = min(100.0, 100.0 * value / max_value) if max_value else 0
    return (
        f'<div style="background:#e9ecef;border-radius:4px;height:10px;width:120px;display:inline-block;vertical-align:middle">'
        f'<div style="background:{color};width:{pct:.1f}%;height:100%;border-radius:4px"></div>'
        f'</div> <span style="font-size:.8em;color:#555">{value:.3f}</span>'
    )


def _strategy_badge(name: str) -> str:
    colors = {
        'zero_shot':  '#6c757d',
        'few_shot':   '#0d6efd',
        'structured': '#198754',
        'cot':        '#fd7e14',
    }
    bg = colors.get(name, '#aaa')
    return f'<span style="background:{bg};color:#fff;padding:2px 8px;border-radius:12px;font-size:.8em;font-weight:600">{name}</span>'


def _build_html_report(summary: pd.DataFrame, scored: list[dict], snippets: list[dict], golden: dict) -> str:
    snippet_map = {s['id']: s['snippet'] for s in snippets}

    # ---- Summary table rows ----
    summary_rows = ''
    best_accuracy = summary['Accuracy (mean/3)'].max()
    best_judge    = summary['Judge Score (mean/4)'].max()
    for strat, row in summary.iterrows():
        acc_bar    = _bar(row['Accuracy (mean/3)'],   3.0, '#198754')
        judge_bar  = _bar(row['Judge Score (mean/4)'], 4.0, '#0d6efd')
        parse_bar  = _bar(row['Parse Rate'],           1.0, '#6f42c1')
        acc_bold   = ' font-weight:700;' if row['Accuracy (mean/3)'] == best_accuracy else ''
        judge_bold = ' font-weight:700;' if row['Judge Score (mean/4)'] == best_judge else ''
        summary_rows += f"""
        <tr>
          <td>{_strategy_badge(strat)}</td>
          <td style="{acc_bold}">{acc_bar}</td>
          <td style="{judge_bold}">{judge_bar}</td>
          <td>{parse_bar}</td>
          <td style="font-family:monospace">${row['Total Cost ($)']:.5f}</td>
          <td style="font-family:monospace">{row['Latency p50 (s)']:.2f}s</td>
        </tr>"""

    # ---- Detail table rows ----
    detail_rows = ''
    for row in sorted(scored, key=lambda r: (r['snippet_id'], r['strategy'])):
        gold = golden[row['snippet_id']]
        ext  = row['extracted'] or {}
        acc  = row['accuracy']
        acc_color = '#198754' if acc == 3 else ('#fd7e14' if acc >= 1 else '#dc3545')
        judge_color = '#198754' if row['llm_judge_score'] == 4 else ('#fd7e14' if row['llm_judge_score'] >= 3 else '#dc3545')

        def cell(extracted_val, gold_val):
            e = str(extracted_val or '—')
            g = str(gold_val if gold_val is not None else 'null')
            match = e.strip().lower() == g.strip().lower()
            icon = '✔' if match else '✘'
            color = '#198754' if match else '#dc3545'
            return f'<td><span style="color:{color};font-weight:700">{icon}</span> {e}<br><small style="color:#888">gold: {g}</small></td>'

        detail_rows += f"""
        <tr>
          <td style="white-space:nowrap">{row['snippet_id']}</td>
          <td>{_strategy_badge(row['strategy'])}</td>
          {cell(ext.get('company'),                  gold.get('company'))}
          {cell(ext.get('role'),                     gold.get('role'))}
          {cell(ext.get('years_experience_required'), gold.get('years_experience_required'))}
          <td style="text-align:center;color:{acc_color};font-weight:700">{acc}/3</td>
          <td style="text-align:center;color:{judge_color};font-weight:700">{row['llm_judge_score']}/4</td>
          <td style="font-family:monospace;font-size:.8em">${row['cost_usd']:.5f}</td>
          <td style="font-family:monospace;font-size:.8em">{row['latency_s']:.2f}s</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MP1 Prompt Lab — Results Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f8f9fa; color: #212529; margin: 0; padding: 24px; }}
    h1   {{ font-size: 1.6rem; margin-bottom: 4px; }}
    h2   {{ font-size: 1.1rem; color: #495057; margin: 32px 0 12px; border-bottom: 2px solid #dee2e6; padding-bottom: 6px; }}
    .meta {{ font-size: .85rem; color: #6c757d; margin-bottom: 28px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff;
             border-radius: 8px; overflow: hidden;
             box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 32px; }}
    th   {{ background: #343a40; color: #fff; padding: 10px 14px; text-align: left;
            font-size: .82rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
    td   {{ padding: 9px 14px; font-size: .88rem; border-bottom: 1px solid #f1f3f5; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f8f9fa; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: .83rem; }}
  </style>
</head>
<body>
  <h1>MP1 Prompt Lab — Results Report</h1>
  <div class="meta">
    Model: <strong>{MODEL}</strong> &nbsp;|&nbsp;
    Judge: <strong>{JUDGE_MODEL}</strong> &nbsp;|&nbsp;
    Snippets: <strong>{len(snippets)}</strong> &nbsp;|&nbsp;
    Strategies: <strong>{len(STRATEGIES)}</strong> &nbsp;|&nbsp;
    Total calls: <strong>{len(scored)}</strong>
  </div>

  <div class="legend">
    {' '.join(_strategy_badge(s) for s in STRATEGIES)}
  </div>

  <h2>Strategy Comparison Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Strategy</th>
        <th>Accuracy (mean / 3)</th>
        <th>Judge Score (mean / 4)</th>
        <th>Parse Rate</th>
        <th>Total Cost</th>
        <th>Latency p50</th>
      </tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2>Per-Row Detail</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Strategy</th>
        <th>Company</th>
        <th>Role</th>
        <th>Years Exp.</th>
        <th>Accuracy</th>
        <th>Judge</th>
        <th>Cost</th>
        <th>Latency</th>
      </tr>
    </thead>
    <tbody>{detail_rows}</tbody>
  </table>
</body>
</html>
"""


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception:
        logger.exception('MP1 Prompt Lab run FAILED with an unhandled exception')
        raise