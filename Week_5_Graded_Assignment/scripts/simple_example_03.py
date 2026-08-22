# =============================================================================
# simple_example_03.py  —  ASYNC CONCURRENT  (Step 3 of 5)
# =============================================================================
#
# LEARNING GOAL
# ─────────────
# Show why asyncio.gather() exists and what it actually buys you.
#
# This script is a minimal diff on simple_example_02:
#   • The four prompt strategy functions are IDENTICAL.
#   • parse_response(), esc(), messages_to_html(), strategy_badge() are IDENTICAL.
#   • The only substantive change is in how the four API calls are dispatched.
#
# Script 02 dispatches like this (sequential):
#   for strategy_name in STRATEGIES:
#       result = await call_strategy(strategy_name, ...)   # wait, then next
#
# Script 03 dispatches like this (concurrent):
#   tasks = [call_strategy(name, ...) for name in STRATEGIES]  # build all
#   results = await asyncio.gather(*tasks)                     # run all at once
#
# That one change turns four sequential waits into one parallel wait.
#
# WHAT IS asyncio.gather()?
# ─────────────────────────
# asyncio.gather(*coroutines) does three things:
#   1. Schedules all coroutines on the event loop at the same time.
#   2. Each coroutine runs until it hits an `await` (I/O wait).
#   3. While one coroutine is waiting for a network response, the event
#      loop runs the other coroutines — they all make progress in parallel.
#   4. gather() itself awaits until EVERY coroutine has finished.
#   5. Returns results in the SAME ORDER as the input list, regardless of
#      which API call finished first.
#
# EXECUTION TIMELINE
# ──────────────────
# Script 02 (sequential await):
#   zero_shot:  [send]───[wait ~1s]───[recv]
#   few_shot:                               [send]───[wait ~1s]───[recv]
#   structured:                                                         [send]───...
#   cot:                                                                           ...
#   Total ≈ 4 × average latency
#
# Script 03 (asyncio.gather):
#   zero_shot:  [send]───[wait ~1s]───[recv]
#   few_shot:   [send]───[wait ~1s]───[recv]   ← all launched at the same time
#   structured: [send]───[wait ~1s]───[recv]
#   cot:        [send]───[wait ~1s]───[recv]
#   Total ≈ 1 × slowest single call
#
# PROGRESSION IN THE SERIES
# ─────────────────────────
#   simple_example_01    sync client,  for loop,       sequential  (blocking)
#   simple_example_02    async client, for loop,       sequential  (async but sequential)
#   simple_example_03  ← YOU ARE HERE
#                        async client, gather(),       concurrent  (truly parallel)
#   intermediate_example async, 10 snippets × 4, accuracy scoring
#   assignment_template  async, 10×4, accuracy + LLM judge + cost + 3 outputs
#
# OUTPUT
# ──────
#   simple_example_03_report.html  — opens in any browser
#   Terminal shows all results arriving after a single wait, not four.
#
# HOW TO RUN
# ──────────
#   Windows:   $env:OPENAI_API_KEY = "sk-..."
#   Mac/Linux: export OPENAI_API_KEY="sk-..."
#   Then:      cd scripts && python simple_example_03.py
# =============================================================================

# ── Standard-library imports ─────────────────────────────────────────────────
import asyncio  # event loop, gather(), run()
import json
import os
import re
import time     # time.monotonic() for per-call latency
from pathlib import Path

# ── Third-party import ───────────────────────────────────────────────────────
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

assert os.environ.get('OPENAI_API_KEY'), (
    'OPENAI_API_KEY is not set.\n'
    'Windows:   $env:OPENAI_API_KEY = "sk-..."\n'
    'Mac/Linux: export OPENAI_API_KEY="sk-..."'
)

# Same async client as simple_example_02 — nothing changes here.
client = AsyncOpenAI()

MODEL       = 'gpt-4o-mini'
TEMPERATURE = 0.0

# scripts/ is one level below the project root; data/ lives at ../data/
DATA_DIR = Path('../data')

STRATEGY_COLORS = {
    'zero_shot':  '#6c757d',
    'few_shot':   '#0d6efd',
    'structured': '#198754',
    'cot':        '#fd7e14',
}


# =============================================================================
# STEP 1 — LOAD ONE SNIPPET
# =============================================================================
# Identical to scripts 01 and 02. One snippet, keep data volume minimal
# so the async speedup is the only new concept to focus on.

all_snippets = [
    json.loads(line)
    for line in (DATA_DIR / 'job_snippets.jsonl').read_text(encoding='utf-8').splitlines()
    if line.strip()
]
snippet = all_snippets[0]

print('─' * 60)
print('STEP 1 — Input data')
print(f'  Snippet ID  : {snippet["id"]}')
print(f'  Snippet text: {snippet["snippet"]}')
print()


# =============================================================================
# STEP 2 — FOUR PROMPT STRATEGY FUNCTIONS
# =============================================================================
# IDENTICAL to simple_example_02. These functions are synchronous —
# building a prompt string involves no I/O, so no await is needed.
# They are reproduced here so this script is self-contained.

def prompt_zero_shot(text: str) -> list[dict]:
    """Strategy 1 — Zero-shot: single user message, no examples, no persona.

    The baseline. Relies on the model's training to understand the task.
    Fastest and cheapest but most prone to edge-case failures.

    Returns: one-element list with a user message.
    """
    return [
        {
            'role': 'user',
            'content': (
                'Extract the following three fields from this job posting and '
                'return them as a JSON object with exactly these keys:\n'
                '  "company"                   — name of the hiring company (string)\n'
                '  "role"                       — exact job title (string)\n'
                '  "years_experience_required"  — minimum years required '
                '(integer, or null if not stated)\n\n'
                f'Job posting:\n{text}'
            ),
        }
    ]


def prompt_few_shot(text: str) -> list[dict]:
    """Strategy 2 — Few-shot: three worked examples before asking.

    Example 1 — explicit integer years  (3+ → 3)
    Example 2 — zero years              (fresh graduates → 0)
    Example 3 — null years              (no requirement → null)

    The trailing "Output:" cue tells the model to reply with JSON directly.

    Returns: one-element list with a (longer) user message.
    """
    examples = (
        'Example 1:\n'
        'Job posting: "TechCorp is hiring a Backend Engineer. '
        'We need someone with 3+ years of Python experience."\n'
        'Output: {"company": "TechCorp", "role": "Backend Engineer", '
        '"years_experience_required": 3}\n\n'

        'Example 2:\n'
        'Job posting: "DataWorks Ltd is looking for a Data Scientist. '
        'Fresh graduates are welcome — no experience required."\n'
        'Output: {"company": "DataWorks Ltd", "role": "Data Scientist", '
        '"years_experience_required": 0}\n\n'

        'Example 3:\n'
        'Job posting: "Omega Systems needs a Principal Architect. '
        'We hire on demonstrated impact and do not list a years requirement."\n'
        'Output: {"company": "Omega Systems", "role": "Principal Architect", '
        '"years_experience_required": null}'
    )
    return [
        {
            'role': 'user',
            'content': (
                'Extract "company", "role", and "years_experience_required" from a job posting. '
                'Return a JSON object with exactly those three keys. '
                'Use an integer for years (take the minimum if a range is given), '
                'or null if no years are stated.\n\n'
                f'{examples}\n\n'
                f'Now extract from this job posting:\n{text}\nOutput:'
            ),
        }
    ]


def prompt_structured(text: str) -> list[dict]:
    """Strategy 3 — Structured: system persona + explicit schema + hard rules.

    Two messages:
        [system] — sets persona, output schema, and hard rules permanently
        [user]   — minimal: just the posting text

    Rules in the system message reduce common failure modes: markdown fences,
    fabrication, and word-number ambiguity (e.g. "three" → 3).

    Returns: two-element list [system_message, user_message].
    """
    return [
        {
            'role': 'system',
            'content': (
                'You are an expert technical recruiter and data extraction specialist. '
                'Your job is to read job posting text and extract structured information '
                'with high precision.\n\n'
                'Always return a JSON object with exactly these three fields:\n'
                '  "company"                   (string)  — the name of the hiring company\n'
                '  "role"                       (string)  — the exact job title\n'
                '  "years_experience_required"  (integer or null) — minimum years required; '
                'use the lower bound if a range is given; use null if no requirement is stated\n\n'
                'Rules:\n'
                '- Output raw JSON only. No markdown fences, no explanation.\n'
                '- Never fabricate information not present in the posting.\n'
                '- If years are written as a word (e.g. "three"), convert to an integer (3).'
            ),
        },
        {
            'role': 'user',
            'content': f'Extract the structured fields from this job posting:\n\n{text}',
        },
    ]


def prompt_cot(text: str) -> list[dict]:
    """Strategy 4 — Chain-of-thought: numbered reasoning steps before the JSON.

    The model writes reasoning for each field before outputting JSON.
    parse_response() handles this by finding the LAST {...} block in the
    response, which is the JSON after the reasoning text.

    Trade-off: longer response → more output tokens → slightly higher cost,
    but better accuracy on ambiguous postings.

    Returns: one-element list with a user message (CoT instructions).
    """
    return [
        {
            'role': 'user',
            'content': (
                'Read the following job posting carefully and extract three fields.\n\n'
                f'Job posting:\n{text}\n\n'
                'Think step by step:\n'
                '1. Identify the company name.\n'
                '2. Identify the job title/role.\n'
                '3. Find any mention of required years of experience.\n'
                '   - If a range is given (e.g. "3–5 years"), use the minimum (3).\n'
                '   - If years are spelled as a word (e.g. "three"), convert to integer (3).\n'
                '   - If no years requirement is stated at all, use null.\n'
                '4. Output a JSON object with exactly these keys:\n'
                '   {"company": ..., "role": ..., "years_experience_required": ...}\n\n'
                'Show your reasoning for each field, then end with the JSON object.'
            ),
        }
    ]


STRATEGIES = {
    'zero_shot':  prompt_zero_shot,
    'few_shot':   prompt_few_shot,
    'structured': prompt_structured,
    'cot':        prompt_cot,
}


# =============================================================================
# STEP 3 — ASYNC CALL FUNCTION  (one strategy, one snippet)
# =============================================================================
# call_strategy() is IDENTICAL to simple_example_02's version.
# The change is entirely in how it is CALLED — see run_all_strategies() below.
#
# In script 02:  `await call_strategy(name, text)` inside a for loop
# In script 03:  all four coroutines are passed to asyncio.gather() at once
#
# The function itself doesn't need to change — the concurrency is handled
# at the call site, not inside the coroutine.

async def call_strategy(strategy_name: str, snippet_text: str) -> dict:
    """Make one async API call for one strategy. Returns a result dict.

    This coroutine is identical to the one in simple_example_02.
    The difference in this script is that four of these are launched
    simultaneously by asyncio.gather() rather than awaited one at a time.

    When Python hits the `await` on client.chat.completions.create():
      - This coroutine suspends.
      - The event loop switches to another coroutine (one of the other
        three strategy calls that are also in-flight).
      - When the HTTP response arrives, the event loop resumes this coroutine.

    Args:
        strategy_name: Key into STRATEGIES dict.
        snippet_text:  Raw job posting text to extract from.

    Returns:
        Dict with keys: strategy, messages, raw_response, elapsed_s,
        prompt_tokens, output_tokens.
    """
    messages = STRATEGIES[strategy_name](snippet_text)

    t_start = time.monotonic()

    # `await` suspends this coroutine while the HTTP request is in flight.
    # In gather() mode, the event loop runs the other three coroutines here.
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )

    elapsed = time.monotonic() - t_start

    return {
        'strategy':      strategy_name,
        'messages':      messages,
        'raw_response':  resp.choices[0].message.content or '',
        'elapsed_s':     elapsed,
        'prompt_tokens': resp.usage.prompt_tokens,
        'output_tokens': resp.usage.completion_tokens,
    }


# =============================================================================
# STEP 4 — RUN ALL STRATEGIES CONCURRENTLY  ← THE KEY CHANGE
# =============================================================================

async def run_all_strategies(snippet_text: str) -> list[dict]:
    """Launch all four strategy calls at the same time using asyncio.gather().

    This is the single function that differs from simple_example_02.
    Script 02 equivalent (sequential):
        results = []
        for name in STRATEGIES:
            result = await call_strategy(name, snippet_text)  # one at a time
            results.append(result)

    This script (concurrent):
        tasks = [call_strategy(name, snippet_text) for name in STRATEGIES]
        results = await asyncio.gather(*tasks)                # all at once

    How gather() works step by step:
        1. The list comprehension creates four coroutine objects.
           None of them have started running yet — they are just objects.
        2. asyncio.gather(*tasks) schedules all four on the event loop.
        3. All four coroutines start immediately and each runs until its
           first `await` (the HTTP request).
        4. While any coroutine is waiting for a network response, the event
           loop runs the others. All four are in-flight simultaneously.
        5. gather() awaits until every coroutine has completed.
        6. Returns a tuple of results in the SAME ORDER as `tasks` —
           not in order of completion.

    Result ordering guarantee:
        gather() preserves input order regardless of which API call finishes
        first. results[0] is always zero_shot, results[1] always few_shot,
        etc. — even if few_shot happened to return before zero_shot.

    Args:
        snippet_text: Raw job posting text, passed to all 4 strategies.

    Returns:
        List of 4 result dicts in STRATEGIES insertion order.
    """
    # Step 1: create all four coroutine objects.
    # This line does NOT start any API calls — it only creates the objects.
    tasks = [
        call_strategy(name, snippet_text)
        for name in STRATEGIES  # zero_shot, few_shot, structured, cot
    ]

    # Step 2: pass all four to gather(). This starts them ALL simultaneously.
    # The `*` unpacks the list: gather(task0, task1, task2, task3).
    # gather() returns a tuple; list() converts it for convenience.
    results = await asyncio.gather(*tasks)

    return list(results)


# =============================================================================
# STEP 5 — PARSE RESPONSE HELPER
# =============================================================================

def parse_response(text: str) -> dict | None:
    """Extract a Python dict from the model's raw response text.

    Identical to simple_example_02. Three-step fallback:
        Step 1 — strip fences, try json.loads()
        Step 2 — regex find last {...} block, try json.loads()
        Step 3 — return None

    The LAST {...} block is used because CoT responses put JSON at the end,
    after the reasoning text.
    """
    cleaned = re.sub(r'```(?:json)?\s*', '', text).strip()
    cleaned = cleaned.replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    matches = list(re.finditer(r'\{[^{}]+\}', text, re.DOTALL))
    for match in reversed(matches):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue

    return None


# =============================================================================
# STEP 6 — HTML REPORT HELPERS
# =============================================================================
# Identical to simple_example_02 — no changes needed.

def esc(text: str) -> str:
    """HTML-escape &, <, >, " — must be applied to all user-supplied text."""
    return (
        text.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
    )


def messages_to_html(msgs: list[dict]) -> str:
    """Render a messages list as color-coded chat bubbles.

    system → yellow, user → blue, assistant → green.
    Consistent color scheme across all scripts in this series.
    """
    bubbles = []
    for msg in msgs:
        role    = msg.get('role', '')
        content = msg.get('content', '')
        if role == 'system':
            bg, label, tc = '#fff3cd', 'SYSTEM',    '#856404'
        elif role == 'user':
            bg, label, tc = '#cfe2ff', 'USER',      '#084298'
        else:
            bg, label, tc = '#d1e7dd', 'ASSISTANT', '#0a3622'
        bubbles.append(
            f'<div style="background:{bg};border-radius:6px;padding:8px 12px;margin-bottom:6px;">'
            f'<span style="font-size:.7em;font-weight:700;color:{tc};'
            f'text-transform:uppercase;letter-spacing:.06em">{label}</span>'
            f'<pre style="margin:4px 0 0;white-space:pre-wrap;font-size:.82em;'
            f'font-family:Menlo,Consolas,monospace;color:#212529">{esc(content)}</pre>'
            f'</div>'
        )
    return ''.join(bubbles)


def strategy_badge(name: str) -> str:
    """Return a colored pill <span> for a strategy name."""
    bg = STRATEGY_COLORS.get(name, '#aaa')
    return (
        f'<span style="background:{bg};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:.8em;font-weight:700">{name}</span>'
    )


def build_html_report(snippet: dict, results: list[dict]) -> str:
    """Assemble the complete HTML report.

    Structure:
        Header  — scope, model, dispatch method
        Callout — explains asyncio.gather() concurrency
        Compare — shows the exact code diff vs simple_example_02
        Section 1 — Input job posting
        Section 2 — One card per strategy:
                     badge + tokens + latency header
                     prompt bubbles → raw response → parsed result

    The report header badges each call with "concurrent via gather()"
    to make it visually distinct from the script 02 report.

    Args:
        snippet: Full snippet dict (id + text).
        results: List of 4 result dicts from run_all_strategies().

    Returns:
        Complete HTML document string.
    """
    strategy_cards = ''
    for row in results:
        ext_str = (
            json.dumps(row['extracted'], indent=2)
            if row['extracted'] is not None
            else 'null  ← parse_response() found no valid JSON in the response above'
        )
        strategy_cards += f"""
    <!-- ── Card: {row['strategy']} ── -->
    <div style="border:1px solid #dee2e6;border-radius:8px;margin-bottom:28px;overflow:hidden;">

      <!-- Header: strategy badge + token counts + latency + dispatch label -->
      <div style="background:#343a40;padding:10px 16px;display:flex;align-items:center;
                  gap:14px;flex-wrap:wrap;">
        {strategy_badge(row['strategy'])}
        <span style="color:#adb5bd;font-size:.8em;">
          {row['prompt_tokens']} prompt tokens
          &nbsp;+&nbsp; {row['output_tokens']} completion tokens
          &nbsp;|&nbsp; {row['elapsed_s']:.2f}s
          &nbsp;|&nbsp; <em>concurrent via gather()</em>
        </span>
      </div>

      <div style="padding:14px 18px;">

        <p style="margin:0 0 6px;font-size:.75em;font-weight:700;color:#6c757d;
           text-transform:uppercase;letter-spacing:.06em">Prompt sent to model</p>
        {messages_to_html(row['messages'])}

        <p style="margin:14px 0 6px;font-size:.75em;font-weight:700;color:#6c757d;
           text-transform:uppercase;letter-spacing:.06em">Raw model response</p>
        <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:10px 12px;">
          <pre style="margin:0;white-space:pre-wrap;font-size:.82em;
               font-family:Menlo,Consolas,monospace">{esc(row['raw_response'])}</pre>
        </div>

        <p style="margin:14px 0 6px;font-size:.75em;font-weight:700;color:#6c757d;
           text-transform:uppercase;letter-spacing:.06em">Parsed result</p>
        <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:10px 12px;">
          <pre style="margin:0;white-space:pre-wrap;font-size:.82em;
               font-family:Menlo,Consolas,monospace">{esc(ext_str)}</pre>
        </div>

      </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Simple Example 03 — Async Concurrent</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body  {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
             background: #f8f9fa; color: #212529; margin: 0; padding: 32px; max-width: 980px; }}
    h1    {{ font-size: 1.5rem; margin-bottom: 4px; }}
    h2    {{ font-size: 1rem; color: #495057; margin: 32px 0 12px;
             border-bottom: 2px solid #dee2e6; padding-bottom: 5px; }}
    .badge  {{ background:#198754; color:#fff; padding:2px 10px; border-radius:12px;
               font-size:.8em; font-weight:700; }}
    .card   {{ background:#fff; border-radius:8px; padding:16px 20px;
               box-shadow:0 1px 4px rgba(0,0,0,.08); margin-bottom:20px; }}
    .snippet-text {{ background:#f8f9fa; border-left:4px solid #6c757d;
                     padding:10px 14px; border-radius:0 4px 4px 0;
                     font-size:.9em; white-space:pre-wrap; }}
    .meta   {{ font-size:.82em; color:#6c757d; margin-bottom:28px; }}
    .tag    {{ display:inline-block; background:#e9ecef; color:#495057;
               padding:1px 8px; border-radius:4px; font-size:.78em;
               font-family:monospace; margin-right:4px; }}
    /* Green callout — gather() is the new thing */
    .callout {{ background:#d1e7dd; border-left:4px solid #198754;
                padding:10px 14px; border-radius:0 6px 6px 0;
                font-size:.85em; margin-bottom:20px; }}
    /* Code diff box */
    .diff {{ background:#212529; color:#f8f9fa; border-radius:6px;
             padding:12px 16px; font-family:Menlo,Consolas,monospace;
             font-size:.82em; margin-bottom:20px; white-space:pre; }}
    .diff .old {{ color:#f1aeb5; }}   /* red  — removed lines */
    .diff .new {{ color:#a3cfbb; }}   /* green — added lines  */
    .diff .neutral {{ color:#adb5bd; }}
  </style>
</head>
<body>
  <h1>Simple Example 03 <span class="badge">Async Concurrent</span></h1>
  <div class="meta">
    Scope: <strong>1 snippet × 4 strategies (async, concurrent)</strong>
    &nbsp;|&nbsp; Model: <strong>{MODEL}</strong>
    &nbsp;|&nbsp; Client: <strong>AsyncOpenAI</strong>
    &nbsp;|&nbsp; Dispatch: <strong>asyncio.gather() — all 4 at once</strong>
  </div>

  <!-- Key concept callout -->
  <div class="callout">
    <strong>What's new vs simple_example_02:</strong>
    <code>asyncio.gather()</code> launches all four coroutines simultaneously.
    While one coroutine is waiting for a network response, the event loop runs
    the others. Total time ≈ slowest single call — not the sum of all four.
    <br><br>
    The strategy functions, <code>call_strategy()</code>, <code>parse_response()</code>,
    and all HTML helpers are identical to script 02.
    The <em>only</em> change is how <code>call_strategy()</code> is invoked.
  </div>

  <!-- Exact code diff between script 02 and script 03 -->
  <div class="diff"><span class="neutral">  # script 02 — sequential (await in a for loop)
</span><span class="old">- results = []
- for name in STRATEGIES:
-     result = await call_strategy(name, snippet['snippet'])
-     results.append(result)
</span>
<span class="neutral">  # script 03 — concurrent (asyncio.gather)
</span><span class="new">+ tasks = [call_strategy(name, snippet['snippet']) for name in STRATEGIES]
+ results = await asyncio.gather(*tasks)</span></div>

  <!-- ═══════════════════════════════════════════════════ SECTION 1 -->
  <h2>1 · Input Data</h2>
  <div class="card">
    <p style="margin:0 0 6px;">
      <span class="tag">{esc(snippet['id'])}</span>
      Job posting text — the same text is sent to all four strategies simultaneously:
    </p>
    <div class="snippet-text">{esc(snippet['snippet'])}</div>
  </div>

  <!-- ═══════════════════════════════════════════════════ SECTION 2 -->
  <h2>2 · Strategy Results (async, concurrent via gather)</h2>
  {strategy_cards}

</body>
</html>
"""


# =============================================================================
# ASYNC MAIN — dispatches all strategies at once via gather()
# =============================================================================

async def main():
    """Run all four strategies concurrently using asyncio.gather().

    All four calls are in-flight simultaneously. main() awaits gather(),
    which returns only after the last call completes. By that point all
    four results are ready.

    Compare to simple_example_02.main(), which awaits each call in sequence.
    The logic here is simpler — no for loop, no append — because gather()
    handles collection and ordering.
    """
    t_start = time.monotonic()

    print('STEP 3+4 — Launching all 4 strategies concurrently via gather()...')
    print(f'  Snippet: {snippet["id"]}')
    print()

    # One call to run_all_strategies fires all four API calls at once.
    # The terminal won't print the "done" lines until ALL four complete.
    results = await run_all_strategies(snippet['snippet'])

    total_elapsed = time.monotonic() - t_start
    print(f'  All 4 calls complete in {total_elapsed:.2f}s total '
          f'(vs ~{sum(r["elapsed_s"] for r in results):.2f}s sequential)')
    print()

    # Parse and display results
    print('STEP 5 — Results:')
    for r in results:
        extracted = parse_response(r['raw_response'])
        r['extracted'] = extracted  # store for HTML report
        print(f'  {r["strategy"]:12s} | {r["elapsed_s"]:.2f}s '
              f'| {r["prompt_tokens"]}in/{r["output_tokens"]}out '
              f'| → {extracted}')
    print()

    print('STEP 6 — Building HTML report...')
    out_path = Path('simple_example_03_report.html')
    out_path.write_text(build_html_report(snippet, results), encoding='utf-8')
    print(f'  Report written → {out_path.resolve()}')
    print()
    print('Done. Open simple_example_03_report.html in a browser.')
    print('─' * 60)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    asyncio.run(main())
