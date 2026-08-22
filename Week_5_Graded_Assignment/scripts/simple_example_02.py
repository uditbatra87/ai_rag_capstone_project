# =============================================================================
# simple_example_02.py  —  ASYNC SEQUENTIAL  (Step 2 of 5)
# =============================================================================
#
# LEARNING GOAL
# ─────────────
# Introduce the async/await syntax without introducing concurrency yet.
# This script runs all four strategies using the async OpenAI client,
# but still calls them ONE AT A TIME in a for loop — exactly like
# simple_example_01.py, except the client and call syntax are now async.
#
# That one change — switching from OpenAI to AsyncOpenAI and adding
# `async def` / `await` — is the ONLY new concept here.
#
# What this script teaches:
#   • AsyncOpenAI  — the async version of the client
#   • async def    — how to declare a coroutine function
#   • await        — how to pause a coroutine while waiting for I/O
#   • asyncio.run() — the synchronous bridge that starts the event loop
#   • Why calling an async function WITHOUT await returns a coroutine
#     object instead of a result (a common beginner mistake)
#
# WHAT IS NOT INTRODUCED YET
# ──────────────────────────
# asyncio.gather() — launching multiple coroutines at the same time —
# is intentionally held back for simple_example_03.py. Understanding
# await on a single coroutine first makes gather() much easier to grasp.
#
# EXECUTION MODEL
# ───────────────
# Even though we use the async client and `await` each call, the four
# strategy calls still run one after another:
#
#   async for loop iteration 1: await zero_shot  → wait → receive
#   async for loop iteration 2: await few_shot   → wait → receive
#   async for loop iteration 3: await structured → wait → receive
#   async for loop iteration 4: await cot        → wait → receive
#
#   Total time ≈ sum of all 4 call durations  (same as simple_example_01)
#
# The event loop exists but is idle between each await — there are no
# other coroutines to switch to. That's what simple_example_03 fixes.
#
# PROGRESSION IN THE SERIES
# ─────────────────────────
#   simple_example_01    sync client, for loop, sequential           (blocking)
#   simple_example_02  ← YOU ARE HERE
#                        async client, for loop, sequential          (async but still sequential)
#   simple_example_03    async client, asyncio.gather(), concurrent  (truly parallel)
#   intermediate_example async, 10 snippets × 4 strategies, accuracy scoring
#   assignment_template  async, 10×4, accuracy + LLM judge + cost + 3 outputs
#
# DIFFERENCE FROM simple_example_01
# ──────────────────────────────────
#   Script 01: from openai import OpenAI          → client = OpenAI()
#              response = client.chat...create()  → plain function call, blocks
#
#   Script 02: from openai import AsyncOpenAI     → client = AsyncOpenAI()
#              response = await client.chat...create()  → coroutine, suspends
#
#   Everything else (data loading, strategy functions, parsing, HTML) is
#   identical. Keeping the diff minimal makes it easy to spot what changed.
#
# DIFFERENCE FROM simple_example_03
# ──────────────────────────────────
#   Script 02: calls strategies sequentially with `await` in a for loop
#   Script 03: calls strategies concurrently with asyncio.gather()
#
#   Script 02 total time ≈ sum of durations
#   Script 03 total time ≈ slowest single call
#
# OUTPUT
# ──────
#   simple_example_02_report.html  — opens in any browser
#   Terminal prints each step as it completes.
#
# HOW TO RUN
# ──────────
#   Windows:   $env:OPENAI_API_KEY = "sk-..."
#   Mac/Linux: export OPENAI_API_KEY="sk-..."
#   Then:      cd scripts && python simple_example_02.py
# =============================================================================

# ── Standard-library imports ─────────────────────────────────────────────────
import asyncio  # provides the event loop, async def machinery, and asyncio.run()
import json
import os
import re
import time     # time.monotonic() for per-call latency measurement
from pathlib import Path

# ── Third-party import ───────────────────────────────────────────────────────
# The critical change from simple_example_01:
#   OpenAI      → blocking, every method is a plain function call
#   AsyncOpenAI → non-blocking, every I/O method returns a coroutine
#                 that must be awaited to get the actual result
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

# AsyncOpenAI — calling .create() without await returns a coroutine object,
# not the API response. You MUST await it to get the response.
# This is the most common beginner mistake when switching from OpenAI to AsyncOpenAI.
client = AsyncOpenAI()

MODEL       = 'gpt-4o-mini'
TEMPERATURE = 0.0

# scripts/ is one level below the project root; data/ lives at ../data/
DATA_DIR = Path('../data')

# Consistent strategy colors across all scripts in this series
STRATEGY_COLORS = {
    'zero_shot':  '#6c757d',  # grey
    'few_shot':   '#0d6efd',  # blue
    'structured': '#198754',  # green
    'cot':        '#fd7e14',  # orange
}


# =============================================================================
# STEP 1 — LOAD ONE SNIPPET
# =============================================================================
# Identical to simple_example_01. One job posting, take the first.
# The async machinery is the only thing changing in this script.

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
# These functions are SYNCHRONOUS — building a prompt string involves no I/O,
# so there is nothing to await. Only the API call that uses the messages list
# needs to be async.
#
# This is an important observation: async/await is about I/O, not about
# everything in the program. Computation stays synchronous.

def prompt_zero_shot(text: str) -> list[dict]:
    """Strategy 1 — Zero-shot: ask directly, no examples, no persona.

    Single user message. Relies entirely on the model's training to
    understand the task. Simplest and fastest prompt shape.

    Message structure:
        [user] "Extract ... return JSON ..."

    Args:
        text: The raw job posting string.

    Returns:
        One-element list with the user message dict.
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

    Three examples cover the main edge cases:
        Example 1 — explicit integer years  (3+ → 3)
        Example 2 — zero years              (fresh graduates → 0)
        Example 3 — null years              (no requirement → null)

    The trailing "Output:" cue tells the model to reply with JSON directly
    rather than adding conversational text first.

    Message structure:
        [user] "...examples...\nNow extract from:\n{text}\nOutput:"

    Args:
        text: The raw job posting string.

    Returns:
        One-element list with the (longer) user message dict.
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

    Two-message structure:
        system — sets persona, schema, and output constraints persistently
        user   — minimal: just the posting text

    The system message is processed first and acts as permanent context.
    Rules placed here ("no fences", "no fabrication", "convert word-numbers")
    apply to the user message that follows.

    Message structure:
        [system] "You are an expert recruiter. Return raw JSON only ..."
        [user]   "Extract from this job posting:\n{text}"

    Args:
        text: The raw job posting string.

    Returns:
        Two-element list: [system_message, user_message].
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

    The model writes its reasoning for each field before outputting JSON.
    This exploits the property that generating intermediate reasoning tokens
    improves the quality of the final output on ambiguous inputs.

    The JSON appears at the END of the response, after the reasoning.
    parse_response() handles this by finding the LAST {...} block.

    Trade-offs:
        + More accurate on ambiguous postings (ranges, word-numbers, implicit nulls)
        + Reasoning is visible — you can see why the model chose each value
        - Longer response → more output tokens → slightly higher cost

    Message structure:
        [user] "Think step by step: 1.company 2.role 3.years 4.JSON"

    Args:
        text: The raw job posting string.

    Returns:
        One-element list with the user message dict.
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


# Strategy registry — one entry per strategy.
# The for loop below iterates this in insertion order.
STRATEGIES = {
    'zero_shot':  prompt_zero_shot,
    'few_shot':   prompt_few_shot,
    'structured': prompt_structured,
    'cot':        prompt_cot,
}


# =============================================================================
# STEP 3 — ASYNC CALL FUNCTION  (single strategy, single snippet)
# =============================================================================
#
# This is the central learning piece of this script.
#
# compare with simple_example_01:
# ─────────────────────────────────────────────────────────────────────────────
#   SYNC (script 01):                    ASYNC (this script):
#   ────────────────                     ────────────────────
#   def call_strategy(...):              async def call_strategy(...):
#       resp = client.chat              →    resp = await client.chat
#                .completions                         .completions
#                .create(...)                         .create(...)
#       return {...}                         return {...}
# ─────────────────────────────────────────────────────────────────────────────
#
# The differences:
#   • `async def` — declares this as a coroutine function.
#     Calling call_strategy(...) does NOT execute the body immediately.
#     It returns a coroutine object. The body only runs when you await it.
#
#   • `await client.chat.completions.create(...)` — suspends this coroutine
#     while the HTTP request is in flight. The event loop can run other work.
#     When the response arrives, the event loop resumes this coroutine.
#
#   In this script, there is no other work to run during the suspension —
#   we are only running one call at a time. But the pattern is in place,
#   ready for simple_example_03 to exploit it with gather().

async def call_strategy(strategy_name: str, snippet_text: str) -> dict:
    """Make one async API call for one strategy and return the result.

    Even though this is async, it is called with `await` in a for loop,
    so it still runs sequentially — one call finishes before the next starts.
    The value of `async def` + `await` here is purely syntactic: it uses
    the same client and pattern that simple_example_03 will use for
    concurrent calls, making the upgrade to gather() a one-line change.

    Args:
        strategy_name: Key into STRATEGIES dict.
        snippet_text:  Raw job posting text to extract from.

    Returns:
        Dict with keys: strategy, messages, raw_response, elapsed_s,
        prompt_tokens, output_tokens.
    """
    # Build the prompt — synchronous, instant, no I/O.
    messages = STRATEGIES[strategy_name](snippet_text)

    # time.monotonic() is a monotonic clock — it never goes backwards,
    # unlike time.time() which can jump if the system clock is adjusted.
    # Use it whenever measuring elapsed duration rather than wall time.
    t_start = time.monotonic()

    # THIS IS THE KEY LINE:
    # `await` suspends this coroutine until the HTTP response arrives.
    # During the suspension the event loop could run other coroutines —
    # but in this script there are none. See simple_example_03 for that.
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )

    elapsed = time.monotonic() - t_start

    return {
        'strategy':      strategy_name,
        'messages':      messages,                               # for the HTML report
        'raw_response':  resp.choices[0].message.content or '', # verbatim model text
        'elapsed_s':     elapsed,
        'prompt_tokens': resp.usage.prompt_tokens,
        'output_tokens': resp.usage.completion_tokens,
    }


# =============================================================================
# STEP 4 — PARSE RESPONSE HELPER
# =============================================================================

def parse_response(text: str) -> dict | None:
    """Extract a Python dict from the model's raw response text.

    Three-step fallback (identical across all scripts in this series):

        Step 1 — Strip markdown fences (``` / ```json), try json.loads().
                 Handles the most common case: clean or fenced JSON.

        Step 2 — Regex-find the LAST {...} block, try json.loads().
                 The LAST block is correct for CoT responses where JSON
                 appears after reasoning text.

        Step 3 — Return None. Caller handles parse failure gracefully.

    Args:
        text: The raw string from response.choices[0].message.content.

    Returns:
        dict on success, None on failure.
    """
    # Step 1: strip fences, attempt direct parse
    cleaned = re.sub(r'```(?:json)?\s*', '', text).strip()
    cleaned = cleaned.replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 2: find last {...} block (handles CoT format)
    matches = list(re.finditer(r'\{[^{}]+\}', text, re.DOTALL))
    for match in reversed(matches):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue

    return None  # Step 3: nothing parseable


# =============================================================================
# STEP 5 — HTML REPORT HELPERS
# =============================================================================

def esc(text: str) -> str:
    """HTML-escape &, <, >, " — applied to all user-supplied text in HTML.

    & must be replaced first to avoid double-escaping other entities.
    """
    return (
        text.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
    )


def messages_to_html(msgs: list[dict]) -> str:
    """Render a messages list as color-coded chat bubbles.

    Role → color (consistent across all scripts in this series):
        system    → yellow  #fff3cd   persistent instructions / persona
        user      → blue    #cfe2ff   the prompt you sent
        assistant → green   #d1e7dd   model reply (if stored)

    Content is placed in a <pre> tag to preserve whitespace exactly.
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
    """Assemble the full HTML page.

    Structure:
        Header  — scope, model, API style
        Callout — explains async-sequential vs concurrent
        Section 1 — Input job posting
        Section 2 — One card per strategy:
                     badge + tokens + latency header
                     prompt bubbles → raw response → parsed result

    Args:
        snippet: Full snippet dict (id + snippet text).
        results: List of result dicts from the async for loop, one per strategy.

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

      <!-- Header: strategy badge + token counts + latency -->
      <div style="background:#343a40;padding:10px 16px;display:flex;align-items:center;
                  gap:14px;flex-wrap:wrap;">
        {strategy_badge(row['strategy'])}
        <span style="color:#adb5bd;font-size:.8em;">
          {row['prompt_tokens']} prompt tokens
          &nbsp;+&nbsp; {row['output_tokens']} completion tokens
          &nbsp;|&nbsp; {row['elapsed_s']:.2f}s
          &nbsp;|&nbsp; <em>awaited individually</em>
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
  <title>Simple Example 02 — Async Sequential</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body  {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
             background: #f8f9fa; color: #212529; margin: 0; padding: 32px; max-width: 980px; }}
    h1    {{ font-size: 1.5rem; margin-bottom: 4px; }}
    h2    {{ font-size: 1rem; color: #495057; margin: 32px 0 12px;
             border-bottom: 2px solid #dee2e6; padding-bottom: 5px; }}
    .badge  {{ background:#0d6efd; color:#fff; padding:2px 10px; border-radius:12px;
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
    /* Blue callout — highlights the one key concept of this script */
    .callout {{ background:#e7f3ff; border-left:4px solid #0d6efd;
                padding:10px 14px; border-radius:0 6px 6px 0;
                font-size:.85em; margin-bottom:24px; }}
    /* Orange comparison box — shows what changes in the next script */
    .compare {{ background:#fff3e0; border-left:4px solid #fd7e14;
                padding:10px 14px; border-radius:0 6px 6px 0;
                font-size:.85em; margin-bottom:24px; }}
  </style>
</head>
<body>
  <h1>Simple Example 02 <span class="badge">Async Sequential</span></h1>
  <div class="meta">
    Scope: <strong>1 snippet × 4 strategies (async, sequential)</strong>
    &nbsp;|&nbsp; Model: <strong>{MODEL}</strong>
    &nbsp;|&nbsp; Client: <strong>AsyncOpenAI</strong>
    &nbsp;|&nbsp; Dispatch: <strong>await in a for loop (one at a time)</strong>
  </div>

  <!-- Key concept callout -->
  <div class="callout">
    <strong>What's new vs simple_example_01:</strong> the client is now
    <code>AsyncOpenAI</code> and each call uses <code>await</code>.
    <code>async def call_strategy()</code> is a coroutine — calling it returns
    a coroutine object, not a result. Only <code>await</code>-ing it runs the body
    and produces the response.
    <br><br>
    The calls are still <em>sequential</em> because we <code>await</code> each one
    individually inside a <code>for</code> loop. Total time ≈ sum of all 4 durations.
  </div>

  <!-- What changes next -->
  <div class="compare">
    <strong>What simple_example_03 changes:</strong> instead of
    <code>await call_strategy(name)</code> in a loop, it passes all four coroutines
    to <code>asyncio.gather()</code> at once — launching them simultaneously.
    Total time drops from ~4× to ~1× the average call duration.
  </div>

  <!-- ═══════════════════════════════════════════════════ SECTION 1 -->
  <h2>1 · Input Data</h2>
  <div class="card">
    <p style="margin:0 0 6px;">
      <span class="tag">{esc(snippet['id'])}</span>
      Job posting text — passed to all four strategies below:
    </p>
    <div class="snippet-text">{esc(snippet['snippet'])}</div>
  </div>

  <!-- ═══════════════════════════════════════════════════ SECTION 2 -->
  <h2>2 · Strategy Results (async, sequential)</h2>
  {strategy_cards}

</body>
</html>
"""


# =============================================================================
# ASYNC MAIN — runs strategies one at a time with individual awaits
# =============================================================================

async def main():
    """Run all four strategies sequentially using await in a for loop.

    Each `await call_strategy(...)` suspends main() until that call
    returns. The next strategy only starts after the previous one finishes.
    This is functionally identical to the synchronous for loop in
    simple_example_01 — the difference is only the client type and syntax.

    The async wrapper (async def main + asyncio.run) is still needed because
    call_strategy() is a coroutine and can only be awaited inside an async
    function or event loop.
    """
    print('STEP 3 — Calling each strategy with await (async, sequential)')
    print(f'  Snippet: {snippet["id"]}')
    print()

    results = []

    # For loop with await — each iteration completes before the next starts.
    # The async client is present but the event loop has nothing else to do
    # while waiting for each response. That changes in simple_example_03.
    for strategy_name, prompt_fn in STRATEGIES.items():
        print(f'  [{strategy_name}] awaiting call_strategy()...')

        # await here suspends main() and runs call_strategy() to completion.
        # Compare to simple_example_03 which creates ALL coroutines first,
        # then passes them to gather() so they all run simultaneously.
        result = await call_strategy(strategy_name, snippet['snippet'])
        results.append(result)

        # Parse for the terminal summary
        extracted = parse_response(result['raw_response'])
        result['extracted'] = extracted  # store for the HTML report

        print(f'       done {result["elapsed_s"]:.2f}s '
              f'| {result["prompt_tokens"]}in/{result["output_tokens"]}out tokens '
              f'| parsed: {extracted is not None}')
        print(f'       result: {extracted}')
        print()

    print('STEP 4 — Building HTML report...')
    out_path = Path('simple_example_02_report.html')
    out_path.write_text(build_html_report(snippet, results), encoding='utf-8')
    print(f'  Report written → {out_path.resolve()}')
    print()
    print('Done. Open simple_example_02_report.html in a browser.')
    print('─' * 60)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# asyncio.run() creates the event loop, runs main() inside it, then tears
# it down. Without asyncio.run() you cannot execute an async function from
# a regular synchronous Python script.
#
# The `if __name__` guard prevents this from firing when the module is
# imported rather than executed directly.
if __name__ == '__main__':
    asyncio.run(main())
