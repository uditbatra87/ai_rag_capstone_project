# =============================================================================
# simple_example_01.py  —  SYNCHRONOUS BASELINE  (Step 1 of 4)
# =============================================================================
#
# LEARNING GOAL
# ─────────────
# Understand the complete extraction loop — prompt → API call → parse → report
# — using the simplest possible execution model: one call at a time, in order.
#
# This version now runs all FOUR prompting strategies, but still does so
# synchronously: strategy 1 finishes completely before strategy 2 starts.
# That sequential, blocking behaviour is the key thing to observe here.
#
# What this script teaches:
#   • How to build four different prompt shapes for the same task
#   • How each strategy changes what the model sees and how it responds
#   • How a synchronous API call works (the program pauses until it responds)
#   • How to loop over strategies and collect results into a list
#   • How to build an HTML report that shows all four side by side
#
# FOUR PROMPT STRATEGIES
# ──────────────────────
#   zero_shot   — ask directly, no examples, no persona
#   few_shot    — show 3 worked examples before asking
#   structured  — set a system persona + explicit output schema
#   cot         — ask the model to reason step-by-step before answering
#
# PROGRESSION IN THE SERIES
# ─────────────────────────
#   simple_example_01  ← YOU ARE HERE
#                         (sync client, for loop, sequential, 1×4, no scoring)
#   simple_example_02    (async client, for loop, sequential, 1×4, no scoring)
#   simple_example_03    (async client, gather(),  concurrent, 1×4, no scoring)
#   intermediate_example (async, gather(), 10×4, accuracy scoring)
#   assignment_template  (async, 10×4, accuracy + LLM judge + cost + 3 outputs)
#
# DIFFERENCE vs simple_example_02
# ─────────────────────────────────
#   This script  → OpenAI client, plain function calls, for loop (blocking)
#   Script 02    → AsyncOpenAI client, await in for loop (async but still sequential)
#   Script 03    → AsyncOpenAI client, asyncio.gather() (truly concurrent)
#
# OUTPUT
# ──────
#   simple_example_01_report.html  — opens in any browser
#   Terminal prints each step as it runs.
#
# HOW TO RUN
# ──────────
#   Windows:   $env:OPENAI_API_KEY = "sk-..."
#   Mac/Linux: export OPENAI_API_KEY="sk-..."
#   Then:      cd scripts && python simple_example_01.py
# =============================================================================

# ── Standard-library imports ─────────────────────────────────────────────────
import json    # loading JSONL data files, serialising results for the report
import os      # reading the OPENAI_API_KEY environment variable
import re      # regex used in parse_response() to find JSON inside response text
import time    # time.time() used to measure how long each blocking call takes
from pathlib import Path  # OS-agnostic file paths

# ── Third-party import ───────────────────────────────────────────────────────
# OpenAI ships two client classes:
#   OpenAI      — synchronous (blocking), used in this script
#   AsyncOpenAI — asynchronous (non-blocking), introduced in simple_example_02
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Fail immediately with a readable message if the key is missing.
# Without this guard the first API call would raise an AuthenticationError
# with a much less helpful message.
assert os.environ.get('OPENAI_API_KEY'), (
    'OPENAI_API_KEY is not set.\n'
    'Windows:   $env:OPENAI_API_KEY = "sk-..."\n'
    'Mac/Linux: export OPENAI_API_KEY="sk-..."'
)

# Synchronous client — every method is a plain blocking function call.
# There is no event loop, no `await`, no coroutines involved.
client = OpenAI()

MODEL       = 'gpt-4o-mini'  # fast and cheap — good for learning / experimentation
TEMPERATURE = 0.0             # 0 = deterministic: same prompt always gives same output

# scripts/ is one level below the project root; data/ lives at ../data/
DATA_DIR = Path('../data')

# Color used in the HTML report to visually distinguish each strategy.
# Consistent with all other scripts in this series.
STRATEGY_COLORS = {
    'zero_shot':  '#6c757d',  # grey
    'few_shot':   '#0d6efd',  # blue
    'structured': '#198754',  # green
    'cot':        '#fd7e14',  # orange
}


# =============================================================================
# STEP 1 — LOAD ONE SNIPPET
# =============================================================================
# job_snippets.jsonl is a JSON Lines file: one JSON object per line.
# Each object has at minimum:
#   "id"      — unique identifier, e.g. "job_001"
#   "snippet" — the raw job posting text
#
# We load all snippets but only use the first one in this script.
# Subsequent scripts in the series scale up to the full 10-snippet dataset.

all_snippets = [
    json.loads(line)
    for line in (DATA_DIR / 'job_snippets.jsonl').read_text(encoding='utf-8').splitlines()
    if line.strip()   # skip any blank lines at the end of the file
]



# Use only snippet[0] — keeps the script focused on strategies, not data volume.
snippet = all_snippets[0]

print('─' * 60)
print('STEP 1 — Input data')
print(f'  Snippet ID  : {snippet["id"]}')
print(f'  Snippet text: {snippet["snippet"]}')
print()


# =============================================================================
# STEP 2 — DEFINE THE FOUR PROMPT STRATEGY FUNCTIONS
# =============================================================================
# Each function takes the raw job posting text and returns a "messages" list.
# That list is passed directly to client.chat.completions.create().
#
# A messages list is how the OpenAI Chat API represents a conversation:
#   [
#       {"role": "system",    "content": "...persistent instructions..."},
#       {"role": "user",      "content": "...your question..."},
#       {"role": "assistant", "content": "...model's reply..."},  ← optional
#   ]
#
# The four strategies below show four different ways to phrase the same task.
# Running them all on the same snippet makes it easy to compare the results.

def prompt_zero_shot(text: str) -> list[dict]:
    """Strategy 1 — Zero-shot: ask directly, no examples, no persona.

    This is the most minimal prompt possible. You describe the task and
    the output format, then hand over the data. No worked examples and
    no system persona are provided.

    When to use:
        Good starting point. Works well when the task is unambiguous and
        the model's training covers it well. Fails more often on edge cases
        (e.g., years expressed as words like "three", or null values).

    Message structure:
        [user]  "Extract ... return JSON ..."

    Args:
        text: The raw job posting string to extract from.

    Returns:
        A one-element list containing the user message dict.
    """
    return [
        {
            'role': 'user',
            # The content describes what to extract, what keys to use,
            # and what type each value should be.
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
    """Strategy 2 — Few-shot: show three worked examples before asking.

    Few-shot prompting embeds example input→output pairs directly in the
    prompt. The model learns the desired pattern from the examples rather
    than from explicit instructions alone.

    The three examples are carefully chosen to cover edge cases:
        Example 1 — explicit integer years ("3+" → 3)
        Example 2 — zero years ("fresh graduates welcome" → 0)
        Example 3 — null years (no requirement mentioned → null)

    If only one or two examples were given, the model might not generalise
    correctly to all three cases.

    The trailing "Output:" cue at the end of the prompt nudges the model
    to reply with JSON immediately rather than adding conversational text.

    When to use:
        When zero-shot struggles with edge cases. The examples act as
        implicit rules — the model infers "oh, when no years are stated,
        I should return null" without you having to explain that rule.

    Message structure:
        [user]  "...examples...\nNow extract from:\n{text}\nOutput:"

    Args:
        text: The raw job posting string to extract from.

    Returns:
        A one-element list containing the user message dict (which is
        longer than zero-shot because it includes the examples).
    """
    # The three examples as a single multi-line string.
    # Formatting is kept consistent (Posting: / Output:) so the model
    # can clearly identify the pattern.
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
    """Strategy 3 — Structured / role-based: system persona + explicit schema.

    This strategy uses TWO messages instead of one:
        1. A SYSTEM message that establishes the model's persona and sets
           hard output rules (no fences, no fabrication, type conversions).
        2. A USER message that is minimal — just the posting text.

    The system message is processed first and acts as persistent background
    context. Anything written there influences how the model interprets
    every subsequent user message. This is a good place for:
        • A role definition ("You are an expert technical recruiter")
        • Output format constraints ("raw JSON only, no markdown fences")
        • Edge-case handling rules ("convert word-numbers to integers")
        • Anti-hallucination rules ("never fabricate information")

    The user message is intentionally brief — the system message already
    did all the heavy lifting.

    When to use:
        When you need consistent, well-formatted output across many inputs.
        The system message rules reduce common failure modes. Works best
        for production-style pipelines where reliability matters.

    Message structure:
        [system]  "You are an expert recruiter. Return raw JSON only ..."
        [user]    "Extract from this job posting:\n{text}"

    Args:
        text: The raw job posting string to extract from.

    Returns:
        A two-element list: [system_message, user_message].
    """
    return [
        {
            # The system role sets persistent instructions for the entire
            # conversation. The model treats these as authoritative context.
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
            # The user message is just the data to process.
            # The system message has already told the model what to do with it.
            'role': 'user',
            'content': f'Extract the structured fields from this job posting:\n\n{text}',
        },
    ]


def prompt_cot(text: str) -> list[dict]:
    """Strategy 4 — Chain-of-thought (CoT): reason step by step first.

    Chain-of-thought prompting instructs the model to write out its
    reasoning before producing the final answer. This exploits a property
    of transformer models: generating intermediate reasoning tokens improves
    the quality of the final output, especially for tasks that require
    inference or disambiguation.

    The numbered steps guide the model to address each field individually:
        Step 1 → company
        Step 2 → role
        Step 3 → years (with explicit rules for ranges, word-numbers, nulls)
        Step 4 → output the JSON

    The model produces something like:
        "1. The company name is ... 2. The role is ... 3. They require ...
        {"company": "...", "role": "...", "years_experience_required": ...}"

    Because the JSON appears at the END of the reasoning text, parse_response()
    is designed to find the LAST {...} block — not the first.

    Trade-offs vs other strategies:
        + More accurate on ambiguous or complex postings
        + The reasoning text is visible — you can see WHY the model chose each value
        - Longer responses → more output tokens → marginally higher cost
        - Requires a more robust JSON parser (handled by parse_response())

    When to use:
        When accuracy matters more than speed or cost. Particularly useful
        when postings use indirect language ("we expect our candidates to
        bring 5+ years") or when years are implied rather than stated.

    Message structure:
        [user]  "Think step by step: 1. company... 2. role... 3. years...
                 4. Output JSON"

    Args:
        text: The raw job posting string to extract from.

    Returns:
        A one-element list containing the user message dict.
    """
    return [
        {
            'role': 'user',
            'content': (
                'Read the following job posting carefully and extract three fields.\n\n'
                f'Job posting:\n{text}\n\n'
                # Numbered steps force the model to reason about each field
                # independently before producing the final JSON.
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


# STRATEGIES is an ordered dict that maps each strategy name to its function.
#
# Using a dict (rather than a list) gives us two benefits:
#   1. Each strategy has a human-readable name we can use in the report.
#   2. Adding a new strategy only requires one new entry here — all loops
#      that iterate over strategies automatically pick it up.
#
# The insertion order (Python 3.7+) is preserved, so the loop below always
# processes strategies in the order: zero_shot → few_shot → structured → cot.
STRATEGIES = {
    'zero_shot':  prompt_zero_shot,
    'few_shot':   prompt_few_shot,
    'structured': prompt_structured,
    'cot':        prompt_cot,
}


# =============================================================================
# STEP 3 — RESPONSE PARSING HELPER
# =============================================================================

def parse_response(text: str) -> dict | None:
    """Extract a Python dict from the model's raw response text.

    The model is asked to return JSON, but its actual output can vary:
        Pattern A — clean JSON (ideal):
            {"company": "Acme", "role": "Engineer", "years_experience_required": 3}

        Pattern B — markdown-fenced JSON (very common):
            ```json
            {"company": "Acme", ...}
            ```

        Pattern C — reasoning text then JSON (CoT strategy):
            "The company is Acme... {"company": "Acme", ...}"

    Three-step fallback handles all patterns:
        1. Strip fence markers (``` / ```json), try json.loads().
        2. Regex-find the last {...} block, try json.loads().
           (The LAST block is the JSON in a CoT response.)
        3. Return None — caller decides how to handle a parse failure.

    Args:
        text: The raw string from response.choices[0].message.content.

    Returns:
        dict on success, None if the response cannot be parsed.
    """
    # Step 1 — strip markdown fences and attempt a direct parse.
    # re.sub removes the opening ``` or ```json marker (including trailing newline).
    # The second .replace removes the closing ``` marker.
    cleaned = re.sub(r'```(?:json)?\s*', '', text).strip()
    cleaned = cleaned.replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass  # move to step 2

    # Step 2 — regex fallback: find last single-depth {...} block.
    # r'\{[^{}]+\}' matches a flat JSON object (curly braces, no nesting).
    # re.DOTALL allows . to match newlines, so multi-line JSON objects work.
    # reversed() iterates from last match to first — CoT puts JSON at the end.
    matches = list(re.finditer(r'\{[^{}]+\}', text, re.DOTALL))
    for match in reversed(matches):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue  # try the previous match

    # Step 3 — nothing parseable found
    return None


# =============================================================================
# STEP 4 — CALL THE API FOR EACH STRATEGY  (synchronous, sequential loop)
# =============================================================================
# This is the core of the synchronous approach.
#
# We loop over STRATEGIES and make one blocking API call per strategy.
# "Blocking" means:
#   • Python sends the HTTP request to OpenAI.
#   • Python STOPS and waits until the full response arrives.
#   • Only then does the loop advance to the next strategy.
#
# Timeline (approximate, each call ~1 s):
#   zero_shot  → [send]─────[wait 1s]─────[receive] →
#   few_shot   →                                       [send]─────[wait 1s]─────[receive] →
#   structured →                                                                             [send]──...
#   cot        →                                                                                        ...
#
# Total time ≈ 4 × (average call time)
#
# Compare this to simple_example_02 where all four calls are in flight
# simultaneously: total time there ≈ 1 × (slowest single call).

print('STEP 4 — Calling the API for each strategy (sequential / blocking)')
print(f'  Snippet: {snippet["id"]}')
print()

# results collects one dict per strategy — used later to build the HTML report.
results = []

for strategy_name, prompt_fn in STRATEGIES.items():
    # ── Build the prompt ──────────────────────────────────────────────────────
    # prompt_fn is whichever strategy function we're on this iteration.
    # It returns a messages list ready to pass directly to the API.
    messages = prompt_fn(snippet['snippet'])

    print(f'  [{strategy_name}] Sending {len(messages)} message(s) to {MODEL}...')

    # ── Make the blocking API call ────────────────────────────────────────────
    # time.time() before and after the call measures real wall-clock seconds.
    # (simple_example_01 uses time.time() here for simplicity; later scripts
    #  use time.monotonic() which is immune to system clock adjustments.)
    t_start  = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        # No max_tokens limit here — we want the full response,
        # especially for the CoT strategy which writes reasoning text.
    )

    elapsed  = time.time() - t_start

    # ── Extract the reply text ────────────────────────────────────────────────
    # response.choices is a list; [0] is the first (and usually only) choice.
    # .message.content is the model's reply as a plain string.
    # `or ''` guards against the rare case where content is None.
    raw_response = response.choices[0].message.content or ''

    # ── Parse the reply into a dict ───────────────────────────────────────────
    extracted = parse_response(raw_response)

    # ── Print a summary line to the terminal ──────────────────────────────────
    # This shows you the progress as the script runs, since each call blocks.
    print(f'       Done in {elapsed:.2f}s  |  tokens: {response.usage.prompt_tokens}in '
          f'/ {response.usage.completion_tokens}out  |  parsed: {extracted is not None}')
    print(f'       Result: {extracted}')
    print()

    # ── Store the full result for the HTML report ─────────────────────────────
    results.append({
        'strategy':      strategy_name,           # used for the section heading
        'messages':      messages,                # rendered as chat bubbles
        'raw_response':  raw_response,            # shown verbatim
        'extracted':     extracted,               # shown as formatted JSON
        'elapsed_s':     elapsed,                 # shown in the report header
        'prompt_tokens': response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens,
    })


# =============================================================================
# STEP 5 — HTML REPORT HELPERS
# =============================================================================
# These functions are defined after the API calls so they appear close to
# where they are used (the HTML building section below).

def esc(text: str) -> str:
    """HTML-escape &, <, >, " in a string.

    Must be applied to ALL user-supplied content before embedding in HTML.
    Failure to do this means a job posting containing '<' or '>' would
    create unintended HTML tags and break the page layout.

    Order matters: & must be replaced first. If we replaced < first and
    then &, we'd turn our own &lt; into &amp;lt; (double-escaping).

    Args:
        text: Any raw string.

    Returns:
        HTML-safe version of the string.
    """
    return (
        text.replace('&', '&amp;')   # ← must come first
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
    )


def messages_to_html(msgs: list[dict]) -> str:
    """Render a messages list as color-coded chat bubbles.

    Role → visual style (consistent across all scripts in this series):
        system    → yellow background  (persistent instructions / persona)
        user      → blue background    (the prompt you wrote)
        assistant → green background   (model reply, if stored)

    Each bubble contains:
        • A small uppercase role label (SYSTEM / USER / ASSISTANT)
        • The full message content in a <pre> tag (preserves whitespace)

    The content is passed through esc() so that special HTML characters
    in job postings or model responses render correctly.

    Args:
        msgs: Messages list in OpenAI format (list of role/content dicts).

    Returns:
        HTML string with one styled <div> per message.
    """
    bubbles = []
    for msg in msgs:
        role    = msg.get('role', '')
        content = msg.get('content', '')

        # Map role to (background color, label text, label text color)
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
    """Return a colored pill <span> for a strategy name.

    Colors come from STRATEGY_COLORS for visual consistency across the report.

    Args:
        name: Strategy name (key in STRATEGIES dict).

    Returns:
        HTML <span> string styled as a colored pill badge.
    """
    bg = STRATEGY_COLORS.get(name, '#aaa')
    return (
        f'<span style="background:{bg};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:.8em;font-weight:700">{name}</span>'
    )


# =============================================================================
# STEP 6 — BUILD AND WRITE THE HTML REPORT
# =============================================================================
# The report has two top-level sections:
#
#   Section 1 — Input Data
#       Shows the job posting text that was passed to all four strategies.
#
#   Section 2 — Strategy Results (one card per strategy)
#       For each strategy (zero_shot, few_shot, structured, cot):
#         2a. Strategy badge + timing / token metadata
#         2b. Prompt sent to the model (rendered as chat bubbles)
#         2c. Raw model response (verbatim text from the API)
#         2d. Parsed result (the final Python dict, pretty-printed as JSON)
#
# Reading the report side by side with the code is the best way to
# understand what each strategy actually sends and receives.

print('STEP 5 — Building HTML report...')

# ── Build one card per strategy result ───────────────────────────────────────
strategy_cards = ''
for row in results:
    ext_str = (
        json.dumps(row['extracted'], indent=2)
        if row['extracted'] is not None
        else 'null  ← parse_response() could not find valid JSON in the response above'
    )
    bg_color = STRATEGY_COLORS.get(row['strategy'], '#aaa')

    strategy_cards += f"""
    <!-- ── Card: {row['strategy']} ── -->
    <div style="border:1px solid #dee2e6;border-radius:8px;margin-bottom:28px;overflow:hidden;">

      <!-- Header bar: strategy badge + call metadata -->
      <div style="background:#343a40;padding:10px 16px;display:flex;align-items:center;
                  gap:14px;flex-wrap:wrap;">
        {strategy_badge(row['strategy'])}
        <span style="color:#adb5bd;font-size:.8em;">
          {row['prompt_tokens']} prompt tokens
          &nbsp;+&nbsp; {row['output_tokens']} completion tokens
          &nbsp;|&nbsp; {row['elapsed_s']:.2f}s
        </span>
      </div>

      <div style="padding:14px 18px;">

        <!-- 2b. Prompt bubbles -->
        <p style="margin:0 0 6px;font-size:.75em;font-weight:700;color:#6c757d;
           text-transform:uppercase;letter-spacing:.06em">Prompt sent to model</p>
        {messages_to_html(row['messages'])}

        <!-- 2c. Raw response -->
        <p style="margin:14px 0 6px;font-size:.75em;font-weight:700;color:#6c757d;
           text-transform:uppercase;letter-spacing:.06em">Raw model response</p>
        <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:10px 12px;">
          <pre style="margin:0;white-space:pre-wrap;font-size:.82em;
               font-family:Menlo,Consolas,monospace">{esc(row['raw_response'])}</pre>
        </div>

        <!-- 2d. Parsed result -->
        <p style="margin:14px 0 6px;font-size:.75em;font-weight:700;color:#6c757d;
           text-transform:uppercase;letter-spacing:.06em">Parsed result</p>
        <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:10px 12px;">
          <pre style="margin:0;white-space:pre-wrap;font-size:.82em;
               font-family:Menlo,Consolas,monospace">{esc(ext_str)}</pre>
        </div>

      </div>
    </div>"""

# ── Assemble the full HTML document ──────────────────────────────────────────
# {{ and }} are escaped braces in f-strings — they produce literal { and }
# in the output HTML, which CSS needs for its rule blocks.
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Simple Example 01 — Synchronous, All 4 Strategies</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body  {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
             background: #f8f9fa; color: #212529; margin: 0; padding: 32px; max-width: 980px; }}
    h1    {{ font-size: 1.5rem; margin-bottom: 4px; }}
    h2    {{ font-size: 1rem; color: #495057; margin: 32px 0 12px;
             border-bottom: 2px solid #dee2e6; padding-bottom: 5px; }}
    .badge {{ background:#6c757d; color:#fff; padding:2px 10px; border-radius:12px;
              font-size:.8em; font-weight:700; }}
    .card  {{ background:#fff; border-radius:8px; padding:16px 20px;
              box-shadow:0 1px 4px rgba(0,0,0,.08); margin-bottom:20px; }}
    .snippet-text {{ background:#f8f9fa; border-left:4px solid #6c757d;
                     padding:10px 14px; border-radius:0 4px 4px 0;
                     font-size:.9em; white-space:pre-wrap; }}
    .meta {{ font-size:.82em; color:#6c757d; margin-bottom:28px; }}
    .tag  {{ display:inline-block; background:#e9ecef; color:#495057;
             padding:1px 8px; border-radius:4px; font-size:.78em;
             font-family:monospace; margin-right:4px; }}
    /* Yellow callout explaining the synchronous / sequential execution model */
    .callout {{ background:#fff3cd; border-left:4px solid #ffc107;
                padding:10px 14px; border-radius:0 6px 6px 0;
                font-size:.85em; margin-bottom:24px; }}
  </style>
</head>
<body>
  <h1>Simple Example 01 <span class="badge">Synchronous</span></h1>
  <div class="meta">
    Scope: <strong>1 snippet × 4 strategies (sequential)</strong>
    &nbsp;|&nbsp; Model: <strong>{MODEL}</strong>
    &nbsp;|&nbsp; API style: <strong>synchronous (blocking, one call at a time)</strong>
  </div>

  <!-- Explains the key concept of this script to someone reading the report -->
  <div class="callout">
    <strong>How this script works:</strong> each strategy is called
    <em>one at a time</em> in a <code>for</code> loop. The program blocks
    (pauses completely) while waiting for each API response before starting
    the next. Total time = sum of all 4 call durations.
    <br><br>
    Compare to <strong>simple_example_02.py</strong>, which uses <code>AsyncOpenAI</code>
    and <code>await</code> in the same loop structure — still sequential, but now async.
    <strong>simple_example_03.py</strong> then introduces <code>asyncio.gather()</code>
    to make all 4 calls truly concurrent.
  </div>

  <!-- ═══════════════════════════════════════════════════ SECTION 1 -->
  <h2>1 · Input Data</h2>
  <div class="card">
    <p style="margin:0 0 6px;">
      <span class="tag">{esc(snippet['id'])}</span>
      Job posting text — the same text is passed to all four strategies below:
    </p>
    <div class="snippet-text">{esc(snippet['snippet'])}</div>
  </div>

  <!-- ═══════════════════════════════════════════════════ SECTION 2 -->
  <h2>2 · Strategy Results (all four, sequential)</h2>
  {strategy_cards}

</body>
</html>
"""

# Write the report next to this script, inside scripts/.
out_path = Path('simple_example_01_report.html')
out_path.write_text(html, encoding='utf-8')

print(f'  Report written → {out_path.resolve()}')
print()
print('Done. Open simple_example_01_report.html in a browser.')
print('─' * 60)
