# simple_example_02.py — Async Sequential

**Position in the learning series:** Step 2 of 5  
**Complexity:** Beginner–Intermediate  
**API style:** Asynchronous (`AsyncOpenAI`), sequential execution  
**Scope:** 1 snippet × 4 strategies — awaited **one at a time** in a `for` loop  
**Output:** `simple_example_02_report.html`

---

## What this script teaches

This is the minimal async introduction. The execution model is still sequential —
strategies run one after another, just like `simple_example_01` — but the client
and syntax switch to async.

The single new concept is: **how do `async def` and `await` work?**

By keeping the calling pattern (a `for` loop) identical to script 01 and changing
only the client type and two keywords, the diff is small enough to study line by line.

After understanding this script you will know:

- What `AsyncOpenAI` is and how it differs from `OpenAI`
- What `async def` means and why calling an async function returns a coroutine
  object instead of a result
- What `await` does — and why leaving it out is a silent bug
- What `asyncio.run()` is and why it is needed
- Why this script is still sequential despite using async syntax

---

## The one-line diff from simple_example_01

```python
# simple_example_01 (synchronous)
from openai import OpenAI
client   = OpenAI()
response = client.chat.completions.create(model=..., messages=..., temperature=...)

# simple_example_02 (async)
from openai import AsyncOpenAI
client   = AsyncOpenAI()
response = await client.chat.completions.create(model=..., messages=..., temperature=...)
```

Everything else — data loading, strategy functions, parse_response(), HTML helpers —
is identical.

---

## Why still sequential?

Using `await` inside a `for` loop suspends the current coroutine until each call
finishes before the loop advances to the next iteration. The event loop exists but
has no other coroutines to run during the wait.

```
for loop iteration 1:   await zero_shot   → [wait] → done
for loop iteration 2:   await few_shot    → [wait] → done
for loop iteration 3:   await structured  → [wait] → done
for loop iteration 4:   await cot         → [wait] → done

Total time ≈ sum of all 4 durations  (same as script 01)
```

`simple_example_03` fixes this with `asyncio.gather()`.

---

## Prerequisites

```bash
pip install openai

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Mac / Linux
export OPENAI_API_KEY="sk-..."
```

---

## How to run

```bash
cd scripts
python simple_example_02.py
```

---

## Data flow

```
data/job_snippets.jsonl
        │
        ▼
  Load all lines → take first snippet
        │
        ▼
  asyncio.run(main())             ← starts the event loop
        │
        ▼
  async def main():
        │
        ▼  ┌──────────────────────────────────────────────────────────┐
           │  for strategy_name, prompt_fn in STRATEGIES.items():     │
           │                                                          │
           │    result = await call_strategy(strategy_name, ...)      │
           │           ↑                                              │
           │           suspends main() here until API responds        │
           │           (event loop has nothing else to run)           │
           │                                                          │
           │    results.append(result)                                │
           └──────────────────────────────────────────────────────────┘
        │
        ▼
  Build HTML → write simple_example_02_report.html
```

---

## Module-level constants

| Name | Value | Purpose |
|---|---|---|
| `MODEL` | `'gpt-4o-mini'` | OpenAI model for all 4 calls |
| `TEMPERATURE` | `0.0` | Deterministic output |
| `DATA_DIR` | `Path('../data')` | Resolves to `data/` one level above `scripts/` |
| `STRATEGY_COLORS` | dict | Hex color per strategy for HTML badges |

---

## Function reference

### Prompt strategy functions

Identical to `simple_example_01`. All four are plain synchronous functions —
building a string involves no I/O, so no `await` is needed or appropriate.

| Function | Strategy | Messages |
|---|---|---|
| `prompt_zero_shot(text)` | Ask directly, no examples | 1 user |
| `prompt_few_shot(text)` | 3 worked examples before asking | 1 user (longer) |
| `prompt_structured(text)` | System persona + schema + rules | 1 system + 1 user |
| `prompt_cot(text)` | Step-by-step reasoning before JSON | 1 user (CoT steps) |

See `README_simple_example_01.md` for a full explanation of each strategy.

---

### `async def call_strategy(strategy_name, snippet_text) -> dict`

**The central learning piece of this script.**

```python
async def call_strategy(strategy_name: str, snippet_text: str) -> dict:
    messages = STRATEGIES[strategy_name](snippet_text)  # synchronous — instant
    resp = await client.chat.completions.create(        # async — suspends here
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )
    ...
```

**`async def` — what it means:**

Declaring a function with `async def` makes it a *coroutine function*.
Calling `call_strategy("zero_shot", text)` does **not** run the body.
It returns a *coroutine object*. To run the body and get a result you must
either `await` it or pass it to `asyncio.gather()`.

```python
# WRONG — silent bug: result is a coroutine object, not a dict
result = call_strategy("zero_shot", text)

# CORRECT — awaiting runs the body and returns the dict
result = await call_strategy("zero_shot", text)
```

**`await` — what it does:**

`await expr` suspends the current coroutine until `expr` completes, then
resumes with the result. While suspended, the event loop is free to run
other coroutines — but in this script there are none queued.

**Why `time.monotonic()` instead of `time.time()`:**

`time.monotonic()` returns a clock that never goes backwards, regardless
of system clock adjustments (NTP sync, daylight saving, etc.). Always use
`monotonic()` to measure elapsed duration.

**Returns:** dict with keys `strategy`, `messages`, `raw_response`,
`elapsed_s`, `prompt_tokens`, `output_tokens`.

---

### `async def main()`

Runs the `for` loop, calls `await call_strategy()` for each strategy,
collects results, parses, and writes the report.

```python
async def main():
    for strategy_name, prompt_fn in STRATEGIES.items():
        result = await call_strategy(strategy_name, snippet['snippet'])
        result['extracted'] = parse_response(result['raw_response'])
        results.append(result)
    ...
```

This is functionally identical to the synchronous `for` loop in script 01.
The event loop runs `main()` start to finish with no interleaving.

---

### `parse_response(text) -> dict | None`

Three-step fallback — identical to all other scripts:

| Step | Action | Handles |
|---|---|---|
| 1 | Strip fences, `json.loads()` | Clean or fenced JSON |
| 2 | Regex find last `{...}` block, `json.loads()` | CoT (reasoning before JSON) |
| 3 | Return `None` | Parse failure |

---

### `esc(text) -> str`

HTML-escapes `&`, `<`, `>`, `"`. Applied to all user-supplied content
before inserting into HTML. `&` must come first to avoid double-escaping.

---

### `messages_to_html(msgs) -> str`

Color-coded chat bubbles. Role → color (consistent across all scripts):

| Role | Background | Label |
|---|---|---|
| `system` | Yellow `#fff3cd` | SYSTEM |
| `user` | Blue `#cfe2ff` | USER |
| `assistant` | Green `#d1e7dd` | ASSISTANT |

---

### `strategy_badge(name) -> str`

Colored pill `<span>` using `STRATEGY_COLORS`. Used in card headers.

---

### `build_html_report(snippet, results) -> str`

Assembles the HTML page. Two sections:

1. Input job posting
2. One card per strategy: header bar (badge + tokens + latency + "awaited individually" label), prompt bubbles, raw response, parsed result

The report includes a blue callout explaining why the calls are still
sequential, and an orange callout previewing what `simple_example_03` changes.

---

### `asyncio.run(main())`

`asyncio.run()` is the synchronous bridge:
1. Creates a new event loop
2. Runs `main()` inside it until completion
3. Closes the event loop

Everything async lives inside `main()`. Without `asyncio.run()` you cannot
call an async function from a regular synchronous script entry point.

---

## Output: `simple_example_02_report.html`

| Section | Contents |
|---|---|
| Blue callout | Explains async-sequential: why `await` in a loop is still sequential |
| Orange callout | Previews the gather() change in script 03 |
| 1 · Input Data | Job posting text |
| 2 · Strategy Results | Four cards — prompt + raw response + parsed result, plus token counts and latency per call |

**What to look for:**

- The card headers say `"awaited individually"` — that label changes to
  `"concurrent via gather()"` in `simple_example_03_report.html`.
- Latency values add up to roughly the total runtime, because calls are sequential.
- The structured strategy card shows two bubbles (SYSTEM + USER); all others show one.
- The cot card response is longer than the others — reasoning text before the JSON.

---

## What this script does NOT do

| Missing feature | Where it's introduced |
|---|---|
| Concurrent API calls | `simple_example_03.py` (`asyncio.gather`) |
| Multiple snippets | `intermediate_example.py` |
| Golden set / accuracy scoring | `intermediate_example.py` |
| Cost tracking in USD | `assignment_template.py` |
| LLM-as-judge scoring | `assignment_template.py` |
| JSON and CSV output | `assignment_template.py` |
