# simple_example_03.py — Async Concurrent

**Position in the learning series:** Step 3 of 5  
**Complexity:** Intermediate  
**API style:** Asynchronous (`AsyncOpenAI`), concurrent execution  
**Scope:** 1 snippet × 4 strategies — all launched **simultaneously** via `asyncio.gather()`  
**Output:** `simple_example_03_report.html`

---

## What this script teaches

This script has one job: show the concrete speedup that `asyncio.gather()` delivers
over sequential `await` calls.

Everything except `run_all_strategies()` is identical to `simple_example_02`. The
strategy functions, `call_strategy()`, `parse_response()`, and all HTML helpers are
unchanged. The only diff is how the four coroutines are dispatched.

After this script you will understand:

- What a *coroutine object* is and why creating one does not start execution
- Why `asyncio.gather()` makes multiple coroutines run in parallel
- Why results come back in input order, not completion order
- Why the total runtime collapses from ~4× to ~1× the average call latency
- How this pattern scales to 40 calls in `intermediate_example.py`

---

## The exact diff from simple_example_02

```diff
  # simple_example_02 — sequential (await in a for loop)
- results = []
- for name in STRATEGIES:
-     result = await call_strategy(name, snippet['snippet'])
-     results.append(result)

  # simple_example_03 — concurrent (asyncio.gather)
+ tasks = [call_strategy(name, snippet['snippet']) for name in STRATEGIES]
+ results = await asyncio.gather(*tasks)
```

Three lines become two. That is the entire change.

---

## Execution timeline comparison

```
Script 02 — sequential await in a for loop:

  zero_shot:  [send]────[wait ~1s]────[recv]
  few_shot:                                  [send]────[wait ~1s]────[recv]
  structured:                                                              [send]────...
  cot:                                                                               ...

  Total ≈ 4 × average latency (~4–6 seconds)


Script 03 — asyncio.gather():

  zero_shot:  [send]────[wait ~1s]────[recv]
  few_shot:   [send]────[wait ~1s]────[recv]   ← all four in-flight at the same time
  structured: [send]────[wait ~1s]────[recv]
  cot:        [send]────[wait ~1s]────[recv]

  Total ≈ 1 × slowest single call (~1–2 seconds)
```

The terminal output in script 03 prints all four results at once after a single wait,
instead of one result appearing every ~1 second.

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
python simple_example_03.py
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
  asyncio.run(main())
        │
        ▼
  run_all_strategies(snippet_text)
        │
        ▼
  tasks = [call_strategy('zero_shot', text),   ← list comprehension
           call_strategy('few_shot',  text),     creates coroutine objects
           call_strategy('structured',text),     (not running yet)
           call_strategy('cot',       text)]
        │
        ▼
  await asyncio.gather(*tasks)    ← all four start simultaneously
        │                            event loop switches between them
        │                            during each network wait
        ▼
  4 results (in input order, regardless of which finished first)
        │
        ▼
  parse_response() on each raw_response
        │
        ▼
  build_html_report(snippet, results)
        │
        ▼
  Write simple_example_03_report.html
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

Identical to `simple_example_01` and `simple_example_02`. See
`README_simple_example_01.md` for full documentation.

| Function | Strategy | Messages |
|---|---|---|
| `prompt_zero_shot(text)` | Ask directly, no examples | 1 user |
| `prompt_few_shot(text)` | 3 worked examples | 1 user (longer) |
| `prompt_structured(text)` | System persona + schema | 1 system + 1 user |
| `prompt_cot(text)` | Step-by-step reasoning | 1 user (CoT steps) |

---

### `async def call_strategy(strategy_name, snippet_text) -> dict`

**Identical to `simple_example_02`.** No changes.

The key point about this function in the context of `gather()`:

```python
# This line does NOT run the function body:
coroutine_obj = call_strategy("zero_shot", text)

# This line runs the body and waits for the result:
result = await call_strategy("zero_shot", text)

# gather() runs multiple bodies in parallel and waits for all:
results = await asyncio.gather(
    call_strategy("zero_shot",   text),   # coroutine object 1
    call_strategy("few_shot",    text),   # coroutine object 2
    call_strategy("structured",  text),   # coroutine object 3
    call_strategy("cot",         text),   # coroutine object 4
)
```

Calling `call_strategy(...)` inside the list comprehension creates four
coroutine objects without starting any of them. `gather()` then schedules
and starts all four on the event loop simultaneously.

**Returns:** dict with keys `strategy`, `messages`, `raw_response`,
`elapsed_s`, `prompt_tokens`, `output_tokens`.

---

### `async def run_all_strategies(snippet_text) -> list[dict]`

**The key function of this script.** Everything else is unchanged from script 02.

```python
async def run_all_strategies(snippet_text: str) -> list[dict]:
    # Step 1: create four coroutine objects (none are running yet)
    tasks = [
        call_strategy(name, snippet_text)
        for name in STRATEGIES
    ]

    # Step 2: start all four simultaneously and await all results
    results = await asyncio.gather(*tasks)

    return list(results)
```

**How `asyncio.gather()` works, step by step:**

1. `gather(*tasks)` schedules all four coroutines on the event loop at once.
2. All four coroutines start and run until their first `await` — the HTTP request.
3. While coroutine A is suspended waiting for its response, the event loop
   runs coroutines B, C, and D. They all send their HTTP requests immediately.
4. As each response arrives, the event loop resumes the corresponding coroutine.
5. `gather()` itself awaits until **every** coroutine in the list has completed.
6. Returns a tuple of results in **input order** — zero_shot is always first,
   cot is always last — regardless of which API call finished first.

**Result ordering guarantee:**

This is important. If you iterate over results and assume index 0 is zero_shot,
that assumption holds even if few_shot happened to respond 200ms faster.
`gather()` reorders results to match the input list.

**`*tasks` unpacking:**

`asyncio.gather(*tasks)` is equivalent to:
```python
asyncio.gather(tasks[0], tasks[1], tasks[2], tasks[3])
```
The `*` operator unpacks a list into positional arguments.

**Parameters:** `snippet_text` — raw job posting text passed to all 4 strategies.  
**Returns:** `list[dict]` — 4 result dicts in `STRATEGIES` insertion order.

---

### `parse_response(text) -> dict | None`

Identical to all other scripts. Three-step fallback:

| Step | Action | Handles |
|---|---|---|
| 1 | Strip fences, `json.loads()` | Clean or fenced JSON |
| 2 | Regex find last `{...}` block, `json.loads()` | CoT (reasoning before JSON) |
| 3 | Return `None` | Parse failure |

---

### `esc(text) -> str`

HTML-escapes `&`, `<`, `>`, `"`. Applied to all user content in HTML.

---

### `messages_to_html(msgs) -> str`

Color-coded chat bubbles. system → yellow, user → blue, assistant → green.

---

### `strategy_badge(name) -> str`

Colored pill `<span>` using `STRATEGY_COLORS`.

---

### `build_html_report(snippet, results) -> str`

Assembles the HTML page. New additions vs script 02:

- **Green callout** instead of blue — visually marks this as the gather() script.
- **Code diff block** — shows the exact lines that changed from script 02 to 03,
  with removed lines in red and added lines in green.
- Card headers say `"concurrent via gather()"` instead of `"awaited individually"`.

---

### `async def main()`

```python
async def main():
    t_start = time.monotonic()
    results = await run_all_strategies(snippet['snippet'])
    total_elapsed = time.monotonic() - t_start
    print(f'All 4 complete in {total_elapsed:.2f}s '
          f'(vs ~{sum(r["elapsed_s"] for r in results):.2f}s sequential)')
    ...
```

The terminal output prints the total elapsed time and the sum of individual
latencies side by side. The gap between them is the concurrency speedup:

```
All 4 complete in 1.43s (vs ~4.82s sequential)
```

---

## Output: `simple_example_03_report.html`

| Section | Contents |
|---|---|
| Green callout | Explains `asyncio.gather()` concurrency |
| Code diff block | Exact lines that changed from script 02 to 03 |
| 1 · Input Data | Job posting text |
| 2 · Strategy Results | Four cards — prompt + response + parsed result, with `"concurrent via gather()"` in the header |

**What to look for:**

- Individual latencies in each card should be similar to script 02 card latencies.
  The total runtime printed in the terminal should be much shorter.
- The four cards show the same prompt/response content as script 02. Only the
  execution model changed, not what was sent or received.
- Compare the terminal output of script 02 (results appear one per second) to
  script 03 (all four appear at once after one wait).

---

## How this scales to intermediate_example.py

`intermediate_example.py` uses the exact same gather pattern, scaled up:

```python
# script 03: 4 tasks (1 snippet × 4 strategies)
tasks = [call_strategy(name, snippet_text) for name in STRATEGIES]
results = await asyncio.gather(*tasks)

# intermediate_example: 40 tasks (10 snippets × 4 strategies)
tasks = [
    call_one(strategy_name, snip)
    for snip in snippets          # outer loop: 10 snippets
    for strategy_name in STRATEGIES  # inner loop: 4 strategies
]
results = await asyncio.gather(*tasks)
```

The pattern is identical — just more tasks in the list. gather() doesn't care
whether there are 4 or 40 coroutines; it starts them all and waits for all.

---

## What this script does NOT do

| Missing feature | Where it's introduced |
|---|---|
| Multiple snippets | `intermediate_example.py` |
| Golden set / accuracy scoring | `intermediate_example.py` |
| Cost tracking in USD | `assignment_template.py` |
| Latency measurement (per call, stored) | `assignment_template.py` |
| LLM-as-judge scoring | `assignment_template.py` |
| JSON and CSV output | `assignment_template.py` |
