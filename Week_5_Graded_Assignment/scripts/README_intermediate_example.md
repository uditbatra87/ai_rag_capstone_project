# intermediate_example.py — Async + Full Dataset + Accuracy Scoring

**Position in the learning series:** Step 3 of 4  
**Complexity:** Intermediate  
**API style:** Asynchronous (`AsyncOpenAI` + `asyncio.gather`)  
**Scope:** 10 snippets × 4 strategies = 40 concurrent calls  
**Output:** `intermediate_example_report.html`

---

## What this script teaches

The bridge between the simple examples and `assignment_template.py`. Three new things are added on top of `simple_example_02`:

1. **Scale to the full dataset** — all 10 snippets, all 4 strategies, 40 calls at once
2. **Golden answers** — load `golden_set.jsonl` and match each result against the correct answer
3. **Accuracy scoring** — deterministic field-by-field comparison, no second LLM needed

The HTML report gains two new sections: a summary table and a per-row detail table, in addition to the full prompt/response trace.

---

## Prerequisites

```bash
pip install openai pandas
export OPENAI_API_KEY=sk-...   # Windows: $env:OPENAI_API_KEY = "sk-..."
```

Both `data/job_snippets.jsonl` and `data/golden_set.jsonl` must be present.

---

## How to run

```bash
python intermediate_example.py
```

---

## Data flow

```
data/job_snippets.jsonl   data/golden_set.jsonl
         │                        │
         ▼                        ▼
    snippets list           golden dict
    (10 items)            { id → gold_row }
         │
         ▼
  asyncio.run(main())
         │
         ▼
  run_all()  ─── 40 coroutines launched via asyncio.gather()
         │        (10 snippets × 4 strategies)
         ▼
  for each result:
    parse_response(raw_response)   → extracted dict or None
    score_accuracy(extracted, gold) → integer 0–3
         │
         ▼
  build_html_report(scored)
    ├─ summary table  (accuracy + parse rate per strategy)
    ├─ detail table   (per-row ✔/✘ comparison)
    └─ full trace     (snippet → prompt → response → parsed)
         │
         ▼
  Write intermediate_example_report.html
```

---

## Function reference

### Module-level constants

| Name | Value | Purpose |
|---|---|---|
| `MODEL` | `'gpt-4o-mini'` | Model used for all 40 calls |
| `TEMPERATURE` | `0.0` | Deterministic output |
| `DATA_DIR` | `Path('data')` | Root data directory |
| `STRATEGY_COLORS` | dict | Hex colors for strategy badges in HTML |

---

### Data loading (module level)

```python
snippets = [json.loads(line) for line in ...]
golden   = {row['id']: row for row in ...}
```

**`snippets`** is a list of dicts, one per job posting. Each dict has at minimum:
- `id` — a unique string identifier (e.g., `"job_001"`)
- `snippet` — the raw job posting text

**`golden`** is a dict keyed by `id`. Each value is a dict with the correct answers:
- `company` — correct company name
- `role` — correct job title
- `years_experience_required` — correct integer, or `None`/`null`

Loading both at module level (not inside `main()`) means they're available to all functions and the HTML builder without being passed as arguments everywhere.

---

### Prompt strategy functions

All four are identical in purpose and structure to `simple_example_02.py`. The only difference is that these versions also receive the snippet as a full dict in `call_one()` so the `snippet_id` can be threaded through to the result.

| Function | Strategy | Messages structure |
|---|---|---|
| `prompt_zero_shot(text)` | Zero-shot | 1 user message |
| `prompt_few_shot(text)` | Few-shot with 3 examples | 1 user message (long) |
| `prompt_structured(text)` | Persona + schema in system message | 1 system + 1 user |
| `prompt_cot(text)` | Chain-of-thought reasoning | 1 user message with numbered steps |

See `README_simple_example_02.md` for a detailed breakdown of what each strategy does and why.

---

### `async def call_one(strategy_name: str, snippet: dict) -> dict`

**What it does:** Makes one API call and returns a result dict that includes the snippet ID.

```python
async def call_one(strategy_name: str, snippet: dict) -> dict:
    messages = STRATEGIES[strategy_name](snippet['snippet'])
    resp = await client.chat.completions.create(
        model=MODEL, messages=messages, temperature=TEMPERATURE,
    )
    return {
        'strategy':     strategy_name,
        'snippet_id':   snippet['id'],    # ← new: needed to look up gold answer
        'messages':     messages,
        'raw_response': resp.choices[0].message.content or '',
    }
```

The critical addition vs `simple_example_02.call_strategy()` is `'snippet_id': snippet['id']`. Without this, you couldn't match results back to their golden answers after all 40 calls complete in arbitrary order.

**Parameters:**
- `strategy_name` — key into `STRATEGIES`
- `snippet` — full snippet dict (uses `snippet['snippet']` for the text, `snippet['id']` for the id)

**Returns:** `dict` with keys `strategy`, `snippet_id`, `messages`, `raw_response`

---

### `async def run_all() -> list[dict]`

**What it does:** Builds 40 coroutines and fires them all at once with `asyncio.gather`.

```python
async def run_all() -> list[dict]:
    tasks = [
        call_one(strategy_name, snippet)
        for snippet in snippets          # outer loop: 10 snippets
        for strategy_name in STRATEGIES  # inner loop: 4 strategies
    ]
    return list(await asyncio.gather(*tasks))
```

The nested list comprehension creates 40 coroutines. `asyncio.gather(*tasks)` starts all 40 simultaneously. The total wall-clock time is approximately equal to the slowest single call — not 40× the average.

**Returns:** `list[dict]` — 40 result dicts in completion order (not necessarily snippet/strategy order, which is why `snippet_id` is included)

---

### `parse_response(text: str) -> dict | None`

Identical to the previous scripts. Three-step fallback:

1. Strip markdown fences, try `json.loads()`
2. Find last `{...}` block with regex, try `json.loads()`
3. Return `None`

---

### `score_accuracy(extracted: dict | None, gold: dict) -> int`

**What it does:** Compares the model's extracted fields to the golden answer and returns a score from 0 to 3 — one point per correctly extracted field.

This is deterministic (no LLM involved) and runs instantly.

```python
def score_accuracy(extracted: dict | None, gold: dict) -> int:
    if extracted is None:
        return 0
    score = 0
    # company: case-insensitive, whitespace-trimmed string comparison
    # role: same
    # years_experience_required: integer comparison; null == null
    return score
```

**Field-by-field logic:**

**Company and Role:**
```python
str(extracted.get('company') or '').strip().lower()
==
str(gold.get('company') or '').strip().lower()
```
Both sides are converted to lowercase strings and stripped of surrounding whitespace. This handles cases where the model returns `"OpenAI"` vs the gold `"openai"`, or adds a trailing space.

**Years experience:**
```python
gold_years = gold.get('years_experience_required')
ext_raw    = extracted.get('years_experience_required')

if gold_years is None:
    if ext_raw is None:
        score += 1          # correct: null → null
else:
    ext_years = int(ext_raw) if ext_raw is not None else None
    if ext_years == int(gold_years):
        score += 1
```

Two separate branches:
- If the gold answer is `null`, the model must also return `null` (or Python `None`)
- If the gold answer is an integer, cast both sides to `int` before comparing — handles models that return `"3"` (string) instead of `3` (integer)

**Parameters:**
- `extracted` — the dict returned by `parse_response()`, or `None`
- `gold` — the full golden answer dict for this snippet

**Returns:** `int` in range 0–3

---

### HTML helper functions

These are identical to `simple_example_02.py` in purpose but all live in the same file here for self-containment.

#### `esc(text: str) -> str`
HTML-escapes `&`, `<`, `>`, `"`. Applied to all user-supplied text before inserting into HTML.

#### `messages_to_html(msgs: list[dict]) -> str`
Renders messages as color-coded chat bubbles (yellow=system, blue=user, green=assistant).

#### `strategy_badge(name: str) -> str`
Returns a colored pill `<span>` for a strategy name using `STRATEGY_COLORS`.

#### `acc_color(acc: int) -> str`
Returns a hex color based on accuracy score:
- `3` → green `#198754` (perfect)
- `1–2` → orange `#fd7e14` (partial)
- `0` → red `#dc3545` (failed)

#### `field_cell(ext_val, gold_val) -> str`
Returns a `<td>` HTML element that shows:
- ✔ in green if the extracted value matches gold
- ✘ in red if it doesn't
- The extracted value on the first line
- `gold: <value>` in grey below it

This is used in the per-row detail table to make it immediately obvious which fields were right and wrong.

---

### `build_html_report(scored: list[dict]) -> str`

**What it does:** Builds the complete HTML report. This is the most complex function in the script.

It has three internal sections:

---

#### Section 1: Summary table

```python
df = pd.DataFrame(scored)
summary = df.groupby('strategy').agg(
    accuracy=('accuracy', 'mean'),
    parse_success=('parse_success', 'mean'),
).round(3)
```

Uses `pandas.groupby` to aggregate per-row scores into per-strategy means. Two metrics:

- **Accuracy (mean/3):** Average field-match score across all 10 snippets. A score of 3.0 means the strategy extracted all three fields correctly on every snippet.
- **Parse Rate:** Fraction of responses that could be parsed into a dict at all. A rate < 1.0 means some responses were unparsable (the model returned prose, or malformed JSON).

The best-performing strategy row is bolded in the table.

The inline progress bars are rendered as pure HTML `<div>` elements — no external charts library needed.

---

#### Section 2: Per-row detail table

Iterates `scored` sorted by `(snippet_id, strategy)` so the table reads in a predictable order.

For each row, calls `field_cell()` for company, role, and years to produce ✔/✘ cells.

---

#### Section 3: Full prompt/response trace

Grouped by snippet ID. For each snippet:
1. Shows the snippet ID badge and the gold answer inline
2. Shows the job posting text in a left-bordered quote block
3. For each strategy: shows prompt bubbles → raw response → parsed result

The trace is the most verbose section but the most educational — it shows exactly why a strategy scored 2/3 instead of 3/3 on a particular snippet.

**Parameters:**
- `scored` — list of result dicts after parsing and accuracy scoring have been applied

**Returns:** Complete HTML document string

---

### `async def main()`

**What it does:** Orchestrates the full run.

```python
async def main():
    raw_results = await run_all()          # 40 concurrent API calls
    scored = []
    for row in raw_results:
        gold             = golden[row['snippet_id']]
        extracted        = parse_response(row['raw_response'])
        row['extracted'] = extracted
        row['parse_success'] = extracted is not None
        row['accuracy']  = score_accuracy(extracted, gold)
        scored.append(row)
    # terminal summary
    df = pd.DataFrame(scored)
    print(df.groupby('strategy')['accuracy'].mean().round(3).to_string())
    # write report
    Path('intermediate_example_report.html').write_text(
        build_html_report(scored), encoding='utf-8'
    )
```

The scoring loop runs **after** all API calls complete. This is correct because `score_accuracy` is synchronous and instant — there's no benefit to running it concurrently.

---

## Output file: `intermediate_example_report.html`

| Section | Contents |
|---|---|
| Green callout | Explains what's new vs `simple_example_02` |
| 1 · Strategy Accuracy Summary | Table with accuracy bars and parse rate per strategy |
| 2 · Per-Row Scored Detail | Full 40-row table with ✔/✘ per field and total accuracy |
| 3 · Full Prompt & Response Trace | Grouped by snippet: posting → prompt per strategy → raw response → parsed result |

---

## What to look for in the report

- **Summary table:** Which strategy has the highest mean accuracy? Is the best strategy always best, or does it vary by snippet type?
- **Detail table:** Look for rows where accuracy is 2/3 — which field did the strategy get wrong? Is it always the same field (e.g., `years_experience_required`)?
- **Trace section:** Find a snippet where strategies disagree. Look at the raw responses to understand why one strategy extracted the field correctly and another didn't.

---

## What this script does NOT do

| Missing feature | Where it's introduced |
|---|---|
| Cost tracking (tokens + $) | `assignment_template.py` |
| Latency measurement | `assignment_template.py` |
| LLM-as-judge scoring | `assignment_template.py` |
| CSV output | `assignment_template.py` |
| JSON output | `assignment_template.py` |
