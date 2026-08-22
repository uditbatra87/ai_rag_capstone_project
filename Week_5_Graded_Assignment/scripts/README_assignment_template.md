# assignment_template.py — Full Pipeline (MP1 Prompt Lab)

**Position in the learning series:** Step 4 of 4 (final)  
**Complexity:** Intermediate–Advanced  
**API style:** Asynchronous (`AsyncOpenAI` + `asyncio.gather`)  
**Scope:** 10 snippets × 4 strategies = 40 extraction calls + 40 LLM judge calls  
**Outputs:** `mp1_report.html`, `mp1_results.json`, `mp1_results.csv`

---

## What this script adds on top of `intermediate_example.py`

| Feature | intermediate_example | assignment_template |
|---|---|---|
| Async API calls | ✔ | ✔ |
| All 4 strategies | ✔ | ✔ |
| Full dataset (10 snippets) | ✔ | ✔ |
| Accuracy scoring | ✔ | ✔ |
| **Cost tracking** | ✘ | ✔ |
| **Latency measurement** | ✘ | ✔ |
| **LLM-as-judge scoring** | ✘ | ✔ |
| **JSON output** | ✘ | ✔ |
| **CSV output** | ✘ | ✔ |
| **3-section HTML report** | ✘ (trace only) | ✔ (summary + detail + trace) |

---

## Prerequisites

```bash
pip install openai pandas tabulate
export OPENAI_API_KEY=sk-...   # Windows: $env:OPENAI_API_KEY = "sk-..."
```

Both `data/job_snippets.jsonl` and `data/golden_set.jsonl` must exist.

> **Note:** `tabulate` is needed for `summary.to_markdown()` inside the pandas call. If you don't need the terminal markdown output you can remove that line.

---

## How to run

```bash
python assignment_template.py
```

Expected runtime: 30–90 seconds depending on API latency. There are 80 API calls in total (40 extraction + 40 judge).

---

## Data flow

```
data/job_snippets.jsonl   data/golden_set.jsonl
         │                        │
         ▼                        ▼
    snippets list           golden dict
         │
         ▼
  asyncio.run(main())
         │
         ├─── run_all()  ─────────────────────────────── 40 concurrent calls
         │     └─ run_one(strategy, snippet)
         │         ├─ build prompt (STRATEGIES[strategy](text))
         │         ├─ await API call   ← measures latency, captures usage
         │         └─ parse_response() + compute cost_usd
         │
         ├─── for each result:
         │     ├─ score_accuracy(extracted, gold)  → 0–3
         │     └─ build score_llm_judge coroutine
         │
         ├─── asyncio.gather(judge_tasks)  ──────────── 40 concurrent judge calls
         │
         ├─── pandas aggregation → summary DataFrame
         │
         ├─── Write mp1_results.json
         ├─── Write mp1_results.csv
         └─── Write mp1_report.html
```

---

## Function reference

### Module-level constants

| Name | Value | Purpose |
|---|---|---|
| `MODEL` | `'gpt-4o-mini'` | Extraction model |
| `JUDGE_MODEL` | `'gpt-4o'` | Judge model (stronger, more expensive) |
| `TEMPERATURE` | `0.0` | Deterministic outputs for both models |
| `RATES` | dict | Token cost rates in $/token for both models |

**`RATES` structure:**

```python
RATES = {
    'gpt-4o-mini': {'in': 0.15 / 1_000_000, 'out': 0.60 / 1_000_000},
    'gpt-4o':      {'in': 2.50 / 1_000_000, 'out': 10.00 / 1_000_000},
}
```

Input tokens (prompt) and output tokens (completion) are priced separately. The `'in'` rate applies to the prompt, `'out'` to the model's response.

---

### Prompt strategy functions

Identical in logic to `intermediate_example.py` but with slightly richer examples (longer worked examples in `prompt_few_shot`, explicit conversion rule for word-numbers in `prompt_structured`).

| Function | Strategy | What makes it distinctive |
|---|---|---|
| `prompt_zero_shot(snippet_text)` | Zero-shot | Minimal — just the question |
| `prompt_few_shot(snippet_text)` | Few-shot | 3 examples cover integer / zero / null cases |
| `prompt_structured(snippet_text)` | Structured | System persona + explicit schema + hard rules (no fences, no fabrication, word→int) |
| `prompt_cot(snippet_text)` | Chain-of-thought | Numbered reasoning steps, ends with JSON |

**`STRATEGIES` dict** maps names to functions. All loops in the script iterate this dict, so adding a fifth strategy only requires adding one entry here.

---

### `parse_response(text: str) -> dict | None`

**What it does:** Extracts a Python dict from the model's raw response text.

Three-step fallback:

1. **Strip fences and parse directly.**  
   `re.sub(r'```(?:json)?\s*', '', text)` removes ` ```json ` and ` ``` ` markers. Then `json.loads()`.

2. **Regex search for last `{...}` block.**  
   `re.finditer(r'\{[^{}]+\}', text, re.DOTALL)` finds all single-depth JSON objects in the text. Iterating from last to first handles CoT responses where the JSON appears at the end after reasoning text.

3. **Return `None`** if nothing parses.

**Parameters:**
- `text` — raw model response string

**Returns:** `dict` or `None`

---

### `async def run_one(strategy_name: str, snippet: dict) -> dict`

**What it does:** Makes one extraction API call and captures everything about it: the response, the cost, and the latency.

```python
async def run_one(strategy_name: str, snippet: dict) -> dict:
    messages = STRATEGIES[strategy_name](snippet['snippet'])

    t_start  = time.monotonic()
    resp     = await client.chat.completions.create(
        model=MODEL, messages=messages, temperature=TEMPERATURE,
    )
    latency_s = time.monotonic() - t_start

    raw_text = resp.choices[0].message.content or ''

    usage          = resp.usage
    prompt_tokens  = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    cost_usd = (
        prompt_tokens    * RATES[MODEL]['in']
        + completion_tokens * RATES[MODEL]['out']
    )

    return {
        'strategy':     strategy_name,
        'snippet_id':   snippet['id'],
        'raw_response': raw_text,
        'extracted':    parse_response(raw_text),
        'cost_usd':     cost_usd,
        'latency_s':    latency_s,
    }
```

**Latency measurement:**  
`time.monotonic()` is used (not `time.time()`) because monotonic clocks never go backwards — they're not affected by system clock adjustments, making them correct for measuring elapsed time.

**Cost calculation:**  
`resp.usage` contains `prompt_tokens` (the input) and `completion_tokens` (the output). Multiplying each by the per-token rate and summing gives the cost in USD for this single call. Typical cost per call: $0.00001–$0.0001.

**Parameters:**
- `strategy_name` — key into `STRATEGIES`
- `snippet` — full snippet dict

**Returns:** `dict` with keys: `strategy`, `snippet_id`, `raw_response`, `extracted`, `cost_usd`, `latency_s`

---

### `async def run_all() -> list[dict]`

**What it does:** Fires all 40 extraction calls concurrently.

```python
async def run_all() -> list[dict]:
    tasks = [
        run_one(strategy_name, snippet)
        for snippet in snippets
        for strategy_name in STRATEGIES
    ]
    return await asyncio.gather(*tasks)
```

40 coroutines, all in-flight at the same time. Returns when the last one finishes.

**Returns:** `list[dict]` — 40 result dicts

---

### `score_accuracy(extracted: dict | None, gold: dict) -> int`

**What it does:** Deterministic field comparison. Returns 0–3.

Identical logic to `intermediate_example.py`:
- Company and role: case-insensitive, whitespace-trimmed string match
- Years: integer comparison; handles `None`/`null` correctly; casts string numbers to `int`

An additional guard: `if ext_company and ext_company == gold_company` — the `and ext_company` check ensures we don't give a point for an empty string matching an empty gold value.

**Parameters:**
- `extracted` — dict from `parse_response()`, or `None`
- `gold` — golden answer dict from `golden[snippet_id]`

**Returns:** `int` 0–3

---

### `async def score_llm_judge(snippet_text: str, extracted: dict | None, gold: dict) -> int`

**What it does:** Uses `gpt-4o` to judge the quality of an extraction on a 1–4 rubric.

This is the key addition over `intermediate_example.py`. A language model judge can catch things that string comparison misses — partial matches, near-correct answers, fabricated fields.

**The judge prompt:**

```
You are a strict evaluator. A model was asked to extract three fields from a job posting.

Job posting:
{snippet_text}

Golden (correct) answer:
{gold_str}

Model extraction:
{extracted_str}

Score the model extraction on a scale of 1–4 using this rubric:
  4 — all three fields correct
  3 — two of three correct, no fabricated data
  2 — one of three correct, or a field was fabricated
  1 — none correct, failed to parse, or null output

Reply with a single integer (1, 2, 3, or 4) and nothing else.
```

**Rubric design:**
- `4` is equivalent to accuracy score 3 — perfect extraction
- `3` credits partial success without penalizing for honesty
- `2` explicitly penalizes fabrication — making up data not present in the posting is worse than admitting uncertainty
- `1` is the floor — total failure

**Parsing the judge response:**

```python
score = int(raw[0])           # take first character (the digit)
return max(1, min(4, score))  # clamp to valid range
```

`max(1, min(4, score))` ensures the function always returns a valid integer even if the judge model adds unexpected text.

**Why `max_tokens=5`?** The judge should reply with a single digit. Limiting to 5 tokens prevents the model from writing an explanation and keeps judge call costs near zero.

**Parameters:**
- `snippet_text` — the raw job posting text (shown to the judge for context)
- `extracted` — the model's extraction dict, or `None`
- `gold` — the golden answer dict

**Returns:** `int` 1–4

---

### `async def main()` — the orchestration layer

**Phase 1: Extraction**

```python
results = await run_all()   # 40 calls in parallel
```

**Phase 2: Deterministic scoring + building judge tasks**

```python
for row in results:
    gold = golden[row['snippet_id']]
    row['parse_success'] = row['extracted'] is not None
    row['accuracy'] = score_accuracy(row['extracted'], gold)
    judge_tasks.append(score_llm_judge(...))
```

Judge coroutines are created here but not yet started. This lets us launch all 40 at once in the next step.

**Phase 3: LLM judge (parallel)**

```python
judge_scores = await asyncio.gather(*judge_tasks)   # 40 judge calls in parallel
```

Batching judge calls with `gather()` is identical to how extraction calls are batched. Total judge time ≈ slowest single judge call, not 40× average.

**Phase 4: Merge + pandas aggregation**

```python
summary = df.groupby('strategy').agg(
    accuracy=('accuracy', 'mean'),
    parse_success=('parse_success', 'mean'),
    llm_judge_score=('llm_judge_score', 'mean'),
    cost_usd=('cost_usd', 'sum'),        # sum (total spend per strategy)
    latency_s=('latency_s', 'median'),   # median (robust to outliers)
).round(3)
```

Note: cost is **summed** (total spend), while accuracy and judge score are **averaged** (mean across snippets), and latency uses the **median** (p50 is more robust to occasional slow calls than the mean).

**Phase 5: Three output files**

---

### Output: `mp1_results.json`

```python
json.dumps(
    [{k: v for k, v in row.items() if k != 'raw_response'} for row in scored],
    indent=2,
)
```

One JSON object per row, excluding `raw_response` (too bulky). Fields per row:

| Field | Type | Description |
|---|---|---|
| `strategy` | str | Strategy name |
| `snippet_id` | str | Snippet identifier |
| `extracted` | dict or null | Parsed extraction result |
| `cost_usd` | float | Cost of this single call |
| `latency_s` | float | Wall-clock time in seconds |
| `parse_success` | bool | Whether extraction was parsable |
| `accuracy` | int | 0–3 field match score |
| `llm_judge_score` | int | 1–4 judge score |

---

### Output: `mp1_results.csv`

The `extracted` dict column is **exploded** into three flat columns:

```python
extracted_df = csv_df['extracted'].apply(
    lambda x: pd.Series(x) if isinstance(x, dict)
              else pd.Series({'company': None, 'role': None, 'years_experience_required': None})
).rename(columns=lambda c: f'extracted_{c}')
```

This produces `extracted_company`, `extracted_role`, `extracted_years_experience_required` columns, making the CSV directly usable in Excel or any BI tool without needing to parse nested JSON.

---

### Output: `mp1_report.html`

Three sections, linked from a table of contents at the top:

| Section | Contents |
|---|---|
| 1 · Strategy Comparison Summary | One row per strategy: accuracy bar, judge score bar, parse rate bar, total cost, latency p50. Best strategy highlighted in bold. |
| 2 · Per-Row Scored Detail | 40-row table: ✔/✘ per field vs gold, accuracy /3, judge score /4, cost per call, latency. |
| 3 · Full Prompt & Response Trace | Grouped by snippet: gold answer banner, job posting text, then for each strategy: exact prompt bubbles + raw response + score inline in the header bar. |

---

### HTML builder functions

These all live in the module scope (not inside `main()`) so they can be called from `_build_html_report()`.

#### `_bar(value: float, max_value: float, color: str) -> str`

Renders a pure-HTML inline progress bar with a value label.

```python
pct = min(100.0, 100.0 * value / max_value) if max_value else 0
```

The bar width is a CSS percentage. No external chart library required — just two nested `<div>` elements with inline styles.

**Parameters:**
- `value` — the number to display
- `max_value` — sets the 100% mark (e.g., `3.0` for accuracy, `4.0` for judge score)
- `color` — hex color for the filled portion

---

#### `_strategy_badge(name: str) -> str`

Returns a colored pill `<span>`. Same purpose as `strategy_badge()` in the intermediate example, but prefixed with `_` to signal it's a private helper in this module.

---

#### `_esc(text: str) -> str`

HTML-escapes `&`, `<`, `>`, `"`. Same as `esc()` in the simpler scripts, prefixed with `_`.

---

#### `_messages_to_html(messages: list[dict]) -> str`

Renders a messages list as color-coded chat bubbles. Same as `messages_to_html()` in the simpler scripts, prefixed with `_`. The `system` / `user` / `assistant` role coloring is consistent across all scripts.

---

#### `_build_html_report(summary, scored, snippets, golden) -> str`

The main HTML assembly function. Combines all helper outputs into a complete HTML document.

Internal structure:

1. **`summary_rows`** — iterates the pandas summary DataFrame to build table rows with `_bar()` calls and bold-best-strategy logic
2. **`detail_rows`** — iterates `scored` sorted by `(snippet_id, strategy)` to build the 40-row detail table using `cell()` (a closure defined inside the loop for compact ✔/✘ cell generation)
3. **`trace_map`** — a `{snippet_id: {strategy: row}}` dict built to group results by snippet for the trace section
4. **`trace_html`** — iterates `trace_map` to build the full trace section, regenerating the prompt messages from `STRATEGIES[strat_name](snippet_text)` (same prompt that was sent) to render them as bubbles

**Why regenerate prompts in the trace section?**  
The extraction results (`run_one`) don't store the messages list to keep the result dict lean (the messages are reconstructable from the strategy function + snippet text). The trace section reconstructs them on the fly using the same deterministic prompt functions.

**Parameters:**
- `summary` — pandas DataFrame from `df.groupby('strategy').agg(...)`
- `scored` — list of 40 fully scored result dicts
- `snippets` — original snippets list (for text lookup)
- `golden` — golden answer dict (for gold value display)

**Returns:** Complete HTML document string

---

## Understanding the two-wave async design

The script makes two separate waves of parallel API calls:

**Wave 1 — Extraction (40 calls, `gpt-4o-mini`)**
```python
results = await run_all()
```

**Wave 2 — Judging (40 calls, `gpt-4o`)**
```python
judge_scores = await asyncio.gather(*judge_tasks)
```

These are sequential waves, not interleaved. Wave 2 only starts after all of Wave 1 is complete, because the judge needs the extraction result to evaluate. Within each wave, all calls run concurrently.

This is the standard pattern for LLM evaluation pipelines:
```
generate → score_deterministic → score_llm_judge
```

---

## Cost breakdown

For a typical run (approximate):

| Component | Model | Calls | Approx. cost |
|---|---|---|---|
| Extraction | gpt-4o-mini | 40 | $0.002–0.005 |
| LLM judge | gpt-4o | 40 | $0.05–0.15 |
| **Total** | | **80** | **~$0.05–0.15** |

The judge calls dominate cost because `gpt-4o` is ~17× more expensive per token than `gpt-4o-mini`.

---

## What to look for in the report

**Summary table:**
- Does `accuracy` and `judge_score` agree on which strategy is best? If they diverge, the judge is catching something the string comparison misses.
- Is `parse_success` < 1.0 for any strategy? That means some responses couldn't be parsed — a prompt design problem.
- Does `cot` have higher latency and cost? It should — longer responses mean more output tokens.

**Detail table:**
- Which field (`company`, `role`, `years`) fails most often? `years_experience_required` is typically the hardest.
- Find rows where `accuracy=2` but `judge_score=4`. The judge may be giving partial credit for a close match that fails exact string comparison.

**Trace section:**
- Find a snippet where strategies disagree. Look at the raw responses to understand why.
- For CoT: notice how the reasoning text varies even when the final JSON is the same.
