# simple_example_01.py — Synchronous, All Four Strategies

**Position in the learning series:** Step 1 of 4  
**Complexity:** Beginner  
**API style:** Synchronous (blocking)  
**Scope:** 1 snippet × 4 strategies — called **one at a time** in a `for` loop  
**Output:** `simple_example_01_report.html`

---

## What this script teaches

This script is the entry point to the series. It introduces all four prompting
strategies on a single job posting, while keeping the execution model as simple
as possible: **one API call at a time, waiting for each to finish before
starting the next**.

By the end of this script you will understand:

- What a `messages` list is and why the Chat API requires one
- How four different prompt shapes produce different model responses for the
  same input
- How synchronous (blocking) API calls work — and why they are simple but slow
- How to extract a JSON dict from a model response that may be wrapped in
  markdown fences or preceded by reasoning text
- How to display the complete pipeline (data → prompt → response → result) in
  an HTML report

---

## The synchronous execution model

```
SEQUENTIAL (this script):

  zero_shot:  [send] ──── wait 1s ──── [receive]
  few_shot:                                       [send] ──── wait 1s ──── [receive]
  structured:                                                                         [send] ──── ...
  cot:                                                                                               ...

  Total time ≈ sum of all 4 call durations (~4–8 seconds)


CONCURRENT (simple_example_02):

  zero_shot:  [send] ──── wait ──── [receive]
  few_shot:   [send] ──── wait ──── [receive]       ← all in-flight simultaneously
  structured: [send] ──── wait ──── [receive]
  cot:        [send] ──── wait ──── [receive]

  Total time ≈ slowest single call (~1–2 seconds)
```

The sequential approach is easier to reason about — you can follow each step
in your head and in the terminal output. That is why it comes first.

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

Run from inside `scripts/` so the relative path `../data/` resolves correctly:

```bash
cd scripts
python simple_example_01.py
```

---

## Data flow

```
data/job_snippets.jsonl
        │
        ▼
  Load all lines → take first snippet
        │
        ▼  ┌──────────────────────────────────────────────────────────┐
           │  for strategy_name, prompt_fn in STRATEGIES.items():     │
           │                                                          │
           │    messages = prompt_fn(snippet['snippet'])              │
           │           ↓                                              │
           │    response = client.chat.completions.create(...)        │
           │           ← blocks until API responds →                  │
           │           ↓                                              │
           │    raw_response = response.choices[0].message.content    │
           │           ↓                                              │
           │    extracted = parse_response(raw_response)              │
           │           ↓                                              │
           │    results.append({...})                                 │
           └──────────────────────────────────────────────────────────┘
        │
        ▼
  Build HTML → write simple_example_01_report.html
```

---

## Module-level constants

| Name | Value | Purpose |
|---|---|---|
| `MODEL` | `'gpt-4o-mini'` | OpenAI model used for all 4 calls |
| `TEMPERATURE` | `0.0` | Deterministic output — same input gives same output |
| `DATA_DIR` | `Path('../data')` | Resolves to `data/` one level above `scripts/` |
| `STRATEGY_COLORS` | dict | Hex color per strategy for HTML report badges |

---

## Function reference

### `prompt_zero_shot(text: str) -> list[dict]`

**Strategy:** Ask directly — no examples, no persona, no rules beyond the output format.

**What the model sees:**
```
[USER] Extract the following three fields from this job posting and
       return them as a JSON object with exactly these keys:
         "company" ...
         "role" ...
         "years_experience_required" ...

       Job posting:
       <text>
```

**Strengths:** Minimal tokens, fast, works well for clear unambiguous postings.  
**Weaknesses:** Most likely to struggle on edge cases — years as words ("three"),
ranges ("3–5 years"), or postings with no years requirement at all.

**Message count:** 1 (user only)

---

### `prompt_few_shot(text: str) -> list[dict]`

**Strategy:** Show three worked examples before asking. The model infers the
rules from the patterns in the examples rather than from explicit instruction.

**Why three examples?** Each covers a different edge case:
- Example 1 — normal integer years (`3+` → `3`)
- Example 2 — zero years (`"fresh graduates welcome"` → `0`)
- Example 3 — null years (`"no years requirement"` → `null`)

Without these examples the model might return `null` for "fresh graduates
welcome" or a string for a number, because it doesn't know how to handle
those cases from the zero-shot instruction alone.

**The trailing `Output:` cue** tells the model to reply immediately with
JSON without adding any conversational text first.

**What the model sees:**
```
[USER] Extract ... Return a JSON object ...

       Example 1:
       Job posting: "TechCorp is hiring ..."
       Output: {"company": "TechCorp", ...}

       Example 2: ...
       Example 3: ...

       Now extract from this job posting:
       <text>
       Output:
```

**Strengths:** Better at edge cases without needing explicit rules.  
**Weaknesses:** Longer prompt → more input tokens. If the examples don't cover a new edge case, it can still fail.

**Message count:** 1 (user only, but longer)

---

### `prompt_structured(text: str) -> list[dict]`

**Strategy:** Separate the instructions (system message) from the data (user
message). Set an expert persona and explicit output rules in the system message.

**Why two messages?** The system message is processed first and acts as persistent
context that the model applies to everything that follows. Putting the schema and
rules there (rather than in the user message) is cleaner and tends to produce more
reliable, consistently-formatted output.

**What the model sees:**
```
[SYSTEM] You are an expert technical recruiter ...
         Always return a JSON object with exactly these fields:
           "company" (string) ...
           "role" (string) ...
           "years_experience_required" (integer or null) ...
         Rules:
           - Output raw JSON only. No markdown fences, no explanation.
           - Never fabricate information ...
           - If years are written as a word ... convert to integer.

[USER]   Extract the structured fields from this job posting:
         <text>
```

**Strengths:** Most consistent output format. Hard rules in the system message
eliminate common failure modes (fences, fabrication, word-numbers).  
**Weaknesses:** Requires thinking carefully about what goes in system vs user.

**Message count:** 2 (system + user)

---

### `prompt_cot(text: str) -> list[dict]`

**Strategy:** Chain-of-thought — instruct the model to reason through each
field step by step before outputting the final JSON.

**How CoT works:** Transformer models produce better answers when they generate
intermediate reasoning tokens. The numbered steps force the model to dedicate
reasoning capacity to each field individually rather than outputting JSON
immediately from pattern-matching.

**What the model sees:**
```
[USER] Job posting:
       <text>

       Think step by step:
       1. Identify the company name.
       2. Identify the job title/role.
       3. Find any mention of required years ...
          - If a range is given ... use the minimum.
          - If years are spelled as a word ... convert to integer.
          - If no years are stated ... use null.
       4. Output a JSON object: {...}

       Show your reasoning for each field, then end with the JSON object.
```

**The model's typical response:**
```
1. The company name is Acme Corp, as stated in the opening sentence.
2. The role is Senior Software Engineer.
3. The posting mentions "5+ years of experience", so the minimum is 5.

{"company": "Acme Corp", "role": "Senior Software Engineer", "years_experience_required": 5}
```

**parse_response() finds the LAST `{...}` block** — which is the JSON after
the reasoning text. That is why the regex fallback iterates in reverse.

**Strengths:** Most accurate on ambiguous or complex postings. The reasoning
is visible — you can see why the model chose each value.  
**Weaknesses:** Longer response → more output tokens → marginally higher cost.
Also requires a parser that handles text-before-JSON format.

**Message count:** 1 (user only, but the response is longer)

---

### `STRATEGIES` dict

```python
STRATEGIES = {
    'zero_shot':  prompt_zero_shot,
    'few_shot':   prompt_few_shot,
    'structured': prompt_structured,
    'cot':        prompt_cot,
}
```

Maps each strategy name (string) to its prompt-building function. The main
`for` loop iterates this dict. To add a fifth strategy: write one new function
and add one entry here — nothing else needs to change.

---

### `parse_response(text: str) -> dict | None`

**What it does:** Extracts a Python dict from the model's raw response.

Three-step fallback:

| Step | What it tries | Handles |
|---|---|---|
| 1 | Strip ` ```json ` / ` ``` ` fences, call `json.loads()` | Clean JSON, fenced JSON |
| 2 | Regex-find last `{...}` block, call `json.loads()` | CoT format (reasoning before JSON) |
| 3 | Return `None` | Total parse failure |

**Why the LAST block?** In a CoT response the JSON appears at the end, after
reasoning text. Earlier `{...}` patterns might be non-JSON fragments inside
the reasoning. Taking the last match gets the right one.

**Returns:** `dict` on success, `None` on failure. Callers check `is not None`
before using the result.

---

### `esc(text: str) -> str`

HTML-escapes `&`, `<`, `>`, `"`. Applied to every piece of user-supplied text
before it is embedded in the HTML report.

**Why `&` first?** If you replaced `<` first, you'd turn `<` into `&lt;` —
and then replacing `&` would turn that into `&amp;lt;` (double-escaping).
Replacing `&` first avoids this problem.

---

### `messages_to_html(msgs: list[dict]) -> str`

Renders a messages list as color-coded bubbles. Role → color:

| Role | Background | Label |
|---|---|---|
| `system` | Yellow `#fff3cd` | SYSTEM |
| `user` | Blue `#cfe2ff` | USER |
| `assistant` | Green `#d1e7dd` | ASSISTANT |

The content is placed inside `<pre>` tags to preserve all whitespace exactly
as it appears in the Python string. Passed through `esc()` for safety.

---

### `strategy_badge(name: str) -> str`

Returns a colored `<span>` pill for a strategy name, using `STRATEGY_COLORS`.
Used in the HTML card headers to make strategies visually distinguishable.

---

### Main script body

This script has no `main()` function and no `if __name__ == '__main__':` guard.
All logic runs at module level, top to bottom. Steps:

1. Load `job_snippets.jsonl`, take `all_snippets[0]`
2. Define the four prompt strategy functions
3. Define `parse_response()`
4. **`for` loop** over `STRATEGIES`: build prompt → call API → parse response → append to `results`
5. Define HTML helper functions
6. Build HTML string from `results` and `snippet`
7. Write `simple_example_01_report.html`

The `for` loop in step 4 is the key point. Each iteration is fully sequential:
```python
for strategy_name, prompt_fn in STRATEGIES.items():
    messages = prompt_fn(snippet['snippet'])     # build prompt
    response = client.chat.completions.create(   # BLOCKS here
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )
    ...                                          # resumes when response arrives
```

---

## Output file: `simple_example_01_report.html`

Open in any browser. Contains two sections:

| Section | Contents |
|---|---|
| 1 · Input Data | The job posting text that was sent to all four strategies |
| 2 · Strategy Results | One card per strategy: prompt bubbles + raw response + parsed result + token count + elapsed time |

The yellow callout at the top of the report explains the sequential execution
model and links it to how `simple_example_02.py` improves on it with async.

---

## What to look for in the report

- **Compare prompts across strategies.** The structured strategy has two
  colored bubbles (SYSTEM + USER); all others have one. This is immediately
  visible in the report.
- **Compare responses across strategies.** The CoT strategy produces a much
  longer response with reasoning text before the JSON. Notice how
  `parse_response()` still extracts the correct dict.
- **Check elapsed time.** The calls run sequentially — you should see roughly
  equal wait time for each, adding up to the total.
- **Look at token counts.** The few-shot prompt uses more input tokens because
  the examples are part of the prompt. The CoT response uses more output tokens
  because the model writes reasoning text.

---

## What this script does NOT do

| Missing feature | Where it's introduced |
|---|---|
| Concurrent API calls | `simple_example_02.py` (all 4 at once with `asyncio.gather`) |
| Full 10-snippet dataset | `intermediate_example.py` |
| Golden set / accuracy scoring | `intermediate_example.py` |
| Cost tracking in USD | `assignment_template.py` |
| LLM-as-judge scoring | `assignment_template.py` |
| JSON and CSV output files | `assignment_template.py` |
