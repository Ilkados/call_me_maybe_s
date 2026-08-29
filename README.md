*This project has been created as part of the 42 curriculum by moboulir.*

# call me maybe — Introduction to Function Calling in LLMs

## Description

`call me maybe` is a function-calling system that translates natural language prompts into structured, machine-executable JSON function calls using a small language model (Qwen3-0.6B) and constrained decoding.

Given a prompt like `"What is the sum of 2 and 3?"`, the system does not answer `5`. Instead it produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The core challenge: small models (0.6B parameters) are unreliable at producing valid JSON when prompted freely — they succeed roughly 30% of the time. This project achieves near-perfect reliability by implementing constrained decoding, a technique that controls token selection at every generation step rather than relying on the model to spontaneously produce correct structure.

## Instructions

### Requirements

- Python 3.10 or later
- `uv` package manager

### Installation

Clone the repository and sync dependencies — this is the only step required:

```bash
uv sync
```

This installs `numpy`, `pydantic`, `torch`, `transformers`, `huggingface_hub`, `flake8`, and `mypy` into a local `.venv`. The `llm_sdk/` directory is copied directly into the repository root alongside `src/` and requires no separate installation.

### Running the program

With default input/output paths:

```bash
uv run python -m src
```

With explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Other Makefile targets

```bash
make install      # uv sync
make run          # run with default paths
make debug        # run under pdb debugger
make clean        # remove __pycache__, .mypy_cache, data/output
make lint         # flake8 + mypy with standard flags
make lint-strict  # flake8 + mypy --strict
```

## Algorithm Explanation

The generation pipeline for each prompt works in two distinct phases.

### Phase 1 — Forced literals (no model involvement)

Known-certain parts of the JSON output (`{"prompt": `, the echoed prompt text, `,"name": "`, etc.) are tokenized and appended directly to the running token sequence without any model call. The model never "chooses" these — they are injected unconditionally. Crucially, they are still added to `self.ids` (the token sequence fed back into every forward pass), because the model needs to see the full context — including the forced scaffold — when making its genuine decisions.

### Phase 2 — Constrained token selection (model + restriction)

For every position where a real decision exists, the pipeline follows the same three-step pattern:

1. **Forward pass**: pass the entire current `self.ids` to the model via `get_logits_from_input_ids`. Self-attention runs over the full sequence and produces a score (logit) for every token in the vocabulary (~150k entries).
2. **Restrict**: compute the set of *legally allowed* tokens at this specific position, then eliminate everything else from consideration. Two mechanisms are used depending on context:
   - **Fancy-indexing** (function name resolution): extract only the candidate token IDs into a small NumPy sub-array, run `argmax` on that sub-array directly, and map the local index back to the real vocabulary ID. Used because the candidate set is a small, pre-known set of integers.
   - **`-inf` masking** (number value generation): build a full vocab-sized array filled with `-inf`, then copy real logit values back only at the allowed positions (digits, decimal point, terminators). `argmax` on the full array can then never select a suppressed position.
3. **Commit**: append the chosen token ID to `self.ids`. This single append is what makes the next forward pass see one more token of context — the autoregressive loop.

### Function name resolution — prefix narrowing

Function names tokenize into multiple tokens (`fn_add_numbers` → `[fn, _add, _numbers]`). Rather than asking the model to predict a whole name at once (impossible — the model only ever produces one token per call), the system resolves names via a `while True` narrowing loop:

- At each step, collect the distinct *next tokens* across all still-possible function names.
- Call `_pick_token` (fancy-indexing argmax) to select among them.
- Eliminate any candidate whose prefix no longer matches.
- Repeat until exactly one candidate survives, then force-complete the rest of its tokens without any further model calls.

### Number value generation — terminator-driven, not parse-driven

The masking at each digit step includes the terminator characters (`,`, `}`) alongside the digits. The number is considered complete the moment the model itself selects one of those terminators — not when the accumulated string happens to already look like a valid number. This is the correct design: stopping on parse-success would truncate multi-digit numbers silently.

### String value generation — unconstrained content

String values use raw, unrestricted `argmax` over the full vocabulary. Only termination is monitored (closing-quote pattern detection). This is the one place where content fidelity is not guaranteed by the mechanism — correctness depends entirely on the model's own judgment, informed by the instruction prompt and tool descriptions.

### The instruction prompt

Before any generation begins, a hand-constructed prompt is assembled: task instructions + the full tool catalog (name, description, parameter types for every candidate function) + the user's actual request. This entire block is tokenized and becomes the starting `self.ids`. The catalog text is what allows self-attention to surface relationships between prompt wording ("sum") and function descriptions ("Add two numbers together and return their sum") — and thus what influences the model's preference among the legal candidate tokens during name resolution.

## Design Decisions

**Why forced tokens are still added to `self.ids`**: the model is stateless between calls — every `get_logits_from_input_ids` call sees only what is passed in. Forced tokens must appear in `self.ids` or the model's subsequent decisions are made with incomplete context.

**Why a fallback tool is added**: `_resolve_tool_name` can only ever pick among the tools it was given — it has no built-in "none of these" option. The fallback tool (`"No function available for that prompt"`) gives the model a legitimate exit for prompts that genuinely don't match any real function.

**Why Pydantic validation is used only as a gate**: `model_validate` is called on each tool dict during load to catch bad schemas early, but the validated Pydantic objects are discarded — the rest of the program operates on the plain Python dicts from `json.load`. This keeps the generation code simple (plain dict indexing) while still enforcing schema correctness at the boundary.

**Why `encode`/`decode` are used via `cache.py`**: the tokenizer is deterministic — the same string always produces the same token IDs. `cache.py` memoizes these results so that repeated calls (tool names, punctuation tokens, digit characters) never re-tokenize the same string twice during a batch run.

**Why two different masking styles exist**: fancy-indexing (for name resolution) is more efficient when the legal set is already a small, pre-computed set of integers. `-inf` masking (for numbers) is more natural when the allowed set is built by string-to-ID lookups and you want to write directly into a full-sized array.

## Performance Analysis

Tested against the moulinette evaluation tool:

| Set | Score |
|---|---|
| Public | 11/11 (100%) |
| Private | 10/11 (90.9%) |

**Known failure (private set, test 8)**: `"Read the file at /home/user/data.json with utf-8 encoding"` — the leading `/` in the file path gets dropped, producing `home/user/data.json` instead of `/home/user/data.json`. Root cause: `_emit_string` uses unrestricted `argmax` for content generation. Nothing in the masking forces the model to reproduce the literal substring from the prompt, so the model's free generation preference happened not to produce the leading slash. The constrained decoding mechanism guarantees *structural validity* (always valid JSON), not *content fidelity* (literal reproduction of specific substrings).

**Speed**: all 11 prompts processed in under 5 minutes on standard hardware (CPU-only macOS). Each prompt requires many sequential forward passes (one per model-driven token decision). No KV cache is used at the code level — every `get_logits_from_input_ids` call is a full recomputation over the entire accumulated sequence.

**Reliability**: output is fully deterministic — same input, same model weights, same output every run. No sampling or temperature is used; all decisions are `argmax`.

## Challenges Faced

**Leading `/` dropped in file-path strings**: the `_emit_string` function generates string content via unrestricted argmax with no mechanism to force literal text reproduction. The model's free generation failed to include the leading slash for one of the private-set prompts. Consciously left unfixed — fixing it properly would require extracting the intended substring from the prompt and masking each content step to force exact reproduction, which is a significant change. Documented as a known limitation rather than papered over.

**`emit_comma`/`emit_quote` cross-prompt state leak**: these two flags on `ConstrainedGenerator` are reset within a prompt's parameter generation but not between prompts in the same batch run. If a string value's terminator token happens to be a fused compound token like `","` (closing quote + comma in one BPE unit), these flags get set to `0` — and if they're not correctly reset before the next prompt, the next prompt's parameter JSON could be malformed. This is a real latent bug, observable in theory, but not triggered in any test run completed so far.

**`llm_sdk` import crashes at import time**: `cache.py`'s module-level `try/except` only catches `KeyboardInterrupt`, not general exceptions. A model-load failure (missing weights, torch error, network failure on first download) would raise uncaught at import time, bypassing all of `__main__.py`'s own error handling entirely. The fix would be catching `Exception` broadly at the module level, but this was not changed to avoid masking real errors during development.

**`uv`/mypy environment resolution**: `make lint` initially ran mypy against a different Python environment than the project's `.venv`, causing false "cannot find module pydantic" errors. Fixed by adding `flake8` and `mypy` as explicit dev dependencies in `pyproject.toml` and running all lint commands via `uv run` to guarantee the correct interpreter.

## Testing Strategy

**Moulinette grading** (primary): both public and private sets run through the official moulinette grader, which exercises real function calls with computed expected outputs — not just JSON structure but actual correctness of function selection and argument values.

**Schema validation layer** (structural, no model needed): `parser.py`'s `ToolSchema` and `PromptEntry` validators can be exercised against both valid and intentionally malformed inputs independently of the model, enabling fast iteration on input handling without triggering model load.

**Manual edge case verification**: missing input files, malformed JSON (trailing commas, top-level objects instead of arrays), empty prompt lists, prompts with no matching function — all confirmed to produce clean error messages and non-zero exit codes without raw tracebacks.

**Moulinette private set edge cases confirmed passing**: embedded single quotes (`'SELECT * FROM users'`), double quotes (`Say "hello" to {name}`), backslash paths (`C:\Users\john\config.ini`), curly braces inside string values (`Hello {user}'s profile!`), multi-parameter calls with decimal numbers (`1234567.89`, `0.0375`).

## Example Usage

**Default paths** (reads from `data/input/`, writes to `data/output/`):

```bash
uv run python -m src
```

**Custom paths**:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

**Expected output format** (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

**Validate output JSON**:

```bash
cat data/output/function_calling_results.json | python3 -m json.tool
```

## Resources

### Topic references

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper; self-attention mechanism underlying the model.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — architecture and training details for the Qwen3 model family used in this project.
- [Outlines: Efficient Guided Generation (Willard & Louf, 2023)](https://arxiv.org/abs/2307.09702) — constrained decoding via finite-state machines; conceptual foundation for the token-restriction approach implemented here.
- [Byte-Pair Encoding (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909) — the tokenization algorithm used by Qwen3's tokenizer.
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/) — validation library used for input schema enforcement.

### AI usage in this project

Claude (Anthropic) was used throughout this project in the following specific ways:

- **Architecture and design**: discussing the constrained decoding approach, the two-phase (forced vs. model-driven) generation structure, and the tradeoffs between `-inf` masking and fancy-indexing for different generation contexts.
- **Debugging**: diagnosing the leading-`/` drop failure, the `emit_comma`/`emit_quote` state-leak issue, and the mypy/venv environment resolution problem.
- **Theory explanation**: building understanding of tokenization (BPE), forward passes, self-attention, logits vs. probabilities, and autoregressive generation — all used to explain and justify design decisions in this README.
- **Code review**: reviewing `parser.py`, `cache.py`, `generator.py`, and `__main__.py` for correctness, identifying known limitations, and discussing what is and isn't guaranteed by the constrained decoding approach.

All AI-generated content was reviewed, verified against actual behavior, and understood before being included. The code itself was written, tested against the moulinette, and debugged by the author.
