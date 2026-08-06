# OrbitDesk Support Agent

A support-agent workflow that classifies a customer question, retrieves evidence from a local knowledge base, generates a cited answer, and verifies the answer before returning it. Every model runs locally — no OpenAI, Anthropic, or Gemini calls anywhere in the pipeline.

Built for the AI Engineer internship assignment.

---

## What it does

You type a support question. The system:

1. **Triages** the question into one of four buckets: `answerable`, `clarification`, `escalation`, `out_of_scope`.
2. **Retrieves** relevant passages from the KB (only if answerable).
3. **Generates** a JSON answer citing the retrieved passages.
4. **Verifies** the answer against a schema and against citation validity.
5. Returns a structured response, or retries once, or hands off to a human.

The graph is built with LangGraph. Every node is a separate file under `src/nodes/`.

See `graph.png` (in the repo root) for the full workflow diagram.

---

## Setup — from scratch, on a fresh machine

Assumes Python 3.10+ and pip.

### 1. Clone the repo

```bash
git clone https://github.com/Navodit-Sahai/AI-engineer-Assignment-Tantrabodh-AI- TantrabodhAssignment
cd TantrabodhAssignment
```

(The explicit target folder keeps the working directory name consistent with the paths inside the code.)

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `langgraph`, `transformers`, `sentence-transformers`, `torch`, `rank-bm25`, `pydantic`, and `numpy`. Total install time: 2-5 minutes depending on your network.

### 4. First run downloads the models

```bash
python main.py --sample
```

On the very first invocation, three models are downloaded from Hugging Face:

| Model | Size | Purpose |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | Dense embeddings |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB | Reranker |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | Generation + triage LLM |

Total download: ~3.2 GB. After download they're cached under `~/.cache/huggingface/` and the app can run **offline**.

The first question takes an extra ~30 seconds to load the models into memory. Subsequent questions in the same run reuse the loaded models.

---

## How to run

Three modes, all from the project root:

**Run all sample questions from `sample_questions.json`:**
```bash
python main.py --sample
```
This also writes every response to `sample_run_output.json` at the repo root — one machine-readable file with all runs for the reviewer to inspect.

**Ask one question:**
```bash
python main.py -q "Can a read-only user create API credentials?"
```

**Interactive chatbot:**
```bash
python main.py
```
Type `exit` or press `Ctrl+C` to quit.

**Custom question file:**
```bash
python main.py --sample --sample-file path/to/questions.json
```

---

## Regenerating the graph diagram

If you change the graph structure and want a fresh image:

```bash
python draw_graph.py
```

Writes `graph.png` at the repo root. Uses LangGraph's built-in Mermaid renderer (one-time internet call to mermaid.ink). If offline, the script falls back to writing a `.mmd` source file that you can paste into https://mermaid.live.

---

## Running the tests

```bash
pytest tests/ -v
```

`tests/test_routing.py` stubs every node so it verifies the **graph shape** — which nodes ran, which branch was taken, that retry stops after one attempt — without depending on the LLM's wording. Runs in under one second. This is the "automated test verifying graph routing without depending on the exact wording" the assignment requires.

Five tests, all deterministic:
- Answerable path visits the full pipeline
- Escalation skips retrieval
- Out-of-scope skips retrieval
- Clarification terminates early
- Verification failure retries once, then produces safe failure — never infinite loops

---

## Configuration — one place, `config.py`

Every knob is in `config.py`. No magic numbers scattered in code.

Common changes:

```python
# Switch to a smaller/faster LLM
llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

# Adjust hybrid weight (0.0 = pure BM25, 1.0 = pure vector)
hybrid_alpha: float = 0.6

# Retry budget
max_retries: int = 1

# Enable NLI-based entailment verification (slow on CPU)
verification_use_nli: bool = False
```

**One thing to check on your machine:** `llm_dtype` in `ModelConfig`. On a modern CPU without AVX-512-BF16, `"bfloat16"` uses a slow emulated kernel — set it to `"float32"` or `"float16"` for practical speed. `"float16"` was used for the reference run.

---

## Output schema

Every response conforms to this shape:

```json
{
  "classification": "answerable",
  "answer": "Only Administrators and Developers can create API credentials [S1].",
  "sources": [
    { "source_id": "02_roles_and_permissions.md", "passage": "02_roles_and_permissions.md#4" }
  ],
  "confidence": 0.88,
  "requires_human": false,
  "reason": "roles-and-permissions source states this directly"
}
```

Field-by-field:
- `classification` — one of `answerable`, `requires_clarification`, `requires_escalation`, `out_of_scope`, `safe_failure`.
- `answer` — human-readable text with inline `[S?]` citation tags.
- `sources` — real filenames and chunk IDs corresponding to each `[S?]` label.
- `confidence` — `0.7 * sigmoid(top_rerank_score) + 0.3 * sigmoid(top_hybrid_score)` for the answerable path. Deterministic values for other paths (see `approach.md`).
- `requires_human` — true only for escalation and safe-failure paths.
- `reason` — short explanation of why this classification/answer was chosen.

A reference copy of the schema is committed as `data/output_schema.json`.

---

## Logs and traceability

Every node run writes a structured line to `logs/agent.jsonl`. Tail it during a run to see the graph execute:

```bash
tail -f logs/agent.jsonl
```

You'll see events like:
```json
{"ts": ..., "node": "triage", "event": "classified", "label": "answerable", "reasoning": "Label: A\nExplanation: The message mentions..."}
{"ts": ..., "node": "retrieval", "event": "reranked", "kept": 5, "top_score": 0.999, "top_source": "CASE-1041"}
{"ts": ..., "node": "generation", "event": "produced", "parsed": true, "citations_repaired": true}
{"ts": ..., "node": "verification", "event": "checked", "passed": true, "errors": []}
{"ts": ..., "node": "finalize_answer", "event": "done", "confidence": 0.88, "rerank_component": 0.999, "hybrid_component": 0.72}
```

Interaction summaries are appended to `data/interaction_metadata.jsonl` — question, classification, sources cited, confidence. AI-generated answers are **never** added back to the retrievable corpus, to prevent contamination.

---

## Repo layout

```
TantrabodhAssignment/
├── .gitignore
├── README.md                       # this file
├── approach.md                     # architectural decisions and rationale
├── config.py                       # all thresholds, model IDs, paths
├── main.py                         # CLI entry point
├── draw_graph.py                   # regenerate graph.png from the LangGraph structure
├── graph.png                       # workflow diagram (also referenced in the video)
├── requirements.txt
├── sample_questions.json           # 6 test questions covering all four routing patterns
├── sample_run_output.json          # last --sample run's responses, written by main.py
├── data/
│   ├── kb/*.md                     # 10 knowledge-base documents
│   ├── resolved_cases.json         # 8 previously solved support cases
│   ├── output_schema.json          # reference copy of the response schema
│   └── interaction_metadata.jsonl  # written at runtime, gitignored
├── logs/agent.jsonl                # structured node-level logs
├── src/
│   ├── state.py                    # AgentState TypedDict (shared state)
│   ├── schemas.py                  # pydantic response schema
│   ├── graph.py                    # LangGraph wiring + conditional routing
│   ├── models/                     # llm.py, embedder.py, reranker.py, device.py
│   ├── retrievers/                 # bm25.py, vector.py, hybrid.py, corpus.py
│   ├── nodes/                      # triage / retrieval / context_builder /
│   │                               # generation / verification / finalize
│   └── utils/                      # logger, chunker, metadata_store

```

---

## Models used (with revisions)

| Role | Model | Revision | License |
|---|---|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | `main` | Apache 2.0 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `main` | Apache 2.0 |
| Generation + Triage | `Qwen/Qwen2.5-1.5B-Instruct` | `main` | Apache 2.0 |
| NLI (optional, off) | `cross-encoder/nli-deberta-v3-base` | `main` | MIT |

Revisions are set to `main` in `config.py`. For strict reproducibility, replace each `*_revision` with the commit SHA visible on the model's Hugging Face page under *Files and versions*.

---

## Hardware requirements

**Minimum (CPU-only, tested):**
- Python 3.10+
- 8 GB RAM (models fit; 4 GB is tight)
- 5 GB free disk (models + cache)
- Any x86_64 or Apple Silicon CPU

**Reference run environment** (the machine `sample_run_output.json` was generated on):
- OS: Windows 11
- CPU: x86_64, CPU-only inference (no GPU / no accelerator)
- RAM: 16 GB
- Python: 3.11
- LLM dtype: `float16`
- Observed latency: ~60-90 s triage, ~100-200 s generation per answerable question

**Performance notes:**
- On CPU with `float16`: end-to-end ~2-3 minutes per answerable question (matches the reference run).
- On CPU with `float32`: comparable or slightly faster on machines without AVX-512-BF16.
- On CUDA GPU: ~3-5 s per question (auto-detected when available).
- On Apple Silicon with `device="mps"`: ~10 s per question.

The pipeline runs entirely offline after the initial model download.

---

## AI Assistant Disclosure

Portions of this codebase were drafted with the assistance of Claude (Anthropic's AI assistant). Every architectural decision, prompt design, and threshold value was reviewed, tested, and where necessary reworked by the author. Specifically:

- The retry-branch fix in `generation.py` (assistant-turn + edit-request pattern) was co-designed after debugging a real failure mode where the model was reviewing an imagined prior answer.
- The deterministic citation-repair path in `generation.py` was added after evidence-based prompt engineering alone failed to override the small model's prose/metadata separation prior.
- The chain-of-thought triage prompt in `triage.py` replaced an earlier example-based prompt that was found to be pattern-matching on surface form rather than reasoning.
- The composite confidence formula in `finalize.py` was designed to replace an uncalibrated LLM self-reported number with a signal-based composite.

See `APPROACH.md` for the full record of decisions and their rationale.