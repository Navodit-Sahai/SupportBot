"""
CLI entry point.

Usage:
    python main.py                                          # interactive REPL
    python main.py -q "your question"                       # single-shot
    python main.py --sample                                 # run bundled sample questions
    python main.py --sample --sample-file path/to/file.json # custom question file
    python main.py --sample --output-file path/to/out.json  # custom output path
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from src.graph import build_graph
from src.state import new_state


# Default location for the bundled sample-questions file. Resolved relative
# to main.py so the CLI works regardless of the caller's current directory.
DEFAULT_SAMPLE_FILE = Path(__file__).parent / "data" / "sample_questions.json"

# Default location for the sample-run output. Written ONLY when --sample is
# used, not from the REPL or single-shot -q modes. Overwritten on each run
# so the file always reflects the latest sample execution.
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "data" / "sample_run_output.json"


def load_sample_questions(path: Path) -> List[Tuple[str, str]]:
    """Load (question_id, question_text) pairs from a sample-questions JSON.

    Expected schema:
        {"product": "...", "questions": [{"question_id": "...", "question": "..."}]}

    """
    if not path.exists():
        raise FileNotFoundError(f"Sample questions file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Sample questions file is not valid JSON ({path}): {e}") from e

    raw_questions = data["questions"]

    pairs: List[Tuple[str, str]] = []
    for idx, item in enumerate(raw_questions):
        if isinstance(item, str):
            pairs.append((f"Q-{idx + 1:03d}", item))
            continue

        qid = str(item.get("question_id") or f"Q-{idx + 1:03d}")
        text = str(item["question"]).strip()
        if not text:
            continue  # skip blank questions from sending to the graph
        pairs.append((qid, text))

    return pairs


def run_once(app, query: str) -> dict:
    state = new_state(query)
    final = app.invoke(state)
    return final


def _pretty(result: dict) -> str:
    resp = result.get("final_response", {})
    trace = " -> ".join(result.get("node_trace", []))
    latencies = result.get("latencies_ms", {})
    return (
        "\n=== Response ===\n"
        + json.dumps(resp, indent=2, ensure_ascii=False)
        + f"\n\nTrace: {trace}"
        + f"\nLatencies (ms): {latencies}\n"
    )


def _record(qid: str, question: str, result: dict) -> dict:
    """Extract the persistable slice of a run result for the output file."""
    return {
        "question_id": qid,
        "question": question,
        "response": result.get("final_response", {}),
        "trace": " -> ".join(result.get("node_trace", [])),
        "latencies_ms": result.get("latencies_ms", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", help="Single query to run")
    parser.add_argument("--sample", action="store_true",
                        help="Run the sample questions from the sample-questions file")
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE,
                        help=f"Path to sample-questions JSON (default: {DEFAULT_SAMPLE_FILE})")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE,
                        help=f"Where to write sample-run results (default: {DEFAULT_OUTPUT_FILE})")
    args = parser.parse_args()

    app = build_graph()

    if args.sample:
        try:
            questions = load_sample_questions(args.sample_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading sample questions: {e}", file=sys.stderr)
            return 1

        if not questions:
            print(f"No questions found in {args.sample_file}", file=sys.stderr)
            return 1

        # Collect every run's persistable record so we can dump them all
        # at the end. Only sample-mode writes to disk — REPL and -q do not.
        records: List[dict] = []
        for qid, q in questions:
            print(f"\n### {qid}: {q}")
            result = run_once(app, q)
            print(_pretty(result))
            records.append(_record(qid, q, result))

        # Ensure output dir exists (it usually does; being defensive).
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_file": str(args.sample_file),
            "runs": records,
        }
        args.output_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved {len(records)} run(s) to {args.output_file}")
        return 0

    if args.query:
        print(_pretty(run_once(app, args.query)))
        return 0

    # Interactive REPL
    print("Support agent ready. Type your question. Ctrl-D or 'exit' to quit.")
    try:
        while True:
            q = input("\n> ").strip()
            if not q or q.lower() in {"exit", "quit"}:
                break
            print(_pretty(run_once(app, q)))
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())