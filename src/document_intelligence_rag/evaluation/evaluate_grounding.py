from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from document_intelligence_rag.evaluation.grounding import (
    DEFAULT_SUPPORT_THRESHOLD,
    evaluate_answer_grounding,
    summarize_grounding_results,
)

logger = logging.getLogger(__name__)


def load_answer_records(path: str | Path) -> list[dict[str, Any]]:
    answers_path = Path(path)
    payload = json.loads(answers_path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("answers"), list):
        records = payload["answers"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ValueError("Answer input must be an answer object, a list, or an object with 'answers'.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every answer record must be an object.")
    return records


def evaluate_grounding_file(
    *,
    answers_path: str | Path,
    output_path: str | Path,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    answer_records = load_answer_records(answers_path)
    results = [
        evaluate_answer_grounding(record, support_threshold=support_threshold)
        for record in answer_records
    ]
    report = {
        "answers_path": str(answers_path),
        "support_threshold": support_threshold,
        "context_note": (
            "Uses cited_sources[].text when available; otherwise falls back to "
            "source_previews[].preview."
        ),
        "metrics": summarize_grounding_results(results),
        "answers": results,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote grounding evaluation report to %s", output)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate deterministic grounding for answer JSON.")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-threshold", type=float, default=DEFAULT_SUPPORT_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = evaluate_grounding_file(
        answers_path=args.answers,
        output_path=args.output,
        support_threshold=args.support_threshold,
    )
    metrics = report["metrics"]
    print(f"answer_count: {metrics['answer_count']}")
    print(f"evaluated_answer_count: {metrics['evaluated_answer_count']}")
    print(f"sentence_support_rate: {metrics['sentence_support_rate']:.4f}")
    print(f"citation_coverage: {metrics['citation_coverage']:.4f}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
