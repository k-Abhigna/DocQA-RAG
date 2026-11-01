"""Evaluate retrieval and answers against a golden set, and gate on quality.

Metrics:
  hit@k            - expected source appears in the top-k retrieved passages
  MRR              - mean reciprocal rank of the expected source
  answer_accuracy  - all expected keywords appear in the answer
                     (for unanswerable questions: the answer says it couldn't find it)

Usage:  python -m evals.run_eval [--k 4] [--min-hit 0.9] [--min-acc 0.8]
Exits non-zero if a threshold is missed, so it can run as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag import RAGPipeline

GOLDEN = Path(__file__).with_name("golden.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-hit", type=float, default=0.9)
    parser.add_argument("--min-acc", type=float, default=0.8)
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.rebuild()  # evaluate against the current documents, not a stale index
    examples = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]

    hits, rr, correct, answerable, failures = 0, 0.0, 0, 0, []
    provider = None
    for ex in examples:
        result = pipeline.ask(ex["question"], k=args.k)
        provider = result["provider"] if result["provider"] != "none" else provider
        sources = [s["source"] for s in result["sources"]]

        if ex["expected_source"]:
            answerable += 1
            if ex["expected_source"] in sources:
                hits += 1
                rr += 1 / (sources.index(ex["expected_source"]) + 1)

        ok = all(kw.lower() in result["answer"].lower() for kw in ex["expected_keywords"])
        correct += ok
        if not ok:
            failures.append((ex["id"], ex["question"], result["answer"][:160]))

    hit_rate, mrr, acc = hits / answerable, rr / answerable, correct / len(examples)
    print(f"Provider: {provider}   Examples: {len(examples)}   k={args.k}")
    print(f"hit@{args.k}:          {hit_rate:.2%}")
    print(f"MRR:             {mrr:.3f}")
    print(f"answer_accuracy: {acc:.2%}")
    for fid, q, a in failures:
        print(f"  FAIL {fid}: {q}\n        -> {a}")

    passed = hit_rate >= args.min_hit and acc >= args.min_acc
    print("GATE:", "PASS" if passed else f"FAIL (need hit>={args.min_hit}, acc>={args.min_acc})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
