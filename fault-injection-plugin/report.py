# SPDX-License-Identifier: AGPL-3.0-or-later
"""Awareness report — joins injections with user reactions on ``request_id``.

This closes the loop: the injection log says what was faked, the feedback log
says how the user reacted, and this report computes the metric the tool exists
for — the *noticed rate* per error type.

Usage:
    python report.py --injections audit/injections.jsonl \\
                     --feedback   audit/feedback.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

# feedback signals that count as "the user noticed the fault"
NOTICED_SIGNALS = {"noticed", "corrected", "reasked", "thumbs_down"}


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def build_report(injections: Iterable[dict], feedback: Iterable[dict]) -> dict:
    # map request_id -> set of signals seen
    signals: Dict[str, set] = defaultdict(set)
    for fb in feedback:
        rid = fb.get("request_id")
        if rid:
            signals[rid].add(fb.get("signal"))

    per_type: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"injected": 0, "noticed": 0, "no_feedback": 0}
    )
    for inj in injections:
        if inj.get("event") != "injected":
            continue
        etype = inj.get("error_type") or "unknown"
        rid = inj.get("request_id")
        bucket = per_type[etype]
        bucket["injected"] += 1
        sigs = signals.get(rid, set())
        if not sigs:
            bucket["no_feedback"] += 1
        elif sigs & NOTICED_SIGNALS:
            bucket["noticed"] += 1

    summary = {}
    for etype, b in per_type.items():
        with_feedback = b["injected"] - b["no_feedback"]
        summary[etype] = {
            **b,
            "noticed_rate": (b["noticed"] / with_feedback) if with_feedback else None,
        }
    totals = {
        "injected": sum(b["injected"] for b in per_type.values()),
        "noticed": sum(b["noticed"] for b in per_type.values()),
        "no_feedback": sum(b["no_feedback"] for b in per_type.values()),
    }
    return {"by_error_type": summary, "totals": totals}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Awareness report")
    ap.add_argument("--injections", default="audit/injections.jsonl")
    ap.add_argument("--feedback", default="audit/feedback.jsonl")
    args = ap.parse_args(argv)
    report = build_report(
        _read_jsonl(args.injections), _read_jsonl(args.feedback)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
