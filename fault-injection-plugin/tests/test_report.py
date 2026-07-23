# SPDX-License-Identifier: AGPL-3.0-or-later
from report import build_report


def test_noticed_rate_per_type():
    injections = [
        {"event": "injected", "error_type": "factual", "request_id": "a"},
        {"event": "injected", "error_type": "factual", "request_id": "b"},
        {"event": "injected", "error_type": "bad_code", "request_id": "c"},
        {"event": "skipped", "error_type": "factual", "request_id": "d"},  # ignored
    ]
    feedback = [
        {"request_id": "a", "signal": "corrected"},  # noticed
        {"request_id": "b", "signal": "none"},        # feedback, not noticed
        # c has no feedback
    ]
    report = build_report(injections, feedback)
    factual = report["by_error_type"]["factual"]
    assert factual["injected"] == 2
    assert factual["noticed"] == 1
    assert factual["no_feedback"] == 0
    assert factual["noticed_rate"] == 0.5

    bad_code = report["by_error_type"]["bad_code"]
    assert bad_code["injected"] == 1
    assert bad_code["no_feedback"] == 1
    assert bad_code["noticed_rate"] is None  # no feedback -> undefined

    assert report["totals"]["injected"] == 3
