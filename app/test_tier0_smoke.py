#!/usr/bin/env python3
"""Small Tier 0 smoke test using Flask test client."""

import os

# Keep this small/fast for smoke testing
os.environ["TIER0_QUESTION_PER_TYPE"] = "1"
os.environ["TIER0_QUESTION_TARGET_COUNT"] = "2"
os.environ["TIER0_QUESTION_MIN_COUNT"] = "1"
os.environ["TIER0_EXPLORATION_BUDGET"] = "2"

from main import app  # noqa: E402


def main() -> None:
    client = app.test_client()

    payload = {
        "strategy": "balanced",
        "total_budget": 1,
    }

    resp = client.post("/api/rag/explore_corpus", json=payload)
    print("explore status", resp.status_code)
    print(resp.data[:1000])

    report = client.get("/api/rag/exploration_report")
    print("report status", report.status_code)
    print(report.data[:600])


if __name__ == "__main__":
    main()
