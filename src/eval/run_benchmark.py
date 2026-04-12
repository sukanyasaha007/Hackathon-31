"""
Run benchmark: baseline (LLM only, no RAG) and full agent.
Usage: python3 -m src.eval.run_benchmark
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from openai import OpenAI
from src.eval.benchmark import (
    BENCHMARK_CASES,
    evaluate_classification,
    compute_benchmark_scores,
    print_benchmark_report,
)
from src.config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL


def classify_baseline(client: OpenAI, description: str) -> str:
    """Baseline: LLM-only classification, no RAG context."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a tariff classification expert. Given a product description, "
                    "return the most likely 6-digit HS code. Respond with ONLY the code in "
                    "format XXXX.XX (e.g., 8518.22). No explanation."
                ),
            },
            {"role": "user", "content": description},
        ],
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    # Extract the HS code pattern from response
    import re
    match = re.search(r"\d{4}\.\d{2}", raw)
    return match.group(0) if match else raw


def run_baseline_benchmark():
    """Run all benchmark cases against the LLM baseline (no RAG)."""
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    results = []
    for case in BENCHMARK_CASES:
        print(f"  Classifying: {case['id']} ...", end=" ", flush=True)
        try:
            predicted = classify_baseline(client, case["description"])
            time.sleep(13)  # Rate limit: 5 RPM free tier for gemini-2.5-flash
        except Exception as e:
            print(f"ERROR: {e}")
            predicted = "0000.00"

        eval_result = evaluate_classification(predicted, case["correct_hs6"])
        eval_result["case_id"] = case["id"]
        eval_result["difficulty"] = case["difficulty"]
        eval_result["predicted_hs6"] = predicted
        eval_result["correct_hs6"] = case["correct_hs6"]
        eval_result["description"] = case["description"]

        status = "PASS" if eval_result["subheading_match"] else "FAIL"
        print(f"{predicted} vs {case['correct_hs6']} [{status}]")
        results.append(eval_result)

    scores = compute_benchmark_scores(results)
    print("\n")
    print_benchmark_report(results, scores)
    return results, scores


if __name__ == "__main__":
    print("Running BASELINE benchmark (LLM only, no RAG)...\n")
    results, scores = run_baseline_benchmark()

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "scores": scores}, f, indent=2)
    print(f"\nResults saved to {out_path}")
