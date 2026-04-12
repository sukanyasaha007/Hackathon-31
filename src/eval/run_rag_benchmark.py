"""
Run the full benchmark with the RAG-powered agent.

Usage: python3 -m src.eval.run_rag_benchmark
"""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.orchestrator import TariffAgent
from src.eval.benchmark import (
    BENCHMARK_CASES,
    evaluate_classification,
    compute_benchmark_scores,
    print_benchmark_report,
)


def run_rag_benchmark():
    """Run all benchmark cases through the RAG agent."""
    agent = TariffAgent()

    results = []
    for case in BENCHMARK_CASES:
        print(f"  [{case['id']}] {case['description'][:60]}...", end=" ", flush=True)
        try:
            result = agent.classify(case["description"])
            predicted = result.hts_subheading  # Use 6-digit for comparison
            eval_result = evaluate_classification(predicted, case["correct_hs6"])

            results.append({
                "id": case["id"],
                "description": case["description"][:80],
                "difficulty": case["difficulty"],
                "correct": case["correct_hs6"],
                "predicted_heading": result.hts_heading,
                "predicted_subheading": result.hts_subheading,
                "predicted_code": result.hts_code,
                "confidence": result.confidence,
                "needs_expert": result.needs_expert_review,
                "elapsed": result.elapsed_seconds,
                **eval_result,
            })

            status = "PASS" if eval_result["heading_match"] else "FAIL"
            print(f"[{status}] {result.hts_subheading} (correct: {case['correct_hs6']}) {result.elapsed_seconds:.1f}s")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": case["id"],
                "description": case["description"][:80],
                "difficulty": case["difficulty"],
                "correct": case["correct_hs6"],
                "predicted_heading": "",
                "predicted_subheading": "",
                "predicted_code": "",
                "confidence": 0,
                "needs_expert": True,
                "elapsed": 0,
                "chapter_match": False,
                "heading_match": False,
                "subheading_match": False,
                "full_match": False,
            })

        # Rate limit padding for Groq
        time.sleep(3)

    # Compute scores
    scores = compute_benchmark_scores(results)
    print_benchmark_report(results, scores)

    # Save results
    output = {
        "run_type": "rag",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "scores": scores,
    }
    out_path = Path(__file__).parent.parent.parent / "src" / "eval" / "rag_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG BENCHMARK")
    print("=" * 60)
    run_rag_benchmark()
