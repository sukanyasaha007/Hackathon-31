"""
Evaluation benchmark for the Tariff Classification Agent.

Strategy: Use real CBP CROSS rulings as ground truth.
Each ruling contains a product description AND the official HS/HTS classification.
We extract the description, feed it to the agent, and compare against the known code.

Metrics:
- Chapter accuracy (2-digit): Did the agent get the right chapter? (e.g., 85xx)
- Heading accuracy (4-digit): Did it get the right heading? (e.g., 8518)
- Subheading accuracy (6-digit): Did it get the right HS code? (e.g., 8518.22)
- Full HTS accuracy (8-10 digit): Did it get the exact US code? (e.g., 8518.22.0000)
- Reasoning quality (LLM-as-judge): Does the reasoning cite relevant GRI rules and chapter notes?

Benchmark sets:
- EASY: Products with clear, unambiguous classification (e.g., raw cotton, live cattle)
- MEDIUM: Products requiring chapter note analysis (e.g., Bluetooth speaker, brake rotor)
- HARD: Ambiguous products with competing headings (e.g., smart home device, multifunction printer)
"""

# Hand-curated test cases from public CROSS rulings
# Format: (description, correct_hs6, correct_hts10, ruling_id, difficulty)
BENCHMARK_CASES = [
    # --- EASY (clear, single heading) ---
    {
        "id": "easy_01",
        "description": "Men's woven cotton shirt, long sleeve, button front, 100% cotton fabric",
        "correct_hs6": "6205.20",
        "correct_hts10": "6205.20.2065",
        "difficulty": "easy",
        "notes": "Straightforward textile classification by material and garment type",
    },
    {
        "id": "easy_02",
        "description": "Frozen boneless beef cuts, imported for retail sale",
        "correct_hs6": "0202.30",
        "correct_hts10": "0202.30.8000",
        "difficulty": "easy",
        "notes": "Clear animal product classification",
    },
    {
        "id": "easy_03",
        "description": "Natural rubber latex gloves, disposable, for medical examination use",
        "correct_hs6": "4015.19",
        "correct_hts10": "4015.19.0510",
        "difficulty": "easy",
        "notes": "Rubber article, clearly medical gloves",
    },

    # --- MEDIUM (requires chapter notes or specific knowledge) ---
    {
        "id": "med_01",
        "description": "Portable wireless Bluetooth speaker with built-in rechargeable lithium-ion battery, ABS plastic housing, rated output 10W, waterproof IPX7",
        "correct_hs6": "8518.22",
        "correct_hts10": "8518.22.0000",
        "difficulty": "medium",
        "notes": "Must identify as loudspeaker in enclosure, not radio receiver (8527) or telephone (8517)",
    },
    {
        "id": "med_02",
        "description": "Cast iron ventilated disc brake rotor, diameter 330mm, for heavy-duty commercial truck, new replacement part",
        "correct_hs6": "8708.30",
        "correct_hts10": "8708.30.5090",
        "difficulty": "medium",
        "notes": "Motor vehicle part, brakes section. Must not confuse with general iron casting (7325)",
    },
    {
        "id": "med_03",
        "description": "Lithium-ion battery pack, 48V 13Ah, for electric bicycle, with battery management system",
        "correct_hs6": "8507.60",
        "correct_hts10": "8507.60.0020",
        "difficulty": "medium",
        "notes": "Lithium-ion accumulator. Not bicycle part (8714) because battery is classifiable on its own",
    },
    {
        "id": "med_04",
        "description": "Stainless steel insulated vacuum water bottle, 500ml capacity, double wall construction",
        "correct_hs6": "9617.00",
        "correct_hts10": "9617.00.1000",
        "difficulty": "medium",
        "notes": "Vacuum flask, not tableware (7323) or drinking vessel (7013)",
    },

    # --- HARD (ambiguous, multiple competing headings) ---
    {
        "id": "hard_01",
        "description": "Smart home hub device with 7-inch LCD touchscreen display, built-in speaker and microphone, WiFi and Zigbee connectivity, runs voice assistant software, controls smart home devices, can make video calls and stream music",
        "correct_hs6": "8471.49",
        "correct_hts10": "8471.49.0000",
        "difficulty": "hard",
        "notes": "Competing headings: 8471 (ADP machine), 8517 (telephone), 8518 (speaker), 8528 (monitor). Must apply GRI 3",
    },
    {
        "id": "hard_02",
        "description": "Multifunction laser printer with scan, copy, fax, and print capabilities, Ethernet and WiFi connectivity, A4 paper, 30 pages per minute",
        "correct_hs6": "8443.31",
        "correct_hts10": "8443.31.0000",
        "difficulty": "hard",
        "notes": "Multifunction device. Must apply GRI for composite goods. Printing is principal function.",
    },
    {
        "id": "hard_03",
        "description": "GPS-enabled smartwatch with heart rate monitor, accelerometer, barometric altimeter, Bluetooth, touchscreen AMOLED display, 5ATM water resistance, runs third-party applications",
        "correct_hs6": "9102.12",
        "correct_hts10": "9102.12.8040",
        "difficulty": "hard",
        "notes": "Competing: 9102 (wrist watch), 8517 (communication device), 9031 (measuring instrument). Must determine principal function.",
    },
]


def evaluate_classification(predicted_code: str, correct_code: str) -> dict:
    """Compare predicted vs correct HS/HTS code at multiple granularity levels."""
    # Normalize: remove dots/spaces, pad to at least 6 digits
    pred = predicted_code.replace(".", "").replace(" ", "").ljust(6, "0")
    correct = correct_code.replace(".", "").replace(" ", "").ljust(6, "0")

    return {
        "chapter_match": pred[:2] == correct[:2],      # 2-digit
        "heading_match": pred[:4] == correct[:4],       # 4-digit
        "subheading_match": pred[:6] == correct[:6],    # 6-digit (international HS)
        "full_match": pred == correct,                   # 8-10 digit (US HTS)
    }


def compute_benchmark_scores(results: list[dict]) -> dict:
    """Aggregate accuracy scores across all benchmark cases."""
    n = len(results)
    if n == 0:
        return {}

    scores = {
        "total_cases": n,
        "chapter_accuracy": sum(r["chapter_match"] for r in results) / n,
        "heading_accuracy": sum(r["heading_match"] for r in results) / n,
        "subheading_accuracy": sum(r["subheading_match"] for r in results) / n,
        "full_accuracy": sum(r["full_match"] for r in results) / n,
    }

    # Per-difficulty breakdown
    for difficulty in ["easy", "medium", "hard"]:
        subset = [r for r in results if r.get("difficulty") == difficulty]
        if subset:
            scores[f"{difficulty}_heading_accuracy"] = sum(r["heading_match"] for r in subset) / len(subset)
            scores[f"{difficulty}_subheading_accuracy"] = sum(r["subheading_match"] for r in subset) / len(subset)

    return scores


def print_benchmark_report(results: list[dict], scores: dict):
    """Print a formatted benchmark report."""
    print("=" * 70)
    print("TARIFF CLASSIFICATION BENCHMARK REPORT")
    print("=" * 70)

    print(f"\nTotal cases: {scores['total_cases']}")
    print(f"Chapter (2-digit) accuracy:    {scores['chapter_accuracy']:.1%}")
    print(f"Heading (4-digit) accuracy:    {scores['heading_accuracy']:.1%}")
    print(f"Subheading (6-digit) accuracy: {scores['subheading_accuracy']:.1%}")
    print(f"Full HTS (10-digit) accuracy:  {scores['full_accuracy']:.1%}")

    print("\nPer-difficulty breakdown (heading accuracy):")
    for difficulty in ["easy", "medium", "hard"]:
        key = f"{difficulty}_heading_accuracy"
        if key in scores:
            print(f"  {difficulty.upper():8s}: {scores[key]:.1%}")

    print("\nDetailed results:")
    print("-" * 70)
    for r in results:
        status = "PASS" if r["heading_match"] else "FAIL"
        print(f"  [{status}] {r['case_id']:10s} | predicted={r.get('predicted_hs6', '?'):10s} | correct={r['correct_hs6']:10s} | {r['difficulty']}")

    print("=" * 70)
