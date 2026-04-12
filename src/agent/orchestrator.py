"""
Tariff Classification Agent orchestrator.

Pipeline:
1. RETRIEVE — Hybrid search over HTS + CROSS rulings (local, no LLM)
2. CLASSIFY — Single LLM call with retrieved context + GRI rules
3. ENRICH  — Add duty rates, similar rulings, confidence score

Usage:
    from src.agent.orchestrator import TariffAgent
    agent = TariffAgent()
    result = agent.classify("Bluetooth wireless speaker with LED lights")
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.llm import GeminiClient
from src.rag.search import TariffSearcher


@dataclass
class ClassificationResult:
    """Complete tariff classification output."""
    query: str
    hts_code: str
    hts_heading: str
    hts_subheading: str
    description: str
    confidence: float
    reasoning: str
    gri_rules_applied: list[str] = field(default_factory=list)
    similar_rulings: list[dict] = field(default_factory=list)
    duty_rate: str = ""
    alternative_codes: list[dict] = field(default_factory=list)
    needs_expert_review: bool = False
    expert_review_reason: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "hts_code": self.hts_code,
            "hts_heading": self.hts_heading,
            "hts_subheading": self.hts_subheading,
            "description": self.description,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "gri_rules_applied": self.gri_rules_applied,
            "similar_rulings": self.similar_rulings,
            "duty_rate": self.duty_rate,
            "alternative_codes": self.alternative_codes,
            "needs_expert_review": self.needs_expert_review,
            "expert_review_reason": self.expert_review_reason,
            "elapsed_seconds": self.elapsed_seconds,
        }


CLASSIFY_PROMPT = """You are a U.S. Customs tariff classification expert. Classify the following product using the Harmonized Tariff Schedule (HTS).

PRODUCT DESCRIPTION:
{query}

RELEVANT HTS SCHEDULE ENTRIES:
{hts_context}

RELEVANT CROSS RULINGS (prior CBP classification decisions):
{cross_context}

CLASSIFICATION RULES:
- Apply the General Rules of Interpretation (GRI) in order: GRI 1 (terms of headings + notes), GRI 2 (incomplete/unassembled goods), GRI 3 (multiple possible headings), GRI 4 (most akin), GRI 5 (packaging), GRI 6 (subheading level)
- Chapter notes and section notes take precedence
- Consider the essential character of the article
- Reference specific CROSS rulings when they support your classification

Return ONLY this JSON (no markdown, no explanation outside JSON):
{{
  "hts_code": "XXXX.XX.XXXX",
  "hts_heading": "XXXX",
  "hts_subheading": "XXXX.XX",
  "description": "official HTS description text",
  "confidence": 0.85,
  "reasoning": "Step 1: ... Step 2: ... Step 3: ...",
  "gri_rules_applied": ["GRI 1", "GRI 6"],
  "duty_rate": "X.X%",
  "alternative_codes": [
    {{"code": "YYYY.YY", "reason": "could also be classified here because..."}}
  ],
  "needs_expert_review": false,
  "expert_review_reason": ""
}}

IMPORTANT:
- Set confidence below 0.7 and needs_expert_review=true if the classification is ambiguous
- If multiple headings could apply, explain why you chose one over the others
- Reference specific ruling numbers (e.g., NY N352071) when relevant
- The hts_code should be as specific as possible (8-10 digits preferred)"""


class TariffAgent:
    """Tariff classification agent using RAG + single LLM call."""

    def __init__(self):
        self.searcher = TariffSearcher()
        self.llm = GeminiClient()

    def classify(self, query: str) -> ClassificationResult:
        """Run the full classification pipeline."""
        t0 = time.time()

        # Stage 1: RETRIEVE — hybrid search (local, no LLM)
        hts_results, cross_results = self._retrieve([query])

        # Stage 2: CLASSIFY — single LLM call
        result = self._classify(query, hts_results, cross_results)

        # Stage 3: ENRICH — add similar rulings
        result.similar_rulings = [
            {"ruling": r["source"], "text": r["text"][:200]}
            for r in cross_results[:3]
        ]
        result.elapsed_seconds = time.time() - t0

        return result

    def _retrieve(self, queries: list[str], top_k_per_query: int = 5) -> tuple[list[dict], list[dict]]:
        """Search HTS and CROSS for relevant context."""
        all_hts = []
        all_cross = []
        seen_texts = set()

        for q in queries[:3]:
            hts = self.searcher.search(
                q, top_k=top_k_per_query, rerank=True,
                chunk_types=["hts_heading", "hts_note"]
            )
            for r in hts:
                key = r["text"][:200]
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_hts.append(r)

            cross = self.searcher.search(
                q, top_k=top_k_per_query, rerank=True,
                chunk_types=["cross_ruling"]
            )
            for r in cross:
                key = r["text"][:200]
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_cross.append(r)

        all_hts.sort(key=lambda x: x["score"], reverse=True)
        all_cross.sort(key=lambda x: x["score"], reverse=True)

        # Limit context size for smaller LLM models
        return all_hts[:4], all_cross[:3]

    def _classify(self, query: str, hts_results: list[dict], cross_results: list[dict]) -> ClassificationResult:
        """Single LLM call: classify with retrieved context."""
        # Truncate each chunk to limit total prompt size
        hts_context = "\n\n---\n\n".join(r["text"][:500] for r in hts_results) or "No relevant HTS entries found."
        cross_context = "\n\n---\n\n".join(r["text"][:400] for r in cross_results) or "No relevant CROSS rulings found."

        prompt = CLASSIFY_PROMPT.format(
            query=query,
            hts_context=hts_context,
            cross_context=cross_context,
        )

        try:
            content = self.llm.generate(prompt, temperature=0.1, max_tokens=1000)

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                hts_code = data.get("hts_code", "")
                hts_heading = data.get("hts_heading", "")
                hts_subheading = data.get("hts_subheading", "")

                # Fix malformed codes: derive heading/subheading from full code
                if hts_code and "." in hts_code:
                    parts = hts_code.split(".")
                    if not hts_heading or len(hts_heading) != 4:
                        hts_heading = parts[0][:4]
                    if not hts_subheading or not hts_subheading.startswith(hts_heading):
                        hts_subheading = f"{hts_heading}.{parts[1][:2]}" if len(parts) > 1 else hts_heading

                return ClassificationResult(
                    query=query,
                    hts_code=hts_code,
                    hts_heading=hts_heading,
                    hts_subheading=hts_subheading,
                    description=data.get("description", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    gri_rules_applied=data.get("gri_rules_applied", []),
                    duty_rate=data.get("duty_rate", ""),
                    alternative_codes=data.get("alternative_codes", []),
                    needs_expert_review=data.get("needs_expert_review", False),
                    expert_review_reason=data.get("expert_review_reason", ""),
                )
        except Exception as e:
            print(f"  Classification error: {e}")

        return ClassificationResult(
            query=query,
            hts_code="",
            hts_heading="",
            hts_subheading="",
            description="Classification failed",
            confidence=0.0,
            reasoning="Error during classification",
            needs_expert_review=True,
            expert_review_reason="Classification pipeline encountered an error",
        )
