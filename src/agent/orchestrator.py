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
    tariff_warnings: list[str] = field(default_factory=list)
    clarifying_questions: list[str] = field(default_factory=list)
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
            "tariff_warnings": self.tariff_warnings,
            "clarifying_questions": self.clarifying_questions,
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

    def classify(self, query: str, country_of_origin: str = "") -> ClassificationResult:
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

        # Stage 4: GUARDRAILS — post-classification validation
        self._apply_guardrails(result, country_of_origin=country_of_origin)

        # Stage 5: HUMAN-IN-THE-LOOP — generate clarifying questions if needed
        self._generate_clarifications(result, hts_results, cross_results)

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

    def _apply_guardrails(self, result: ClassificationResult, country_of_origin: str = "") -> None:
        """Post-classification validation: flag trade remedies, tariff volatility, and country-of-origin risks."""
        warnings = []
        heading = result.hts_heading
        code = result.hts_code
        duty = result.duty_rate.strip().lower() if result.duty_rate else ""
        country = country_of_origin.strip().lower()

        # --- Section 301 / IEEPA chapters at risk of additional duties ---
        section_301_chapters = {
            "84": "machinery/computers",
            "85": "electrical/electronics",
            "87": "vehicles",
            "90": "optical/medical instruments",
            "73": "iron/steel articles",
            "39": "plastics",
            "40": "rubber",
            "94": "furniture",
        }
        chapter = heading[:2] if heading else ""

        # Countries with known additional duty programs (approximate rates, mid-2026)
        high_tariff_countries = {
            "china": "Section 301 (up to 25%) + IEEPA tariffs apply. Effective total duty on many goods exceeds 100%.",
            "hong kong": "Treated as China for tariff purposes under current policy. Section 301 + IEEPA tariffs apply.",
            "russia": "Column 2 (non-NTR) rates apply. Most imports prohibited or subject to sanctions.",
            "belarus": "Column 2 (non-NTR) rates apply.",
        }
        # Countries with moderate additional tariffs
        moderate_tariff_countries = {
            "vietnam": "Subject to reciprocal tariff of 46% (paused/reduced — verify current status).",
            "india": "Subject to reciprocal tariff of 26% (paused/reduced — verify current status).",
            "thailand": "Subject to reciprocal tariff of 36% (paused/reduced — verify current status).",
            "indonesia": "Subject to reciprocal tariff of 32% (paused/reduced — verify current status).",
            "taiwan": "Subject to reciprocal tariff of 32% (paused/reduced — verify current status).",
            "south korea": "Subject to reciprocal tariff of 25% (paused/reduced — verify current status).",
            "japan": "Subject to reciprocal tariff of 24% (paused/reduced — verify current status).",
        }
        # Free trade agreement countries (generally no additional tariffs)
        fta_countries = {
            "canada", "mexico",  # USMCA
            "australia", "singapore", "chile", "peru", "colombia", "panama",
            "south korea",  # KORUS (but also has reciprocal — complex)
            "israel", "jordan", "bahrain", "oman", "morocco",
        }

        if country:
            # --- Country-specific warnings ---
            if country in high_tariff_countries:
                warnings.append(
                    f"Country of origin: {country_of_origin.strip().title()}. "
                    f"{high_tariff_countries[country]} "
                    f"The displayed duty rate is the Column 1 General (MFN) base rate only."
                )
            elif country in moderate_tariff_countries:
                warnings.append(
                    f"Country of origin: {country_of_origin.strip().title()}. "
                    f"{moderate_tariff_countries[country]} "
                    f"The displayed duty rate is the Column 1 General (MFN) base rate only."
                )
            elif country in fta_countries and chapter in section_301_chapters:
                # FTA countries: preferential rate may apply
                warnings.append(
                    f"Country of origin: {country_of_origin.strip().title()}. "
                    f"A free trade agreement may provide preferential (reduced or zero) duty rates. "
                    f"Verify Rules of Origin requirements are met."
                )
            # Chapter-specific trade remedy warning for non-FTA countries
            if chapter in section_301_chapters and country not in fta_countries:
                if country not in high_tariff_countries and country not in moderate_tariff_countries:
                    warnings.append(
                        f"Products in Chapter {chapter} ({section_301_chapters[chapter]}) may be subject to "
                        f"additional trade remedy duties depending on the specific product and origin."
                    )
        else:
            # --- No country provided: generic warning for high-risk chapters ---
            if chapter in section_301_chapters:
                warnings.append(
                    f"The displayed duty rate is the Column 1 General (MFN) base rate. "
                    f"Products in Chapter {chapter} ({section_301_chapters[chapter]}) are commonly subject to "
                    f"additional duties (Section 301, IEEPA, reciprocal tariffs) depending on country of origin. "
                    f"Specify country of origin for a more accurate assessment."
                )

        # --- Zero/low duty rate note (country-aware) ---
        duty_pct = 0.0
        if duty:
            m = re.search(r'([\d.]+)\s*%', duty)
            if m:
                duty_pct = float(m.group(1))
        if duty_pct < 1.0 and code:
            if country and country in high_tariff_countries:
                warnings.append(
                    f"Base MFN rate is {result.duty_rate or '0%'}, but actual landed duty for "
                    f"{country_of_origin.strip().title()}-origin goods will be substantially higher "
                    f"due to additional tariffs."
                )
            elif not country:
                warnings.append(
                    f"Base duty rate is {result.duty_rate or '0%'}. Additional duties may apply "
                    f"depending on country of origin and current trade policy."
                )

        # --- Volatility disclaimer (always) ---
        if code:
            warnings.append(
                "Tariff rates are subject to change. Verify current effective rates with a licensed "
                "customs broker or the USITC HTS database before making import decisions."
            )

        result.tariff_warnings = warnings

        # Flag for expert review if trade remedy warnings fired
        if len(warnings) > 1 and not result.needs_expert_review:
            result.needs_expert_review = True
            reason = ""
            if country and (country in high_tariff_countries or country in moderate_tariff_countries):
                reason = f"Additional tariffs likely apply for {country_of_origin.strip().title()}-origin goods."
            else:
                reason = "Additional duties (Section 301/IEEPA/reciprocal) may apply based on country of origin."
            result.expert_review_reason = (
                result.expert_review_reason + " " if result.expert_review_reason else ""
            ) + reason

    def _generate_clarifications(self, result: ClassificationResult,
                                  hts_results: list[dict], cross_results: list[dict]) -> None:
        """Generate clarifying questions when the agent lacks information to classify confidently."""
        questions = []

        # Trigger 1: Low confidence — the LLM is uncertain
        if result.confidence < 0.7:
            # Check if there are competing alternatives
            if result.alternative_codes:
                alts = ", ".join(a["code"] for a in result.alternative_codes[:3])
                questions.append(
                    f"Multiple headings could apply ({alts}). "
                    f"What is the primary function or intended use of this product?"
                )
            else:
                questions.append(
                    "The description may be too general. Can you provide more detail — "
                    "material composition, dimensions, intended use, or how it is powered?"
                )

        # Trigger 2: Very short query (< 4 words) — not enough detail to distinguish subheadings
        if len(result.query.split()) < 4:
            questions.append(
                "A more detailed description improves accuracy. Consider adding: "
                "material, size/weight, power source, end use, or brand/model."
            )

        # Trigger 3: Sparse retrieval — not enough HTS context found
        if len(hts_results) < 2:
            questions.append(
                "Limited tariff schedule data was found for this product. "
                "Can you describe it using standard trade terminology or specify the material?"
            )

        # Trigger 4: No CROSS rulings found — no precedent
        if not cross_results:
            questions.append(
                "No prior CBP classification rulings were found for similar products. "
                "Can you provide a more specific product type or industry category?"
            )

        # Trigger 5: Empty or failed classification
        if not result.hts_code:
            questions = [
                "The agent could not determine a classification. Please provide: "
                "the product's material, primary function, and intended use."
            ]

        result.clarifying_questions = questions

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
            content = self.llm.generate(prompt, temperature=0.0, max_tokens=1000)

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

                # Validate duty_rate — LLM sometimes guesses; cross-check against HTS data + live API
                raw_duty = data.get("duty_rate", "")
                duty_rate = self._validate_duty_rate(raw_duty, hts_results, hts_code)

                return ClassificationResult(
                    query=query,
                    hts_code=hts_code,
                    hts_heading=hts_heading,
                    hts_subheading=hts_subheading,
                    description=data.get("description", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    gri_rules_applied=data.get("gri_rules_applied", []),
                    duty_rate=duty_rate,
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

    @staticmethod
    def _validate_duty_rate(raw_duty: str, hts_results: list[dict], hts_code: str = "") -> str:
        """Check if duty rate appears in retrieved HTS context. Falls back to live USITC API."""
        if not raw_duty:
            return ""
        raw_lower = raw_duty.strip().lower()
        # "Free" is a known valid rate
        if raw_lower == "free":
            # Verify it actually appears in the HTS chunks
            hts_text = " ".join(r["text"].lower() for r in hts_results)
            if "free" in hts_text:
                return "Free"
            return "Free (unverified)"
        # Check if the percentage appears in retrieved HTS text
        m = re.search(r'([\d.]+)\s*%', raw_duty)
        if m:
            pct_str = m.group(1)
            hts_text = " ".join(r["text"] for r in hts_results)
            if pct_str in hts_text:
                return raw_duty  # Rate found in source data
            # Fallback: query live USITC API to verify
            if hts_code:
                live_rate = TariffAgent._lookup_usitc_rate(hts_code)
                if live_rate and pct_str in live_rate:
                    return f"{raw_duty} (verified via USITC API)"
                elif live_rate:
                    return f"{live_rate} (USITC API)"
            # Rate not found in retrieved chunks or API — may be hallucinated
            return f"{raw_duty} (unverified)"
        return raw_duty

    @staticmethod
    def _lookup_usitc_rate(hts_code: str) -> str:
        """Query the live USITC HTS API for the general duty rate of a specific code."""
        try:
            import httpx
            resp = httpx.get(
                "https://hts.usitc.gov/reststop/search",
                params={"keyword": hts_code},
                timeout=5,
            )
            resp.raise_for_status()
            results = resp.json()
            # Find exact or closest match
            for r in results:
                if r.get("htsno", "").replace(".", "") == hts_code.replace(".", ""):
                    rate = r.get("general", "")
                    if rate:
                        return rate
            # Try prefix match (heading level)
            prefix = hts_code.split(".")[0] if "." in hts_code else hts_code[:4]
            for r in results:
                if r.get("htsno", "").startswith(prefix) and r.get("general"):
                    return r["general"]
        except Exception:
            pass
        return ""
