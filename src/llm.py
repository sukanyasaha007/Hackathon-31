"""
LLM client using OpenAI-compatible API (works with Groq, Gemini, OpenAI, etc.)
"""

import os
import time
import json
from openai import OpenAI, RateLimitError

from src.config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL

# Fallback model for when primary model TPD is exhausted
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "llama-3.1-8b-instant")


class LLMClient:
    """OpenAI-compatible LLM client with retry logic and model fallback."""

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.model = model or LLM_MODEL
        self.fallback_model = FALLBACK_MODEL
        self._client = OpenAI(
            api_key=api_key or LLM_API_KEY,
            base_url=base_url or LLM_BASE_URL,
        )

    def generate(self, prompt: str, temperature: float = 0.0,
                 max_tokens: int = 2000, max_retries: int = 3) -> str:
        """Generate text with automatic retry on rate limits and model fallback."""
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                error_msg = str(e)
                # If daily token limit hit, try fallback model immediately
                if "tokens per day" in error_msg and self.fallback_model != self.model:
                    print(f"  Daily limit hit on {self.model}, falling back to {self.fallback_model}...")
                    try:
                        resp = self._client.chat.completions.create(
                            model=self.fallback_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=min(max_tokens, 1000),
                        )
                        return resp.choices[0].message.content or ""
                    except Exception as fallback_err:
                        print(f"  Fallback also failed: {fallback_err}")
                if attempt < max_retries - 1:
                    delay = 15 * (attempt + 1)
                    print(f"  Rate limited. Waiting {delay}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Error: {e}. Retrying in 10s...")
                    time.sleep(10)
                else:
                    raise

        raise RuntimeError(f"Failed after {max_retries} retries")


# Backward compat alias
GeminiClient = LLMClient
