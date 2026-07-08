"""AI precision filter: one batched call per pull cycle.

Heuristics handle the clear-cut mentions for free; only the ambiguous ones
are sent here, in a single batched request, for a final relevance verdict.
"""

from __future__ import annotations

import json

AI_MODEL = "claude-opus-4-8"

MAX_BATCH = 150          # posts per call; more than enough for one pull cycle
SNIPPET = 300            # chars of each post the judge sees

SYSTEM = """You judge whether social media posts are about a specific product
or company. You receive the brand keyword, a description of the product, and a
numbered list of posts.

A post is relevant if it discusses the product, the company, their apps or
services, in any way: praise, complaints, questions, comparisons, news.

A post is NOT relevant if the keyword is used as an ordinary word (a notion,
horsepower, hit points), refers to something else with the same name, or the
post has nothing to do with the product.

Return only the ids of the relevant posts."""


def verify_relevance(texts: list[str], brand: str, product_hint: str,
                     api_key: str) -> list[bool]:
    """Classify a batch of borderline posts in one API call.

    Returns one boolean per input text. Raises on API failure so the caller
    can fall back to heuristics.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    numbered = "\n".join(
        f"{i}. {t[:SNIPPET].replace(chr(10), ' ')}" for i, t in enumerate(texts))
    prompt = (f"BRAND KEYWORD: {brand}\n"
              f"PRODUCT: {product_hint or brand}\n\n"
              f"POSTS:\n{numbered}\n\n"
              "Which posts are about this product or company?")

    response = client.messages.create(
        model=AI_MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "relevant_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        }
                    },
                    "required": ["relevant_ids"],
                    "additionalProperties": False,
                },
            }
        },
    )

    text = next(b.text for b in response.content if b.type == "text")
    ids = set(json.loads(text)["relevant_ids"])
    return [i in ids for i in range(len(texts))]
