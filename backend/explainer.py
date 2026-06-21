"""
Plain-English fraud explanation.

If ANTHROPIC_API_KEY is not set, falls back to template-based explanation.
"""

import os
from .fusion import RiskResult

_client = None


def _get_client():
    global _client
    if _client is None and os.getenv("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic
        _client = Anthropic()
    return _client


def explain(risk: RiskResult) -> str:
    client = _get_client()
    if client is None:
        return _fallback_explain(risk)

    prompt = f"""You are a fraud analyst assistant for a bank.
Given the signals below from a live phone call, write a 2-3 sentence
plain-English explanation of WHY this call was flagged. Be specific.
Do not invent details. Address the agent who is on the call.

Risk score: {risk.score}/100  (band: {risk.band})
CNN synthetic-voice probability: {risk.cnn_prob}
Call metadata flags: {risk.metadata or 'none'}
"""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _fallback_explain(risk: RiskResult) -> str:
    parts = [f"Risk {risk.band} ({risk.score}/100)."]
    if risk.cnn_prob > 0.5:
        parts.append(
            f"The voice analysis model believes this is synthetic audio "
            f"(P={risk.cnn_prob:.2f}).")
    if risk.metadata:
        flags = [k for k, v in risk.metadata.items() if v]
        if flags:
            parts.append(f"Metadata flags: {', '.join(flags)}.")
    if len(parts) == 1:
        parts.append("No strong signals detected; score reflects low synthetic probability.")
    return " ".join(parts)
