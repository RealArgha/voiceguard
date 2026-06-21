"""
Plain-English fraud explanation via Claude.

If ANTHROPIC_API_KEY is not set, we fall back to a template-based
explanation so the demo still works offline.
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


def explain(risk: RiskResult, transcript_snippet: str = "") -> str:
    """Returns a 2-3 sentence explanation suitable for an agent's screen."""

    client = _get_client()
    if client is None:
        return _fallback_explain(risk, transcript_snippet)

    prompt = f"""You are a fraud analyst assistant for a bank.
Given the signals below from a live phone call, write a 2-3 sentence
plain-English explanation of WHY this call was flagged. Be specific.
Do not invent details. Address the agent who is on the call.

Risk score: {risk.score}/100  (band: {risk.band})
CNN synthetic-voice probability: {risk.cnn_prob}
Suspicious phrases detected: {risk.keyword_hits or 'none'}
Call metadata flags: {risk.metadata or 'none'}
Transcript snippet (may be empty): "{transcript_snippet}"
"""

    msg = client.messages.create(
        model="claude-opus-4-7",          # adjust to your available model
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _fallback_explain(risk: RiskResult, transcript: str) -> str:
    """Used when no API key is configured."""
    parts = [f"Risk {risk.band} ({risk.score}/100)."]
    if risk.cnn_prob > 0.5:
        parts.append(
            f"The spectrogram CNN believes the voice is synthetic "
            f"(P={risk.cnn_prob:.2f}) — likely AI-generated.")
    if risk.keyword_hits:
        parts.append(
            f"Suspicious phrases detected in transcript: "
            f"{', '.join(risk.keyword_hits[:3])}.")
    if risk.metadata:
        flags = [k for k, v in risk.metadata.items() if v]
        if flags:
            parts.append(f"Metadata flags: {', '.join(flags)}.")
    if len(parts) == 1:
        parts.append("No strong individual signals; score driven by "
                     "combined weak indicators.")
    return " ".join(parts)
