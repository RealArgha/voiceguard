"""
Risk score fusion.

Combines CNN probability and optional call metadata into a 0-100 risk score.
Keyword/transcript signals removed — CNN-only detection runs with no delay.
"""

from dataclasses import dataclass


W_CNN      = 0.90
W_METADATA = 0.10

CNN_THRESHOLD = 0.56


@dataclass
class RiskResult:
    score: float          # 0-100
    band:  str            # "LOW" | "MEDIUM" | "HIGH"
    action: str           # "PASS" | "OTP_CHALLENGE" | "BLOCK"
    cnn_prob: float
    metadata: dict


def fuse(cnn_prob: float, metadata: dict | None = None) -> RiskResult:

    metadata = metadata or {}

    cnn_component = cnn_prob * 100.0

    md_component = 0.0
    if metadata.get("unknown_number"):  md_component += 50
    if metadata.get("voip_origin"):     md_component += 30
    if metadata.get("foreign_country"): md_component += 20
    md_component = min(md_component, 100.0)

    score = W_CNN * cnn_component + W_METADATA * md_component

    if   score <= 34: band, action = "LOW",    "PASS"
    elif score <= 70: band, action = "MEDIUM", "OTP_CHALLENGE"
    else:             band, action = "HIGH",   "BLOCK"

    return RiskResult(
        score=round(score, 1),
        band=band,
        action=action,
        cnn_prob=round(cnn_prob, 3),
        metadata=metadata,
    )
