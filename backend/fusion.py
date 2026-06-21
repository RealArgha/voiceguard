"""
Risk score fusion.

Combines three signals into a 0-100 risk score:
  - cnn_prob     : P(synthetic) from the spectrogram CNN  [0, 1]
  - keyword_hits : count of suspicious phrases in transcript
  - metadata     : optional dict, e.g. {"unknown_number": True}

Weights are intentionally simple and visible. Tune to taste.
"""

from dataclasses import dataclass


# Weight knobs — must sum to 1.0
W_CNN      = 0.60
W_KEYWORD  = 0.30
W_METADATA = 0.10

# Calibrated from finetuned model (105 real + 38 high-quality fakes).
# Real voice: mean=0.470, max=0.565 => max score 33.9
# Fake (clean TTS): min=0.561, mean=0.640 => score 33.7+
CNN_THRESHOLD = 0.56


@dataclass
class RiskResult:
    score: float          # 0-100
    band:  str            # "LOW" | "MEDIUM" | "HIGH"
    action: str           # "PASS" | "OTP_CHALLENGE" | "BLOCK"
    cnn_prob: float
    keyword_hits: list[str]
    metadata: dict


def fuse(cnn_prob: float,
         keyword_hits: list[str],
         metadata: dict | None = None) -> RiskResult:

    metadata = metadata or {}

    # 1. CNN contribution: probability of synthetic, scaled to 0-100.
    cnn_component = cnn_prob * 100.0

    # 2. Keyword contribution: each hit adds risk, capped at 100.
    #    3+ hits = full weight.
    keyword_component = min(len(keyword_hits) / 3.0, 1.0) * 100.0

    # 3. Metadata contribution: simple flag-based for now.
    md_component = 0.0
    if metadata.get("unknown_number"):  md_component += 50
    if metadata.get("voip_origin"):     md_component += 30
    if metadata.get("foreign_country"): md_component += 20
    md_component = min(md_component, 100.0)

    score = (W_CNN      * cnn_component
           + W_KEYWORD  * keyword_component
           + W_METADATA * md_component)

    # Band thresholds calibrated to finetuned model (filtered fakes).
    # Real voice max CNN ~0.565 => max score ~33.9 => safely under 34 => LOW.
    # Clean TTS min CNN ~0.574 => score ~34.4 => MEDIUM.
    if   score <= 34: band, action = "LOW",    "PASS"
    elif score <= 70: band, action = "MEDIUM", "OTP_CHALLENGE"
    else:             band, action = "HIGH",   "BLOCK"

    return RiskResult(
        score=round(score, 1),
        band=band,
        action=action,
        cnn_prob=round(cnn_prob, 3),
        keyword_hits=keyword_hits,
        metadata=metadata,
    )
