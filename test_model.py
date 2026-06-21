
"""
Quick diagnostic: score audio files directly through the trained CNN,
bypassing the browser-mic pipeline entirely.

USAGE
-----
    python test_model.py clip.wav
    python test_model.py clip1.mp3 clip2.wav clip3.flac          # batch
    python test_model.py --weights weights/voiceguard_cnn_aug.pt clip.mp3
"""

import argparse
import sys
from pathlib import Path

import librosa
import torch

from backend.model       import load_model, predict_synthetic_prob
from backend.spectrogram import audio_to_logmel, SAMPLE_RATE


def score(model, path: str) -> float:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if len(audio) < SAMPLE_RATE:
        print(f"  ⚠ clip too short ({len(audio)/SAMPLE_RATE:.2f}s)")

    target = 3 * SAMPLE_RATE
    if len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[start:start + target]

    mel = audio_to_logmel(audio)
    return predict_synthetic_prob(model, mel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",
                        default="weights/voiceguard_cnn.pt",
                        help="path to a .pt checkpoint")
    parser.add_argument("files", nargs="+",
                        help="audio files to score")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"No weights at {args.weights}")

    print(f"Loading model from {args.weights}...")
    model = load_model(args.weights)
    print()

    for path in args.files:
        if not Path(path).exists():
            print(f"  SKIP: {path} not found"); continue
        try:
            p = score(model, path)
            verdict = "SYNTHETIC" if p > 0.56 else "REAL"
            bar = "#" * int(p * 30) + "-" * (30 - int(p * 30))
            print(f"  {p:.3f}  {bar}  {verdict:9}  {path}")
        except Exception as e:
            print(f"  ERROR on {path}: {e}")


if __name__ == "__main__":
    main()