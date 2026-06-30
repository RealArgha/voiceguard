"""
Evaluate VoiceGuard CNN — confusion matrix, accuracy, precision, recall, F1.

Runs two evaluations:
  1. Personal validation set (data/mini/real + data/mini/fake) — always fast
  2. ASVspoof 2019 LA dev set (24,844 clips)         — use --dev flag

Usage
-----
    python eval_metrics.py                         # personal set only
    python eval_metrics.py --dev                   # + full ASVspoof dev set
    python eval_metrics.py --dev --max 2000        # dev set capped at 2000 per class
    python eval_metrics.py --weights weights/voiceguard_cnn_2021.pt --dev
"""

import argparse
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from backend.model       import load_model, predict_synthetic_prob
from backend.spectrogram import audio_to_logmel, SAMPLE_RATE

THRESHOLD = 0.56


# ── inference ──────────────────────────────────────────────────────────────────

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def score_file(model, path: str) -> float:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    # centre-crop to 3 s (same window the live pipeline uses)
    target = 3 * SAMPLE_RATE
    if len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[start:start + target]
    mel = audio_to_logmel(audio).to(_DEVICE)
    return predict_synthetic_prob(model, mel)


def evaluate(model, files, labels, desc: str):
    """Run model on (files, labels) and return metrics dict."""
    y_true, y_pred, scores = [], [], []
    n = len(files)
    t0 = time.time()

    for i, (path, label) in enumerate(zip(files, labels)):
        try:
            p = score_file(model, path)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue

        pred = 1 if p > THRESHOLD else 0
        scores.append(p)
        y_true.append(label)
        y_pred.append(pred)

        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  [{i+1}/{n}]  elapsed {elapsed:.0f}s  ETA {eta:.0f}s", end="\r")

    print()

    if not y_true:
        print("  No clips scored — skipping.")
        return {}

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    scores  = np.array(scores)

    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc      = accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true, y_pred, zero_division=0)
    f1       = f1_score(y_true, y_pred, zero_division=0)

    # EER via bisection on the ROC
    eer = _compute_eer(y_true, scores)

    _print_report(desc, cm, acc, prec, rec, f1, eer, len(y_true))
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "eer": eer, "cm": cm}


def _compute_eer(y_true, scores) -> float:
    """EER at the threshold where FAR == FRR."""
    thresholds = np.linspace(0, 1, 1000)
    best_eer   = 1.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        # class 1 = spoof/synthetic
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tn = np.sum((pred == 0) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        far = fp / (fp + tn + 1e-10)   # false acceptance (real flagged as fake)
        frr = fn / (fn + tp + 1e-10)   # false rejection (fake missed)
        if abs(far - frr) < abs(best_eer - (far + frr) / 2):
            best_eer = (far + frr) / 2
    return best_eer


def _print_report(desc, cm, acc, prec, rec, f1, eer, n):
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{'='*60}")
    print(f"  {desc}  (n={n}, threshold={THRESHOLD})")
    print(f"{'='*60}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                  Pred REAL   Pred FAKE")
    print(f"  Actual REAL        {tn:6d}      {fp:6d}")
    print(f"  Actual FAKE        {fn:6d}      {tp:6d}")
    print(f"\n  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%   (of flagged-fake, how many are actually fake)")
    print(f"  Recall     : {rec*100:.2f}%   (of actual fakes, how many were caught)")
    print(f"  F1-Score   : {f1*100:.2f}%")
    print(f"  EER        : {eer*100:.2f}%")
    print(f"\n  False Positive Rate : {fp/(fp+tn)*100:.2f}%  (real voices wrongly flagged)")
    print(f"  False Negative Rate : {fn/(fn+tp)*100:.2f}%  (fakes that slipped through)")
    print()


# ── data loaders ───────────────────────────────────────────────────────────────

def load_personal_set():
    real_dir = Path("data/mini/real")
    fake_dir = Path("data/mini/fake")
    if not real_dir.exists() or not fake_dir.exists():
        return None, None

    real_files = sorted(real_dir.glob("*.wav")) + sorted(real_dir.glob("*.flac"))
    fake_files = sorted(fake_dir.glob("*.wav")) + sorted(fake_dir.glob("*.flac"))

    files  = [str(f) for f in real_files + fake_files]
    labels = [0] * len(real_files) + [1] * len(fake_files)   # 0=real, 1=spoof
    return files, labels


def load_dev_set(max_per_class: int | None = None):
    protocol = Path(
        "ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_cm_protocols"
        "/ASVspoof2019.LA.cm.dev.trl.txt"
    )
    audio_dir = Path("ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_dev/flac")

    if not protocol.exists():
        print(f"  Protocol not found: {protocol}")
        return None, None

    bonafide_rows, spoof_rows = [], []
    with open(protocol) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            fid, label = parts[1], parts[4]
            path = audio_dir / f"{fid}.flac"
            if not path.exists():
                continue
            (bonafide_rows if label == "bonafide" else spoof_rows).append(str(path))

    random.shuffle(bonafide_rows)
    random.shuffle(spoof_rows)

    if max_per_class:
        bonafide_rows = bonafide_rows[:max_per_class]
        spoof_rows    = spoof_rows[:max_per_class]

    files  = bonafide_rows + spoof_rows
    labels = [0] * len(bonafide_rows) + [1] * len(spoof_rows)
    return files, labels


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None,
                        help="Override weight file (default: finetuned → 2021 → aug → base)")
    parser.add_argument("--dev", action="store_true",
                        help="Also evaluate on ASVspoof 2019 LA dev set")
    parser.add_argument("--max", type=int, default=None, dest="max_per_class",
                        help="Max files per class for the dev set (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve weights
    if args.weights:
        weights = args.weights
    else:
        for candidate in [
            "weights/voiceguard_cnn_finetuned.pt",
            "weights/voiceguard_cnn_2021.pt",
            "weights/voiceguard_cnn_aug.pt",
            "weights/voiceguard_cnn.pt",
        ]:
            if Path(candidate).is_file():
                weights = candidate
                break
        else:
            sys.exit("No weights found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {weights}  ({device.upper()})")
    model = load_model(weights, device=device)

    # ── evaluation 1: personal set ────────────────────────────────────────────
    files, labels = load_personal_set()
    if files:
        print(f"\nPersonal set: {labels.count(0)} real, {labels.count(1)} fake")
        evaluate(model, files, labels, f"Personal mic data  [{Path(weights).name}]")
    else:
        print("Personal data/mini/ not found — skipping.")

    # ── evaluation 2: ASVspoof dev set ────────────────────────────────────────
    if args.dev:
        files, labels = load_dev_set(args.max_per_class)
        if files:
            n_real = labels.count(0)
            n_spoof = labels.count(1)
            print(f"ASVspoof 2019 LA dev: {n_real} bonafide, {n_spoof} spoof")
            cap = f"  (capped at {args.max_per_class}/class)" if args.max_per_class else ""
            evaluate(model, files, labels,
                     f"ASVspoof 2019 LA dev{cap}  [{Path(weights).name}]")
        else:
            print("ASVspoof dev set not found — skipping.")


if __name__ == "__main__":
    main()
