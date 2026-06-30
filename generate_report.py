"""
VoiceGuard — Evaluation Report Generator

Evaluates the model on three datasets and produces a multi-page PDF report
with confusion matrices, ROC curves, score distributions, and metrics tables.

Datasets
--------
  1. Personal mic validation (data/mini/real + fake) — 143 clips
  2. ASVspoof 2019 LA dev set — 24,844 clips
  3. ASVspoof 2021 LA eval set — up to MAX_PER_CLASS per class

Usage
-----
    python generate_report.py
    python generate_report.py --max 3000    # cap 2021 clips per class (default 3000)
    python generate_report.py --out report.pdf
"""

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve, auc,
)

from backend.model       import load_model, predict_synthetic_prob
from backend.spectrogram import audio_to_logmel, SAMPLE_RATE

THRESHOLD = 0.56
_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ── palette ────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
CARD     = "#161b22"
ACCENT   = "#00d4aa"
RED      = "#ff4d6d"
BLUE     = "#4361ee"
MUTED    = "#8b949e"
WHITE    = "#e6edf3"
GOLD     = "#ffd166"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    MUTED,
    "axes.labelcolor":   WHITE,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        WHITE,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})


# ── inference ──────────────────────────────────────────────────────────────────

def score_file(model, path: str) -> float:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    target = 3 * SAMPLE_RATE
    if len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[start:start + target]
    mel = audio_to_logmel(audio).to(_DEVICE)
    return predict_synthetic_prob(model, mel)


def run_eval(model, files, labels, desc: str):
    y_true, y_pred, scores = [], [], []
    n  = len(files)
    t0 = time.time()
    print(f"\n  Scoring {n} clips for [{desc}]...")

    for i, (path, label) in enumerate(zip(files, labels)):
        try:
            p = score_file(model, path)
        except Exception as e:
            print(f"\n  SKIP {path}: {e}")
            continue

        y_true.append(label)
        y_pred.append(1 if p > THRESHOLD else 0)
        scores.append(p)

        if (i + 1) % 500 == 0 or (i + 1) == n:
            el  = time.time() - t0
            eta = el / (i + 1) * (n - i - 1)
            print(f"  [{i+1}/{n}]  {el:.0f}s elapsed  ETA {eta:.0f}s", end="\r")

    print()

    if not y_true:
        return None

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    scores  = np.array(scores)

    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr_roc, tpr_roc, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr_roc, tpr_roc)

    eer = _compute_eer(y_true, scores)

    result = {
        "desc":      desc,
        "n":         len(y_true),
        "y_true":    y_true,
        "y_pred":    y_pred,
        "scores":    scores,
        "cm":        cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "eer":       eer,
        "fpr_roc":   fpr_roc,
        "tpr_roc":   tpr_roc,
        "roc_auc":   roc_auc,
        "fpr":       fp / (fp + tn + 1e-10),
        "fnr":       fn / (fn + tp + 1e-10),
    }

    # console summary
    print(f"\n  {'='*50}")
    print(f"  {desc}  (n={result['n']})")
    print(f"  Accuracy  {result['accuracy']*100:.2f}%   F1 {result['f1']*100:.2f}%")
    print(f"  Precision {result['precision']*100:.2f}%   Recall {result['recall']*100:.2f}%")
    print(f"  EER       {result['eer']*100:.2f}%   AUC {result['roc_auc']:.4f}")
    print(f"  CM: TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    return result


def _compute_eer(y_true, scores) -> float:
    thresholds = np.linspace(0, 1, 2000)
    best = 1.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tn = np.sum((pred == 0) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        far = fp / (fp + tn + 1e-10)
        frr = fn / (fn + tp + 1e-10)
        if abs(far - frr) < abs(best - (far + frr) / 2):
            best = (far + frr) / 2
    return best


# ── data loaders ───────────────────────────────────────────────────────────────

def load_personal():
    rd = Path("data/mini/real")
    fd = Path("data/mini/fake")
    if not rd.exists():
        return None, None
    rf = sorted(rd.glob("*.wav")) + sorted(rd.glob("*.flac"))
    ff = sorted(fd.glob("*.wav")) + sorted(fd.glob("*.flac"))
    return [str(f) for f in rf + ff], [0]*len(rf) + [1]*len(ff)


def load_asvspoof2019_dev():
    proto = Path("ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_cm_protocols"
                 "/ASVspoof2019.LA.cm.dev.trl.txt")
    audio = Path("ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_dev/flac")
    if not proto.exists():
        return None, None
    bon, spf = [], []
    with open(proto) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) < 5:
                continue
            path = audio / f"{p[1]}.flac"
            if not path.exists():
                continue
            (bon if p[4] == "bonafide" else spf).append(str(path))
    return bon + spf, [0]*len(bon) + [1]*len(spf)


def load_asvspoof2021_la(max_per_class: int, seed: int):
    proto = Path("archive/LA-keys-full/keys/LA/CM/trial_metadata.txt")
    audio = Path("archive/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac")
    if not proto.exists():
        return None, None
    bon, spf = [], []
    with open(proto) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) < 6:
                continue
            path = audio / f"{p[1]}.flac"
            if not path.exists():
                continue
            (bon if p[5] == "bonafide" else spf).append(str(path))
    rng = random.Random(seed)
    rng.shuffle(bon); rng.shuffle(spf)
    bon = bon[:max_per_class]; spf = spf[:max_per_class]
    return bon + spf, [0]*len(bon) + [1]*len(spf)


# ── PDF pages ──────────────────────────────────────────────────────────────────

def _fig(w=11, h=8.5):
    return plt.figure(figsize=(w, h), facecolor=BG)


def page_cover(pdf, weights_path, results, generated_at):
    fig = _fig()
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # gradient header band
    ax.add_patch(FancyBboxPatch((0, 0.72), 1, 0.28,
                                boxstyle="square,pad=0",
                                facecolor="#0f2027", edgecolor="none"))
    ax.axhline(0.72, color=ACCENT, linewidth=3)

    ax.text(0.5, 0.90, "VoiceGuard", ha="center", fontsize=44,
            fontweight="bold", color=ACCENT)
    ax.text(0.5, 0.80, "AI Voice Detection — Evaluation Report",
            ha="center", fontsize=18, color=WHITE)

    ax.text(0.5, 0.68, f"Model: {Path(weights_path).name}",
            ha="center", fontsize=13, color=MUTED)
    ax.text(0.5, 0.63, f"Generated: {generated_at}",
            ha="center", fontsize=11, color=MUTED)

    # summary cards
    labels  = ["Accuracy", "F1-Score", "Precision", "Recall", "EER"]
    dataset = results[0]                    # personal set headline numbers
    values  = [
        f"{dataset['accuracy']*100:.1f}%",
        f"{dataset['f1']*100:.1f}%",
        f"{dataset['precision']*100:.1f}%",
        f"{dataset['recall']*100:.1f}%",
        f"{dataset['eer']*100:.2f}%",
    ]
    colors  = [ACCENT, BLUE, GOLD, "#a8dadc", RED]

    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        x = 0.05 + i * 0.19
        ax.add_patch(FancyBboxPatch((x, 0.44), 0.17, 0.13,
                                    boxstyle="round,pad=0.01",
                                    facecolor=CARD, edgecolor=col,
                                    linewidth=2))
        ax.text(x + 0.085, 0.535, val, ha="center", fontsize=16,
                fontweight="bold", color=col)
        ax.text(x + 0.085, 0.458, lab, ha="center", fontsize=9, color=MUTED)

    ax.text(0.5, 0.40, "Based on personal mic validation set",
            ha="center", fontsize=9, color=MUTED, style="italic")

    # dataset table
    headers = ["Dataset", "Clips", "Accuracy", "F1", "Precision", "Recall", "EER"]
    col_x   = [0.03, 0.28, 0.42, 0.52, 0.61, 0.71, 0.82]

    ax.add_patch(FancyBboxPatch((0.02, 0.12), 0.96, 0.25,
                                boxstyle="round,pad=0.01",
                                facecolor=CARD, edgecolor=MUTED, linewidth=1))

    for j, (h, x) in enumerate(zip(headers, col_x)):
        ax.text(x, 0.355, h, fontsize=9, fontweight="bold", color=ACCENT)

    ax.axhline(0.345, xmin=0.02, xmax=0.98, color=MUTED, linewidth=0.5)

    for i, r in enumerate(results):
        y = 0.305 - i * 0.065
        row = [
            r["desc"],
            f"{r['n']:,}",
            f"{r['accuracy']*100:.2f}%",
            f"{r['f1']*100:.2f}%",
            f"{r['precision']*100:.2f}%",
            f"{r['recall']*100:.2f}%",
            f"{r['eer']*100:.2f}%",
        ]
        for j, (val, x) in enumerate(zip(row, col_x)):
            col = WHITE if j > 0 else GOLD
            ax.text(x, y, val, fontsize=9, color=col)

    ax.text(0.5, 0.05,
            "Threshold = 0.56  |  Label: 0 = bonafide (human), 1 = spoof (AI/TTS)",
            ha="center", fontsize=8, color=MUTED)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_confusion_matrices(pdf, results):
    fig = _fig(11, 8.5)
    fig.suptitle("Confusion Matrices", fontsize=16, color=WHITE, y=0.97)

    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    axes = fig.subplots(rows, cols, squeeze=False)

    for idx, r in enumerate(results):
        ax  = axes[idx // cols][idx % cols]
        cm  = r["cm"].astype(float)
        tot = cm.sum(axis=1, keepdims=True) + 1e-10
        pct = cm / tot * 100

        cmap = plt.cm.get_cmap("YlOrRd")
        img  = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, aspect="auto")

        labels_xy = ["Real (bonafide)", "Fake (synthetic)"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels_xy, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels_xy, fontsize=8, rotation=45)
        ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
        ax.set_ylabel("Actual",    color=MUTED, fontsize=9)
        ax.set_title(r["desc"], fontsize=9, color=ACCENT, pad=8)

        tn, fp, fn, tp = r["tn"], r["fp"], r["fn"], r["tp"]
        cells = [[tn, fp], [fn, tp]]
        names = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val = cells[i][j]
                pv  = pct[i][j]
                ax.text(j, i, f"{names[i][j]}\n{val:,}\n({pv:.1f}%)",
                        ha="center", va="center", fontsize=8,
                        color="white" if pv > 50 else "#1a1a2e",
                        fontweight="bold")

        plt.colorbar(img, ax=ax, shrink=0.75, label="%")

    # hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_roc_curves(pdf, results):
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_facecolor(CARD)

    colors = [ACCENT, BLUE, GOLD, RED]
    for r, col in zip(results, colors):
        ax.plot(r["fpr_roc"], r["tpr_roc"],
                color=col, linewidth=2,
                label=f"{r['desc']}  (AUC={r['roc_auc']:.4f})")

    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="Random classifier")

    # EER points
    for r, col in zip(results, colors):
        eer_pt = r["eer"]
        ax.scatter([eer_pt], [1 - eer_pt], color=col, s=80, zorder=5)
        ax.annotate(f"EER={eer_pt*100:.2f}%",
                    (eer_pt, 1 - eer_pt),
                    textcoords="offset points", xytext=(8, -10),
                    fontsize=7.5, color=col)

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax.set_title("ROC Curves — All Datasets", fontsize=14, color=WHITE, pad=12)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=MUTED, labelcolor=WHITE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_score_distributions(pdf, results):
    n   = len(results)
    fig = _fig(11, 4.5 * n)
    fig.suptitle("CNN Score Distributions  (threshold = 0.56)", fontsize=14,
                 color=WHITE, y=0.99)

    axes = fig.subplots(n, 1, squeeze=False)

    for idx, r in enumerate(results):
        ax     = axes[idx][0]
        real   = r["scores"][r["y_true"] == 0]
        fake   = r["scores"][r["y_true"] == 1]
        bins   = np.linspace(0, 1, 50)

        ax.hist(real, bins=bins, alpha=0.65, color=BLUE,  label="Real (bonafide)", density=True)
        ax.hist(fake, bins=bins, alpha=0.65, color=RED,   label="AI/TTS (spoof)",  density=True)
        ax.axvline(THRESHOLD, color=GOLD, linewidth=2, linestyle="--",
                   label=f"Threshold = {THRESHOLD}")

        ax.set_title(r["desc"], fontsize=10, color=ACCENT)
        ax.set_xlabel("CNN synthetic probability score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, facecolor=CARD, edgecolor=MUTED, labelcolor=WHITE)
        ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.98], pad=1.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_metrics_table(pdf, results):
    fig = _fig(11, 8.5)
    fig.suptitle("Detailed Metrics by Dataset", fontsize=16, color=WHITE, y=0.97)
    ax  = fig.add_axes([0.02, 0.05, 0.96, 0.88])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    headers = [
        "Dataset", "Clips", "Real", "Fake",
        "TP", "TN", "FP", "FN",
        "Accuracy", "Precision", "Recall",
        "F1-Score", "EER", "AUC-ROC",
        "FPR", "FNR",
    ]
    col_w  = [0.16, 0.05, 0.045, 0.045,
              0.04, 0.05, 0.04, 0.04,
              0.065, 0.065, 0.055,
              0.065, 0.045, 0.065,
              0.045, 0.045]
    col_x  = [0.0]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w)

    row_h  = 0.88 / (len(results) + 2)
    y_head = 1.0 - row_h

    # header row
    ax.add_patch(FancyBboxPatch((0, y_head - 0.01), 1, row_h + 0.01,
                                boxstyle="square,pad=0",
                                facecolor="#1c2b3a", edgecolor="none"))
    for h, x in zip(headers, col_x):
        ax.text(x + 0.005, y_head + row_h * 0.3, h,
                fontsize=7.5, fontweight="bold", color=ACCENT, wrap=True)

    ax.axhline(y_head, color=ACCENT, linewidth=1)

    for i, r in enumerate(results):
        y = y_head - (i + 1) * row_h
        bg = "#161b22" if i % 2 == 0 else "#0d1117"
        ax.add_patch(FancyBboxPatch((0, y), 1, row_h,
                                    boxstyle="square,pad=0",
                                    facecolor=bg, edgecolor="none"))

        real_n = int((r["y_true"] == 0).sum())
        fake_n = int((r["y_true"] == 1).sum())
        row = [
            r["desc"], f"{r['n']:,}",
            f"{real_n:,}", f"{fake_n:,}",
            f"{r['tp']:,}", f"{r['tn']:,}",
            f"{r['fp']:,}", f"{r['fn']:,}",
            f"{r['accuracy']*100:.2f}%",
            f"{r['precision']*100:.2f}%",
            f"{r['recall']*100:.2f}%",
            f"{r['f1']*100:.2f}%",
            f"{r['eer']*100:.2f}%",
            f"{r['roc_auc']:.4f}",
            f"{r['fpr']*100:.2f}%",
            f"{r['fnr']*100:.2f}%",
        ]
        for val, x in zip(row, col_x):
            ax.text(x + 0.005, y + row_h * 0.3, val, fontsize=7.5, color=WHITE)

        ax.axhline(y, color=MUTED, linewidth=0.3)

    # footnote
    ax.text(0.5, 0.01,
            "FPR = Real voice wrongly flagged as AI  |  FNR = AI voice that slipped through  |  "
            "EER = Equal Error Rate (lower is better)  |  AUC-ROC closer to 1.0 is better",
            ha="center", fontsize=7, color=MUTED)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_architecture(pdf, weights_path):
    fig = _fig()
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.95, "Model Architecture & Training Summary",
            ha="center", fontsize=18, fontweight="bold", color=WHITE)
    ax.axhline(0.92, color=ACCENT, linewidth=2)

    sections = [
        ("VoiceGuardCNN Architecture", ACCENT, [
            "Input : Log-mel spectrogram  (B, 1, 128, T)  —  128 mel bins, 10 ms hop, 16 kHz",
            "Block 1: Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool2d(2×2)",
            "Block 2: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool2d(2×2)",
            "Block 3: Conv2d(64→128, 3×3) → BatchNorm → ReLU → MaxPool2d(2×2)",
            "Global Average Pool  →  Flatten  →  FC(128→1)  →  Sigmoid",
            "Output: P(synthetic) ∈ [0, 1]   |   Threshold: 0.56",
        ]),
        ("Training Datasets", BLUE, [
            "Phase 1 — ASVspoof 2019 LA:   2,580 bonafide  +  22,800 spoof  (train split)",
            "Phase 2 — ASVspoof 2021 LA:  181,566 clips  (codec / telephone conditions)",
            "Phase 3 — ASVspoof 2021 DF:  611,829 clips  (deepfake / neural TTS)",
            "Phase 4 — ASVspoof 2021 PA:  up to 60k clips  (physical access / replay)",
            "Fine-tune — Personal mic:    105 real  +  38 high-confidence fake clips",
            "Total training exposure:     > 640,000 labelled audio clips",
        ]),
        ("Training Details", GOLD, [
            "Optimizer : AdamW  |  Loss: BCELoss  |  Mixed precision (AMP, GradScaler)",
            "LR schedule: cosine annealing with 2-epoch warmup",
            "Augmentation: gain, Gaussian noise (10–30 dB SNR), low-pass filter, codec sim",
            "SpecAugment: 2 freq masks (max 20 bins)  +  2 time masks (max 30 frames)",
            "Class balancing: WeightedRandomSampler  (counters 9:1 spoof/bonafide imbalance)",
            "Fine-tune: progressive unfreezing  (FC-only phase → full network at 0.1× LR)",
        ]),
        ("Live Pipeline", ACCENT, [
            "Browser mic → 16 kHz float32 PCM → WebSocket (64 KB / 1-second frames)",
            "3-second rolling buffer  +  3-frame CNN score smoothing (deque maxlen=3)",
            "CNN + Whisper run concurrently via asyncio.gather every frame",
            "Fusion: score = 0.60×CNN + 0.30×keyword + 0.10×metadata",
            "Band thresholds: LOW <= 34  |  MEDIUM 34-45  |  HIGH > 45",
            "Actions: PASS  |  OTP_CHALLENGE  |  BLOCK",
        ]),
    ]

    y = 0.88
    for title, col, items in sections:
        ax.text(0.04, y, title, fontsize=12, fontweight="bold", color=col)
        y -= 0.04
        for item in items:
            ax.text(0.07, y, f"• {item}", fontsize=9, color=WHITE)
            y -= 0.033
        y -= 0.02

    ax.text(0.5, 0.025,
            f"Active weights: {Path(weights_path).name}  |  Device: {_DEVICE.upper()}",
            ha="center", fontsize=9, color=MUTED)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max",  type=int, default=3000,
                        help="Max clips per class for 2021 eval (default 3000)")
    parser.add_argument("--out",  default="voiceguard_eval_report.pdf")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # resolve weights
    if args.weights:
        weights = args.weights
    else:
        for c in ["weights/voiceguard_cnn_finetuned.pt",
                  "weights/voiceguard_cnn_2021.pt",
                  "weights/voiceguard_cnn_aug.pt",
                  "weights/voiceguard_cnn.pt"]:
            if Path(c).is_file():
                weights = c; break
        else:
            sys.exit("No weights found.")

    print(f"\nVoiceGuard Evaluation Report")
    print(f"  Model : {weights}")
    print(f"  Device: {_DEVICE.upper()}")
    print(f"  Output: {args.out}\n")

    model = load_model(weights, device=_DEVICE)

    results = []

    # 1. Personal set
    files, labels = load_personal()
    if files:
        r = run_eval(model, files, labels, "Personal Mic Validation")
        if r: results.append(r)

    # 2. ASVspoof 2019 LA dev
    files, labels = load_asvspoof2019_dev()
    if files:
        r = run_eval(model, files, labels, "ASVspoof 2019 LA Dev")
        if r: results.append(r)

    # 3. ASVspoof 2021 LA eval
    files, labels = load_asvspoof2021_la(args.max, args.seed)
    if files:
        r = run_eval(model, files, labels,
                     f"ASVspoof 2021 LA Eval (max {args.max}/class)")
        if r: results.append(r)

    if not results:
        sys.exit("No datasets found — nothing to report.")

    # ── build PDF ─────────────────────────────────────────────────────────────
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\nBuilding PDF: {args.out}")

    with PdfPages(args.out) as pdf:
        page_cover(pdf, weights, results, generated_at)
        page_confusion_matrices(pdf, results)
        page_roc_curves(pdf, results)
        page_score_distributions(pdf, results)
        page_metrics_table(pdf, results)
        page_architecture(pdf, weights)

        info = pdf.infodict()
        info["Title"]   = "VoiceGuard — Evaluation Report"
        info["Author"]  = "VoiceGuard CNN"
        info["Subject"] = "AI Voice Detection Metrics"
        info["Keywords"] = "ASVspoof, anti-spoofing, EER, confusion matrix, ROC"

    print(f"\nDone — {args.out}")


if __name__ == "__main__":
    main()
