"""
Train VoiceGuardCNN on the combined ASVspoof 2019 LA + 2021 LA + 2021 DF + 2021 PA corpus.

This script is ADDITIVE — it never touches or deletes the 2019 dataset.
It starts from the best existing weights (transfer-learning) and trains
on the combined corpus, saving to weights/voiceguard_cnn_2021.pt.

Dataset layout expected
-----------------------
  ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_train/flac/           ← 2019 train
  ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_dev/flac/             ← 2019 dev (val only)
  ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_cm_protocols/         ← 2019 labels
  archive/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac/   ← 2021 LA (181k files)
  archive/ASVspoof2021_DF_eval_part{00,01,02}/
        ASVspoof2021_DF_eval/flac/                          ← 2021 DF (458k files)
  archive/ASVspoof2021_PA_eval_part{00,01,02,03}/
        ASVspoof2021_PA_eval/flac/                          ← 2021 PA (600k files)
  archive/LA-keys-full/keys/LA/CM/trial_metadata.txt        ← 2021 LA labels
  archive/DF-keys-full/keys/DF/CM/trial_metadata.txt        ← 2021 DF labels
  archive/PA-keys-full/keys/PA/CM/trial_metadata.txt        ← 2021 PA labels

Protocol column layout
----------------------
  2019:     col[1]=file_id  col[4]=label(bonafide|spoof)
  2021 LA:  col[1]=file_id  col[5]=label(bonafide|spoof)
  2021 DF:  col[1]=file_id  col[5]=label(bonafide|spoof)
  2021 PA:  col[1]=file_id  col[9]=label(bonafide|spoof)

USAGE
-----
    # Recommended: augmentation + 10 epochs (GPU ~3 h)
    python -m backend.train_2021 --augment --epochs 10

    # Start from random init instead of transfer-learning
    python -m backend.train_2021 --augment --epochs 10 --from-scratch

    # Quick sanity check (2 epochs, 1 k samples)
    python -m backend.train_2021 --quick

    # Control dataset caps (default 60k each for 2021 splits)
    python -m backend.train_2021 --augment --max-la21 40000 --max-df21 80000 --max-pa21 60000

Output
------
    weights/voiceguard_cnn_2021.pt   (best val-EER checkpoint)
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import (
    ConcatDataset, DataLoader, Dataset, WeightedRandomSampler,
)
from tqdm import tqdm

from .model        import VoiceGuardCNN
from .spectrogram  import audio_to_logmel, SAMPLE_RATE
from .augmentation import augment_audio, spec_augment
from .train        import equal_error_rate

# ── paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent.parent          # project root

# 2019 — honour ASVSPOOF_ROOT env var the same way train.py does
_ASVROOT = Path(os.environ.get(
    "ASVSPOOF_ROOT",
    str(_HERE / "ASVSPOOF_ROOT" / "LA" / "LA"),
))
_PROTO_2019      = _ASVROOT / "ASVspoof2019_LA_cm_protocols"
_TRAIN_2019_FLAC = _ASVROOT / "ASVspoof2019_LA_train" / "flac"
_DEV_2019_FLAC   = _ASVROOT / "ASVspoof2019_LA_dev"   / "flac"
_TRAIN_PROTO_2019 = _PROTO_2019 / "ASVspoof2019.LA.cm.train.trn.txt"
_DEV_PROTO_2019   = _PROTO_2019 / "ASVspoof2019.LA.cm.dev.trl.txt"

# 2021
_ARCHIVE   = _HERE / "archive"
_LA21_FLAC = _ARCHIVE / "ASVspoof2021_LA_eval" / "ASVspoof2021_LA_eval" / "flac"
_DF21_PARTS = [
    _ARCHIVE / f"ASVspoof2021_DF_eval_part{i:02d}" / "ASVspoof2021_DF_eval" / "flac"
    for i in range(3)
]
_PA21_PARTS = [
    _ARCHIVE / f"ASVspoof2021_PA_eval_part{i:02d}" / "ASVspoof2021_PA_eval" / "flac"
    for i in range(4)
]
_LA21_PROTO = _ARCHIVE / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt"
_DF21_PROTO = _ARCHIVE / "DF-keys-full" / "keys" / "DF" / "CM" / "trial_metadata.txt"
_PA21_PROTO = _ARCHIVE / "PA-keys-full" / "keys" / "PA" / "CM" / "trial_metadata.txt"

CLIP_SAMPLES = 3 * SAMPLE_RATE          # 3 s clips


# ── protocol parsers ──────────────────────────────────────────────────────────

def _parse_2019(proto: Path, flac_dir: Path,
                limit: int | None = None) -> list[tuple[Path, int]]:
    """Parse ASVspoof 2019 CM protocol.  Label at column index 4."""
    items: list[tuple[Path, int]] = []
    with open(proto) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            p = flac_dir / f"{parts[1]}.flac"
            if p.exists():
                items.append((p, 0 if parts[4] == "bonafide" else 1))
    if limit:
        random.Random(42).shuffle(items)
        items = items[:limit]
    return items


def _build_lookup(dirs: list[Path]) -> dict[str, Path]:
    """Scan one or more flac directories → {file_stem: path}."""
    lookup: dict[str, Path] = {}
    for d in dirs:
        if not d.exists():
            print(f"  [warn] directory missing: {d}")
            continue
        for p in tqdm(d.glob("*.flac"), desc=f"  scan {d.name}", leave=False):
            lookup[p.stem] = p
    return lookup


def _parse_2021(proto: Path, lookup: dict[str, Path],
                limit: int | None = None,
                label_col: int = 5) -> list[tuple[Path, int]]:
    """Parse ASVspoof 2021 CM protocol.

    label_col=5  for LA/DF eval protocols
    label_col=9  for PA eval protocol (parts[9] = CM label)

    Applies stratified sampling when limit is set so both classes are
    represented: up to limit//2 bonafide + remainder from spoof.
    """
    bonafide: list[tuple[Path, int]] = []
    spoof:    list[tuple[Path, int]] = []

    with open(proto) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= label_col:
                continue
            file_id = parts[1]
            if file_id not in lookup:
                continue
            if parts[label_col] == "bonafide":
                bonafide.append((lookup[file_id], 0))
            else:
                spoof.append((lookup[file_id], 1))

    rng = random.Random(42)
    rng.shuffle(bonafide)
    rng.shuffle(spoof)

    if limit:
        n_bon  = min(limit // 2, len(bonafide))
        n_spo  = min(limit - n_bon, len(spoof))
        items  = bonafide[:n_bon] + spoof[:n_spo]
    else:
        items = bonafide + spoof

    rng.shuffle(items)
    return items


# ── dataset ───────────────────────────────────────────────────────────────────

class _AudioDataset(Dataset):
    """Generic (path, label) dataset with mel caching and optional augmentation.

    mel_cache=True  : cache log-mel .npy → fast, no audio augmentation
    mel_cache=False : cache raw audio .npy → slower, full augmentation pipeline
    """

    def __init__(self, items: list[tuple[Path, int]], cache_dir: Path,
                 augment: bool = False, mel_cache: bool = True,
                 name: str = ""):
        self.items     = items
        self.augment   = augment
        self.mel_cache = mel_cache
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.labels    = np.array([lbl for _, lbl in items])
        counts = np.bincount(self.labels, minlength=2)
        tag = f"[{name or cache_dir.name}]"
        print(f"  {tag:20s}  bonafide={counts[0]:>6}  spoof={counts[1]:>6}  "
              f"{'augment' if augment else 'no-aug'}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        cache_path  = self.cache_dir / f"{path.stem}.npy"

        if cache_path.exists():
            cached = np.load(cache_path)
        else:
            audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
            audio    = _pad_or_crop(audio)
            cached   = audio_to_logmel(audio).numpy() if self.mel_cache \
                       else audio.astype(np.float32)
            np.save(cache_path, cached)

        if self.mel_cache:
            mel = cached
            if self.augment:
                mel = spec_augment(mel)
        else:
            audio = cached
            if self.augment:
                audio = augment_audio(audio)
            mel = audio_to_logmel(audio).numpy()
            if self.augment:
                mel = spec_augment(mel)

        return (torch.from_numpy(mel).unsqueeze(0),
                torch.tensor(label, dtype=torch.float32))


def _pad_or_crop(audio: np.ndarray) -> np.ndarray:
    if len(audio) < CLIP_SAMPLES:
        return np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    return audio[:CLIP_SAMPLES]


# ── epoch loop ────────────────────────────────────────────────────────────────

def _run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
               opt: torch.optim.Optimizer | None,
               device: str, train: bool,
               scaler: GradScaler | None = None) -> dict:
    model.train() if train else model.eval()
    total_loss = total_correct = total_n = 0
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc="train" if train else "val ", leave=False):
            x, y = x.to(device), y.to(device)

            if train and scaler is not None:
                # AMP forward — BCELoss must run in float32, so cast pred out of autocast
                with autocast(device_type="cuda"):
                    pred = model(x)
                loss = loss_fn(pred.float(), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                with autocast(device_type="cuda", enabled=(device == "cuda")):
                    pred = model(x)
                loss = loss_fn(pred.float(), y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            total_loss    += loss.item() * x.size(0)
            total_correct += ((pred >= 0.5).float() == y).sum().item()
            total_n       += x.size(0)
            all_scores.append(pred.detach().float().cpu().numpy())
            all_labels.append(y.detach().float().cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return {
        "loss":     total_loss / total_n,
        "accuracy": total_correct / total_n,
        "eer":      equal_error_rate(scores, labels),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def train(epochs: int, batch_size: int, lr: float,
          quick: bool, augment: bool, from_scratch: bool,
          max_la21: int, max_df21: int, max_pa21: int) -> None:

    if not _TRAIN_2019_FLAC.exists():
        sys.exit(f"[error] 2019 train flac not found: {_TRAIN_2019_FLAC}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"\n[train_2021] device={device}  amp={use_amp}  augment={augment}")

    # ── caps for quick/full runs ────────────────────────────────────────────
    lim_2019 = 1_000        if quick else None
    lim_la21 = 400          if quick else max_la21
    lim_df21 = 400          if quick else max_df21
    lim_pa21 = 200          if quick else max_pa21
    lim_dev  = 250          if quick else None

    # ── parse manifests ─────────────────────────────────────────────────────
    print("\nLoading manifests ...")
    items_train_2019 = _parse_2019(_TRAIN_PROTO_2019, _TRAIN_2019_FLAC, lim_2019)
    items_dev_2019   = _parse_2019(_DEV_PROTO_2019,   _DEV_2019_FLAC,   lim_dev)

    print("  Building 2021 LA file lookup ...")
    la21_lookup = _build_lookup([_LA21_FLAC])
    print(f"  -> {len(la21_lookup)} LA files found")

    print("  Building 2021 DF file lookup (3 parts) ...")
    df21_lookup = _build_lookup(_DF21_PARTS)
    print(f"  -> {len(df21_lookup)} DF files found")

    items_la21 = _parse_2021(_LA21_PROTO, la21_lookup, lim_la21)
    items_df21 = _parse_2021(_DF21_PROTO, df21_lookup, lim_df21)

    # PA — optional; skip gracefully if data is missing
    items_pa21: list[tuple[Path, int]] = []
    if _PA21_PROTO.exists():
        print("  Building 2021 PA file lookup (4 parts) ...")
        pa21_lookup = _build_lookup(_PA21_PARTS)
        print(f"  -> {len(pa21_lookup)} PA files found")
        if pa21_lookup:
            items_pa21 = _parse_2021(_PA21_PROTO, pa21_lookup, lim_pa21,
                                     label_col=9)
    else:
        print(f"  [skip] PA protocol not found: {_PA21_PROTO}")

    # ── create datasets ──────────────────────────────────────────────────────
    print("\nDataset summary:")
    mel_cache = not augment          # raw-audio cache when augmenting
    cache_pfx = "cache_audio" if augment else "cache"

    ds_train_2019 = _AudioDataset(
        items_train_2019,
        Path(f"{cache_pfx}/train"),
        augment=augment, mel_cache=mel_cache, name="2019-train",
    )
    ds_la21 = _AudioDataset(
        items_la21,
        Path(f"{cache_pfx}/la21"),
        augment=augment, mel_cache=mel_cache, name="2021-LA",
    )
    ds_df21 = _AudioDataset(
        items_df21,
        Path(f"{cache_pfx}/df21"),
        augment=augment, mel_cache=mel_cache, name="2021-DF",
    )
    # Dev never augmented — always use mel cache; reuse existing cache/dev/
    ds_dev = _AudioDataset(
        items_dev_2019,
        Path("cache/dev"),
        augment=False, mel_cache=True, name="2019-dev",
    )

    train_parts = [ds_train_2019, ds_la21, ds_df21]
    label_parts = [ds_train_2019.labels, ds_la21.labels, ds_df21.labels]

    if items_pa21:
        ds_pa21 = _AudioDataset(
            items_pa21,
            Path(f"{cache_pfx}/pa21"),
            augment=augment, mel_cache=mel_cache, name="2021-PA",
        )
        train_parts.append(ds_pa21)
        label_parts.append(ds_pa21.labels)

    combined   = ConcatDataset(train_parts)
    all_labels = np.concatenate(label_parts)

    counts         = np.bincount(all_labels)
    sample_weights = (1.0 / counts)[all_labels]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights.astype(np.float64)),
        num_samples=len(combined),
        replacement=True,
    )

    nw   = min(4, os.cpu_count() or 1)
    pin  = (device == "cuda")
    train_loader = DataLoader(combined, batch_size=batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    dev_loader   = DataLoader(ds_dev,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))

    print(f"\n[train_2021] combined train={len(combined):,}  dev={len(ds_dev):,}"
          f"  class balance  bonafide={counts[0]:,}  spoof={counts[1]:,}")

    # ── model ────────────────────────────────────────────────────────────────
    model = VoiceGuardCNN().to(device)
    if not from_scratch:
        candidates = [
            "weights/voiceguard_cnn_aug.pt",
            "weights/voiceguard_cnn.pt",
        ]
        base = next((c for c in candidates if Path(c).exists()), None)
        if base:
            state = torch.load(base, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"[model] transfer-learning from {base}")
        else:
            print("[model] no existing weights - starting from random init")
    else:
        print("[model] --from-scratch: random init")

    # Cosine-annealing LR schedule with a short linear warm-up
    warmup_epochs = max(1, epochs // 10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        t = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    loss_fn   = nn.BCELoss()
    scaler    = GradScaler("cuda") if use_amp else None

    out_path = Path("weights/voiceguard_cnn_2021.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train_2021] best checkpoint -> {out_path}\n")

    best_eer = 1.0

    def _save_interrupt(*_):
        p = out_path.with_suffix(".interrupt.pt")
        torch.save(model.state_dict(), p)
        print(f"\n[train_2021] interrupted - saved {p}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_interrupt)

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(epochs):
        current_lr = scheduler.get_last_lr()[0] if epoch > 0 else lr
        tr = _run_epoch(model, train_loader, loss_fn, opt,
                        device, train=True,  scaler=scaler)
        va = _run_epoch(model, dev_loader,   loss_fn, None,
                        device, train=False, scaler=None)
        scheduler.step()

        print(
            f"epoch {epoch+1:>2}/{epochs}  "
            f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.3f}  |  "
            f"val loss {va['loss']:.4f} acc {va['accuracy']:.3f} "
            f"EER {va['eer']*100:.2f}%  "
            f"lr {current_lr:.2e}"
        )

        if not np.isnan(va["eer"]) and va["eer"] < best_eer:
            best_eer = va["eer"]
            torch.save(model.state_dict(), out_path)
            print(f"  ^ new best EER {best_eer*100:.2f}% -> {out_path}")

    print(f"\n[train_2021] done.  best val EER: {best_eer*100:.2f}%")
    print(f"[train_2021] set CNN_WEIGHTS=weights\\voiceguard_cnn_2021.pt")
    print(f"             then: uvicorn backend.main:app --reload --port 8000")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train VoiceGuardCNN on combined ASVspoof 2019 + 2021 data."
    )
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=256,
                   help="256 fits comfortably on RTX 4070 Laptop")
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--quick",       action="store_true",
                   help="2-epoch sanity check on ~1 k samples")
    p.add_argument("--augment",     action="store_true",
                   help="audio + spec augmentation (strongly recommended)")
    p.add_argument("--from-scratch", action="store_true",
                   help="ignore existing weights, train from random init")
    p.add_argument("--max-la21",    type=int,   default=60_000,
                   help="max 2021 LA samples (stratified); 0=all")
    p.add_argument("--max-df21",    type=int,   default=60_000,
                   help="max 2021 DF samples (stratified); 0=all")
    p.add_argument("--max-pa21",    type=int,   default=60_000,
                   help="max 2021 PA samples (stratified); 0=all")
    args = p.parse_args()

    if args.quick:
        args.epochs = 2

    max_la = args.max_la21 or None
    max_df = args.max_df21 or None
    max_pa = args.max_pa21 or None

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        quick=args.quick,
        augment=args.augment,
        from_scratch=args.from_scratch,
        max_la21=max_la,
        max_df21=max_df,
        max_pa21=max_pa,
    )
