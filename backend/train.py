"""
Train VoiceGuardCNN on ASVspoof 2019 LA partition.

NEW IN THIS VERSION
-------------------
  * --augment      : apply audio + spec augmentation. Closes the domain
                     gap from clean studio audio → consumer mic capture.
                     Saves to weights/voiceguard_cnn_aug.pt (separate
                     file so your clean model isn't overwritten).
  * Audio cache    : when --augment is on, we cache raw 3s audio (not
                     mel) so augmentations apply each epoch.

USAGE
-----
    export ASVSPOOF_ROOT=~/datasets/LA
    python -m backend.train --augment --epochs 10 --batch-size 128
    python -m backend.train --quick --augment           # 3-min sanity
"""

import argparse
import os
import signal
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .model        import VoiceGuardCNN
from .spectrogram  import audio_to_logmel, SAMPLE_RATE
from .augmentation import augment_audio, spec_augment


# ---------- paths ---------------------------------------------------------
ROOT       = Path(os.environ.get("ASVSPOOF_ROOT", "data/LA"))
PROTO_DIR  = ROOT / "ASVspoof2019_LA_cm_protocols"
TRAIN_FLAC = ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_FLAC   = ROOT / "ASVspoof2019_LA_dev"   / "flac"
TRAIN_PROTO = PROTO_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTO   = PROTO_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"

CLIP_SECONDS = 3
CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE


# ---------- datasets ------------------------------------------------------
class ASVspoofDataset(Dataset):
    """
    Two cache modes:
      mel_cache=True  : store log-mel .npy (fast, no augmentation possible)
      mel_cache=False : store raw audio .npy (slower per epoch, augment OK)

    The cache_dir differs so the two modes don't fight each other.
    """

    def __init__(self, proto: Path, flac_dir: Path, split: str,
                 limit: int | None = None,
                 augment: bool = False,
                 mel_cache: bool = True):
        self.flac_dir   = flac_dir
        self.augment    = augment
        self.mel_cache  = mel_cache
        cache_root      = Path("cache" if mel_cache else "cache_audio")
        self.cache_dir  = cache_root / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.items = []
        with open(proto) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                file_id, label = parts[1], parts[4]
                self.items.append((file_id, 0 if label == "bonafide" else 1))

        if limit:
            import random
            random.Random(42).shuffle(self.items)
            self.items = self.items[:limit]

        self.labels = np.array([lbl for _, lbl in self.items])
        counts = np.bincount(self.labels, minlength=2)
        if counts.min() == 0:
            raise RuntimeError(
                f"[{split}] only one class: bonafide={counts[0]} spoof={counts[1]}")
        print(f"[{split}] bonafide={counts[0]}  spoof={counts[1]}  "
              f"(ratio {counts[0]/(counts[0]+counts[1]):.1%} real)  "
              f"{'+ augmentation' if augment else ''}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        file_id, label = self.items[idx]
        cache_path = self.cache_dir / f"{file_id}.npy"

        if cache_path.exists():
            cached = np.load(cache_path)
        else:
            audio, _ = librosa.load(self.flac_dir / f"{file_id}.flac",
                                    sr=SAMPLE_RATE, mono=True)
            if len(audio) < CLIP_SAMPLES:
                audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
            else:
                audio = audio[:CLIP_SAMPLES]

            if self.mel_cache:
                cached = audio_to_logmel(audio).numpy()
            else:
                cached = audio.astype(np.float32)
            np.save(cache_path, cached)

        if self.mel_cache:
            mel = cached
            if self.augment:
                mel = spec_augment(mel)        # mel-domain aug only
        else:
            audio = cached
            if self.augment:
                audio = augment_audio(audio)
            mel = audio_to_logmel(audio).numpy()
            if self.augment:
                mel = spec_augment(mel)

        return (torch.from_numpy(mel).unsqueeze(0),
                torch.tensor(label, dtype=torch.float32))


# ---------- metrics --------------------------------------------------------
def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold-independent error metric. NaN if one class missing."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thresholds = np.linspace(0, 1, 1001)
    far = np.array([(neg >= t).mean() for t in thresholds])
    frr = np.array([(pos <  t).mean() for t in thresholds])
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2)


# ---------- epoch loop ----------------------------------------------------
def _run_epoch(model, loader, loss_fn, opt, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_scores, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc="train" if train else "val", leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total_loss    += loss.item() * x.size(0)
            total_correct += ((pred >= 0.5).float() == y).sum().item()
            total_n       += x.size(0)
            all_scores.append(pred.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return {"loss":     total_loss / total_n,
            "accuracy": total_correct / total_n,
            "eer":      equal_error_rate(scores, labels)}


# ---------- main ----------------------------------------------------------
def train(epochs, batch_size, lr, quick, augment):
    if not ROOT.exists():
        sys.exit(f"[train] ASVSPOOF_ROOT not found: {ROOT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {device}   augment: {augment}")

    limit = 1000 if quick else None
    train_ds = ASVspoofDataset(TRAIN_PROTO, TRAIN_FLAC, "train", limit,
                               augment=augment, mel_cache=not augment)
    # Dev set is NEVER augmented — we need true metrics.
    dev_ds   = ASVspoofDataset(DEV_PROTO, DEV_FLAC, "dev",
                               limit // 4 if quick else None,
                               augment=False, mel_cache=True)
    print(f"[train] train={len(train_ds)}  dev={len(dev_ds)}")

    # Balance the training batches.
    class_counts   = np.bincount(train_ds.labels)
    sample_weights = (1.0 / class_counts)[train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds),
                                    replacement=True)

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=2, pin_memory=pin)
    dev_loader   = DataLoader(dev_ds, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=pin)

    model   = VoiceGuardCNN().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    out_name = "voiceguard_cnn_aug.pt" if augment else "voiceguard_cnn.pt"
    out_path = Path("weights") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train] best checkpoints → {out_path}")

    best_eer = 1.0

    def _save_on_exit(*_):
        torch.save(model.state_dict(), out_path.with_suffix(".interrupt.pt"))
        print(f"\n[train] interrupted — saved {out_path}.interrupt.pt")
        sys.exit(0)
    signal.signal(signal.SIGINT, _save_on_exit)

    for epoch in range(epochs):
        tr = _run_epoch(model, train_loader, loss_fn, opt, device, train=True)
        va = _run_epoch(model, dev_loader,   loss_fn, opt, device, train=False)
        print(f"epoch {epoch+1:>2}/{epochs}  "
              f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.3f}  |  "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.3f} "
              f"EER {va['eer']*100:.2f}%")
        if not np.isnan(va["eer"]) and va["eer"] < best_eer:
            best_eer = va["eer"]
            torch.save(model.state_dict(), out_path)
            print(f"[train] ↑ new best (EER {best_eer*100:.2f}%) → {out_path}")

    print(f"[train] done. best val EER: {best_eer*100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=8)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--quick",      action="store_true")
    p.add_argument("--augment",    action="store_true",
                   help="audio + spec augmentation (saves to *_aug.pt)")
    args = p.parse_args()
    if args.quick: args.epochs = 2
    train(args.epochs, args.batch_size, args.lr, args.quick, args.augment)
