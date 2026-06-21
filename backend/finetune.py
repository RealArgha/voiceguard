"""
Fine-tune the trained CNN on your own mic-recorded mini dataset.

Strategy (progressive unfreezing):
  Phase 1 (warmup epochs): freeze conv blocks, train FC only at full LR.
  Phase 2 (remaining epochs): unfreeze ALL layers at 10x lower LR + cosine decay.

This preserves the pretrained spectral-artifact detectors while adapting
the decision boundary to your specific mic + room.

With augmentation + 100+ real clips + diverse AI voices, one fine-tune
should stay valid as long as you use the same mic.

USAGE
-----
    python -m backend.finetune                   # recommended defaults
    python -m backend.finetune --epochs 30       # more data -> more epochs
    python -m backend.finetune --no-augment      # skip augmentation
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm

from .model        import load_model
from .spectrogram  import audio_to_logmel, SAMPLE_RATE
from .augmentation import augment_audio, spec_augment
from .train        import equal_error_rate


REAL_DIR     = Path("data/mini/real")
FAKE_DIR     = Path("data/mini/fake")
CLIP_SAMPLES = 3 * SAMPLE_RATE


class MiniDataset(Dataset):
    def __init__(self, real_dir: Path, fake_dir: Path, augment: bool = False):
        self.items   = []
        self.augment = augment

        for f in sorted(real_dir.glob("*.wav")):
            self.items.append((f, 0))
        for f in sorted(fake_dir.glob("*.wav")):
            self.items.append((f, 1))

        if not self.items:
            raise RuntimeError(
                f"No clips found in {real_dir} or {fake_dir}.\n"
                "Run:  python record_dataset.py --record-real-session --duration 180\n"
                "      python record_dataset.py --record-fake --duration 90")

        real_count = sum(1 for _, l in self.items if l == 0)
        fake_count = sum(1 for _, l in self.items if l == 1)
        print(f"[mini] real={real_count}  fake={fake_count}  augment={augment}")

        if real_count == 0 or fake_count == 0:
            raise RuntimeError(
                f"Need BOTH real and fake clips. real={real_count} fake={fake_count}")

        self.labels = np.array([l for _, l in self.items])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        audio, _    = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if len(audio) < CLIP_SAMPLES:
            audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
        else:
            audio = audio[:CLIP_SAMPLES]

        if self.augment:
            audio = augment_audio(audio)

        mel = audio_to_logmel(audio).numpy()

        if self.augment:
            mel = spec_augment(mel)

        return (torch.from_numpy(mel).unsqueeze(0),
                torch.tensor(label, dtype=torch.float32))


def _run_val(model, loader, loss_fn, device, use_amp):
    model.eval()
    scores_list, labels_list = [], []
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast(device_type="cuda", enabled=use_amp):
                pred = model(x)
            loss_fn(pred.float(), y)
            correct += ((pred.float() >= 0.5).float() == y).sum().item()
            total   += x.size(0)
            scores_list.append(pred.float().cpu().numpy())
            labels_list.append(y.cpu().numpy())
    scores = np.concatenate(scores_list)
    labels = np.concatenate(labels_list)
    return correct / total if total else 0, equal_error_rate(scores, labels)


def finetune(epochs: int, lr: float, augment: bool, warmup: int):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    scaler  = GradScaler("cuda") if use_amp else None

    candidates = [
        "weights/voiceguard_cnn_2021.pt",
        "weights/voiceguard_cnn_aug.pt",
        "weights/voiceguard_cnn.pt",
    ]
    base = next((c for c in candidates if Path(c).exists()), None)
    if base is None:
        raise RuntimeError("No base weights found. Train on ASVspoof first.")

    print(f"[finetune] base weights : {base}")
    print(f"[finetune] device       : {device}  amp={use_amp}")
    print(f"[finetune] epochs       : {epochs}  "
          f"(warmup={warmup} FC-only, then {epochs-warmup} full)")
    print(f"[finetune] augmentation : {augment}")

    model = load_model(base, device=device)

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = MiniDataset(REAL_DIR, FAKE_DIR, augment=augment)
    n       = len(dataset)
    split   = int(n * 0.8)
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_labels = dataset.labels[train_idx]
    counts  = np.bincount(train_labels, minlength=2)
    weights = (1.0 / counts)[train_labels]
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

    bs = min(32, max(8, len(train_ds) // 4))
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=0)

    loss_fn  = nn.BCELoss()
    out_path = Path("weights/voiceguard_cnn_finetuned.pt")
    best_eer = 1.0

    # ── phase 1: FC-only warmup ───────────────────────────────────────────────
    for name, param in model.named_parameters():
        param.requires_grad = ("fc" in name)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[phase 1] FC-only  trainable={n_params:,}  lr={lr:.1e}")

    opt1   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=max(1, warmup), eta_min=lr * 0.01)

    opt2 = sched2 = None

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(epochs):

        # Switch to phase 2 after warmup
        if epoch == warmup:
            for param in model.parameters():
                param.requires_grad = True
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            lr2  = lr * 0.1
            print(f"\n[phase 2] all layers  trainable={n_params:,}  lr={lr2:.1e}")
            opt2   = torch.optim.AdamW(model.parameters(), lr=lr2,
                                       weight_decay=1e-4)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=max(1, epochs - warmup), eta_min=lr2 * 0.01)

        opt   = opt1   if epoch < warmup else opt2
        sched = sched1 if epoch < warmup else sched2

        # Train
        model.train()
        for x, y in tqdm(train_loader,
                         desc=f"epoch {epoch+1:>2} train", leave=False):
            x, y = x.to(device), y.to(device)
            if use_amp:
                with autocast(device_type="cuda"):
                    pred = model(x)
                loss = loss_fn(pred.float(), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(x)
                loss = loss_fn(pred.float(), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        sched.step()

        acc, eer = _run_val(model, val_loader, loss_fn, device, use_amp)
        print(f"epoch {epoch+1:>2}/{epochs}  val acc {acc:.3f}  EER {eer*100:.2f}%")

        if not np.isnan(eer) and eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), out_path)
            print(f"  ^ new best -> {out_path}")

    print(f"\n[finetune] done.  best EER: {best_eer*100:.2f}%")
    print(f"[finetune] weights: {out_path}")
    print(f"[finetune] start:   uvicorn backend.main:app --reload --port 8000")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=20,
                   help="total epochs (default 20)")
    p.add_argument("--warmup",     type=int,   default=3,
                   help="FC-only warmup epochs (default 3)")
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--no-augment", action="store_true",
                   help="disable audio + spec augmentation")
    args = p.parse_args()
    finetune(args.epochs, args.lr, not args.no_augment, args.warmup)
