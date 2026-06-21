"""
VoiceGuard CNN — Real vs Synthetic voice classifier.

Architecture matches the description in the deck:
    Conv2D -> BatchNorm -> MaxPool  (x3 blocks)
    -> Flatten -> FC -> Sigmoid

Input : Log-Mel spectrogram tensor, shape (B, 1, 128, T)
Output: scalar in [0, 1] — probability that the clip is SYNTHETIC.

Train this on ASVspoof 2019 LA partition. Until trained weights are
loaded, the model returns garbage — see `load_model()` below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class VoiceGuardCNN(nn.Module):
    def __init__(self, n_mels: int = 128):
        super().__init__()

        # Three conv blocks. Each halves the spatial dims via MaxPool.
        self.block1 = self._conv_block(in_ch=1,  out_ch=32)
        self.block2 = self._conv_block(in_ch=32, out_ch=64)
        self.block3 = self._conv_block(in_ch=64, out_ch=128)

        # Global average pool collapses the time axis so the model
        # accepts variable-length clips. Avoids a brittle fixed FC dim.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(128, 1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)        # (B, 128)
        x = self.fc(x).squeeze(-1)        # (B,)
        return torch.sigmoid(x)           # P(synthetic)


def load_model(weights_path: str | None = None,
               device: str = "cpu") -> VoiceGuardCNN:
    """
    Loads the CNN. If `weights_path` exists, loads trained weights.
    Otherwise returns a randomly initialized model — useful for plumbing
    the pipeline end-to-end before training is done.
    """
    model = VoiceGuardCNN().to(device)

    if weights_path and Path(weights_path).is_file():
        # weights_only=True silences the PyTorch security warning and is
        # safe for state_dicts (just tensors, no pickle code).
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[model] loaded weights from {weights_path}")
    else:
        print("[model] WARNING: no weights found — running UNTRAINED. "
              "Scores will be meaningless until you train on ASVspoof.")

    model.eval()
    return model


@torch.no_grad()
def predict_synthetic_prob(model: VoiceGuardCNN,
                           mel_spec: torch.Tensor) -> float:
    """
    mel_spec: (n_mels, T) float tensor — output of librosa log-mel.
    Returns probability that the clip is AI-synthesized.
    """
    # Add batch and channel dims: (1, 1, n_mels, T)
    x = mel_spec.unsqueeze(0).unsqueeze(0)
    return model(x).item()
