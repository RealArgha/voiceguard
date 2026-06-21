"""
Audio → Log-Mel spectrogram.

Kept deliberately small — one function. Tune the constants below if you
change the CNN input shape.
"""

import numpy as np
import librosa
import torch

SAMPLE_RATE = 16_000   # ASVspoof 2019 LA is 16 kHz
N_MELS      = 128
N_FFT       = 512
HOP_LENGTH  = 160      # 10 ms hop at 16 kHz


def audio_to_logmel(audio: np.ndarray,
                    sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    audio: 1-D float32 numpy array in [-1, 1], mono.
    Returns a (N_MELS, T) float32 torch tensor on CPU.
    """
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # dB scale

    # Normalize to roughly [-1, 1] — helps CNN training stability.
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    return torch.from_numpy(log_mel.astype(np.float32))
