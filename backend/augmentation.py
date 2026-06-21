"""
Audio augmentation for closing the domain gap.

The CNN trained on clean ASVspoof studio audio doesn't generalize to
laptop/phone mic capture. By augmenting training audio with synthetic
mic-distortion effects, we teach the model to be invariant to those
distortions while still picking up the real/fake spectral artifacts.

Each augmentation has an independent probability. The combinations are
realistic — e.g. a phone call gets codec + low-pass + noise.
"""

import numpy as np
import librosa
from scipy.signal import butter, lfilter


def augment_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Apply random distortions matching real-world capture conditions.
    Input/output: float32 mono in roughly [-1, 1].
    """
    out = audio.copy()

    # 1. Random gain (60%) — devices have wildly different mic levels.
    if np.random.random() < 0.6:
        out = out * np.random.uniform(0.4, 1.4)

    # 2. Additive Gaussian noise (50%) — room noise + mic self-noise.
    #    SNR between 10 dB (very noisy) and 30 dB (clean).
    if np.random.random() < 0.5:
        snr_db = np.random.uniform(10, 30)
        sig_power = float(np.mean(out ** 2)) + 1e-10
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(out)).astype(np.float32) * np.sqrt(noise_power)
        out = out + noise

    # 3. Low-pass filter (35%) — laptop/earpod mics roll off above ~7 kHz.
    if np.random.random() < 0.35:
        cutoff = np.random.uniform(3500, 7500)
        b, a = butter(N=4, Wn=cutoff / (sr / 2), btype="low")
        out = lfilter(b, a, out).astype(np.float32)

    # 4. Phone-codec simulation (40%) — downsample + upsample roundtrip.
    #    G.711 (8 kHz) and narrow-band codecs destroy high frequencies.
    if np.random.random() < 0.4:
        target_sr = int(np.random.choice([8000, 11025]))
        out = librosa.resample(out, orig_sr=sr, target_sr=target_sr)
        out = librosa.resample(out, orig_sr=target_sr, target_sr=sr)

    # 5. Re-normalize so peak amplitude stays sane.
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out = out / peak

    return out.astype(np.float32)


def spec_augment(mel: np.ndarray,
                 freq_mask_max: int = 20,
                 time_mask_max: int = 30,
                 n_masks: int = 2) -> np.ndarray:
    """
    SpecAugment-style masking on the log-mel spectrogram (Park et al. 2019).
    Applied AFTER mel extraction. Helps the CNN ignore narrow corruption
    bands and forces it to learn distributed features.
    """
    out = mel.copy()
    n_mels, n_frames = out.shape
    fill = float(out.mean())

    for _ in range(n_masks):
        # Frequency mask
        f = int(np.random.randint(0, freq_mask_max + 1))
        if f > 0 and n_mels - f > 0:
            f0 = int(np.random.randint(0, n_mels - f))
            out[f0:f0 + f, :] = fill

        # Time mask
        t = int(np.random.randint(0, time_mask_max + 1))
        if t > 0 and n_frames - t > 0:
            t0 = int(np.random.randint(0, n_frames - t))
            out[:, t0:t0 + t] = fill

    return out
