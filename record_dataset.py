"""
Record a real-vs-fake dataset for fine-tuning VoiceGuard.

We need:
  - 100+ real clips : your voice, recorded via mic
  - 100+ fake clips : multiple AI voices played through YOUR SPEAKERS via mic
                      (captures room acoustics — the exact scenario to detect)

OUTPUT
------
    data/mini/real/*.wav   (3-second clips, 16kHz mono)
    data/mini/fake/*.wav

USAGE
-----
    # Recommended: continuous session (no button-mashing)
    python record_dataset.py --record-real-session --duration 180   # 3 min -> 60 clips
    python record_dataset.py --record-fake --duration 120           # per AI voice file

    # Legacy: manual clip-by-clip
    python record_dataset.py --record-real --count 50

    # Slice AI files directly (no room acoustics — for software-channel fakes)
    python record_dataset.py --slice-fake tts1.mp3 tts2.mp3

    # Clear old fake clips before a fresh fake recording session
    python record_dataset.py --clear-fake

    # Fine-tune after recording
    python -m backend.finetune
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

SAMPLE_RATE  = 16000
CLIP_SECONDS = 3
CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE


def record_real(count: int):
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("Install sounddevice first:  pip install sounddevice")

    out_dir = Path("data/mini/real")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.wav")))

    print(f"\nRecording {count} real-voice clips ({CLIP_SECONDS}s each).")
    print("Speak naturally — pretend you're calling your bank.")
    print("Say things like: 'I need to check my account balance' or")
    print("'Can you help me with a transaction?' etc.\n")

    for i in range(count):
        idx = existing + i + 1
        out_path = out_dir / f"real_{idx:04d}.wav"
        if out_path.exists():
            print(f"  [{idx:>3}/{existing+count}] {out_path.name} already exists, skipping")
            continue

        input(f"  [{idx:>3}/{existing+count}] Press Enter to record {CLIP_SECONDS}s... ")
        audio = sd.rec(CLIP_SAMPLES, samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        # Normalise
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio / peak * 0.9

        sf.write(out_path, audio, SAMPLE_RATE)
        print(f"           saved → {out_path}")

    print(f"\nDone. {count} real clips in {out_dir}/")


def record_real_session(duration: int):
    """Record a continuous real-voice session and auto-slice into 3s clips.

    Much faster than --record-real: speak naturally for N minutes,
    get duration//3 clips automatically with no button-pressing.
    """
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("Install sounddevice first:  pip install sounddevice")

    out_dir = Path("data/mini/real")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.wav")))

    print(f"\nContinuous real-voice session: {duration}s -> ~{duration//CLIP_SECONDS} clips")
    print("Speak naturally and continuously — sentences, questions, anything.")
    print("Vary your pace, tone and volume. Pauses are fine.")
    print("Examples: read a news article aloud, describe your day, count, etc.\n")
    input("Press Enter when ready to start recording... ")

    print(f"Recording {duration}s... speak now!")
    audio = sd.rec(duration * SAMPLE_RATE, samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    print("Done. Slicing into 3s clips...")

    idx = existing
    saved = skipped = 0
    for start in range(0, len(audio) - CLIP_SAMPLES, CLIP_SAMPLES):
        chunk = audio[start:start + CLIP_SAMPLES]
        if float(np.abs(chunk).max()) < 0.02:   # skip silent gaps
            skipped += 1
            continue
        peak = float(np.abs(chunk).max())
        if peak > 0:
            chunk = chunk / peak * 0.9
        idx += 1
        sf.write(out_dir / f"real_{idx:04d}.wav", chunk, SAMPLE_RATE)
        saved += 1

    print(f"Saved {saved} real clips (skipped {skipped} silent) -> {out_dir}/")
    print(f"Total real clips: {len(list(out_dir.glob('*.wav')))}")


def record_fake(duration: int):
    """Record AI audio played through speakers, captured by mic.

    The user plays TTS/AI audio on their speakers while this records
    from the mic — capturing room acoustics, the physical replay scenario.
    """
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("Install sounddevice first:  pip install sounddevice")

    out_dir = Path("data/mini/fake")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.wav")))

    print(f"\nAbout to record {duration}s from your mic.")
    print("BEFORE pressing Enter:")
    print("  1. Open voice_preview_robot.mp3 or robot2.mp3 in a media player")
    print("  2. Set it to loop/repeat")
    print("  3. Start playing it through your SPEAKERS (not headphones)")
    print("  4. Then press Enter here to start capturing\n")
    input("Press Enter when AI audio is playing through your speakers... ")

    print(f"Recording {duration}s... (let the AI audio keep playing)")
    audio = sd.rec(duration * SAMPLE_RATE, samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    print("Done recording. Slicing into 3s clips...")

    idx = existing
    saved = 0
    for start in range(0, len(audio) - CLIP_SAMPLES, CLIP_SAMPLES):
        chunk = audio[start:start + CLIP_SAMPLES]
        if float(np.abs(chunk).max()) < 0.01:
            continue
        peak = float(np.abs(chunk).max())
        if peak > 0:
            chunk = chunk / peak * 0.9
        idx += 1
        out_path = out_dir / f"fake_{idx:04d}.wav"
        sf.write(out_path, chunk, SAMPLE_RATE)
        saved += 1

    print(f"Saved {saved} fake clips from room recording -> {out_dir}/")
    print(f"Total fake clips: {len(list(out_dir.glob('*.wav')))}")


def slice_fake(paths: list[str]):
    """
    Slice downloaded TTS files into 3-second clips at 16kHz.
    Skips chunks that are mostly silence (energy < threshold).
    """
    out_dir = Path("data/mini/fake")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.wav")))
    idx = existing

    for path in paths:
        if not Path(path).exists():
            print(f"  SKIP: {path} not found"); continue

        print(f"\nSlicing {path}...")
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        n_clips = len(audio) // CLIP_SAMPLES
        saved = 0
        for c in range(n_clips):
            chunk = audio[c * CLIP_SAMPLES:(c + 1) * CLIP_SAMPLES]

            # Skip near-silent chunks
            if float(np.abs(chunk).max()) < 0.01:
                continue

            # Normalise
            chunk = chunk / float(np.abs(chunk).max()) * 0.9
            idx += 1
            out_path = out_dir / f"fake_{idx:04d}.wav"
            sf.write(out_path, chunk, SAMPLE_RATE)
            saved += 1

        print(f"  -> {saved} clips from {path}")

    total = len(list(out_dir.glob("*.wav")))
    print(f"\nDone. {total} fake clips total in {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-real",         action="store_true",
                        help="record clips one at a time (press Enter each)")
    parser.add_argument("--record-real-session", action="store_true",
                        help="record continuously for --duration seconds, auto-slice")
    parser.add_argument("--record-fake",         action="store_true",
                        help="record AI audio played through speakers via mic")
    parser.add_argument("--slice-fake",          nargs="+", metavar="FILE",
                        help="slice downloaded TTS files into fake clips")
    parser.add_argument("--clear-fake",          action="store_true",
                        help="delete all existing fake clips before recording")
    parser.add_argument("--clear-real",          action="store_true",
                        help="delete all existing real clips before recording")
    parser.add_argument("--count",    type=int, default=50,
                        help="clips for --record-real (default 50)")
    parser.add_argument("--duration", type=int, default=120,
                        help="seconds for --record-real-session / --record-fake (default 120)")
    args = parser.parse_args()

    if not any([args.record_real, args.record_real_session,
                args.record_fake, args.slice_fake]):
        parser.print_help()
        sys.exit(1)

    if args.clear_real:
        import shutil
        real_dir = Path("data/mini/real")
        if real_dir.exists():
            shutil.rmtree(real_dir)
            print(f"Cleared {real_dir}")

    if args.clear_fake:
        import shutil
        fake_dir = Path("data/mini/fake")
        if fake_dir.exists():
            shutil.rmtree(fake_dir)
            print(f"Cleared {fake_dir}")

    if args.record_real:
        record_real(args.count)

    if args.record_real_session:
        record_real_session(args.duration)

    if args.record_fake:
        record_fake(args.duration)

    if args.slice_fake:
        slice_fake(args.slice_fake)


if __name__ == "__main__":
    main()
