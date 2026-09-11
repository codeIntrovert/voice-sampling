"""Teach the system your voice.

  python enroll.py                  record 20 s from the mic
  python enroll.py --seconds 30
  python enroll.py --file me.m4a    use an existing recording (any audio format)

Run it again in other conditions (noisy room, mic further away): samples accumulate.
"""
import argparse
import sys

import numpy as np
import sounddevice as sd

import config
import core
from speakers import Directory, centroid

SR = config.SAMPLE_RATE
WINDOW, HOP = 3 * SR, 3 * SR // 2  # one voiceprint per 3 s of speech, 50% overlap


def record(seconds):
    input(f"Press Enter, then talk naturally for {seconds} s (read something aloud, describe your day)...")
    print("Recording...")
    audio = sd.rec(seconds * SR, samplerate=SR, channels=1, dtype="float32", device=config.MIC_DEVICE)
    sd.wait()
    print("Done.")
    return audio[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--file")
    args = ap.parse_args()

    audio = core.load_audio(args.file) if args.file else record(args.seconds)
    if np.abs(audio).max() < 0.01:
        sys.exit("Recording is (almost) silent - check your mic, or MIC_DEVICE in config.py")

    speech = core.speech_only(audio)
    print(f"{len(speech) / SR:.1f} s of speech found")
    if len(speech) < 5 * SR:
        sys.exit("Need at least 5 s of speech - try again.")

    embs = np.array([core.embed(speech[i:i + WINDOW]) for i in range(0, len(speech) - WINDOW + 1, HOP)])
    n = len(embs)
    consistency = ((embs @ embs.T).sum() - n) / (n * (n - 1))
    print(f"{n} voiceprints, self-similarity {consistency:.2f} (one clean voice is usually > 0.6)")

    d = Directory()
    d.enroll_user(embs, speech)
    me = d.speakers["me"]
    print(f"Saved -> {config.SPEAKERS_FILE} ({len(me['embeddings'])} voiceprints for 'me')")

    # If listen.py ran before you enrolled, you may already exist as an Unknown
    for sid, spk in d.speakers.items():
        score = float(centroid(me) @ centroid(spk))
        if sid != "me" and score >= config.MATCH_THRESHOLD:
            print(f"  {sid} ({spk['name']}) sounds like you ({score:.2f}) -> python manage.py merge {sid} me")


if __name__ == "__main__":
    main()
