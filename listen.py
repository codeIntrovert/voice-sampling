"""Always-on listener: Silero VAD -> who is speaking (ECAPA) -> what they said (faster-whisper).

  python listen.py                   live microphone, Ctrl+C to stop
  python listen.py --file talk.wav   run a recording through the same pipeline

Each utterance is appended to data/transcripts/YYYY-MM-DD.jsonl; new voices go to data/speakers.json.
"""
import argparse
import collections
import datetime
import json
import os
import queue
import threading

import numpy as np
import sounddevice as sd
import torch

import config
import core
from speakers import Directory

SR, CHUNK = config.SAMPLE_RATE, config.CHUNK
GREEN, CYAN, YELLOW, GREY, RESET = "\033[92m", "\033[96m", "\033[93m", "\033[90m", "\033[0m"


class Segmenter:
    """Runs 512-sample chunks through Silero VAD and hands complete utterances to on_segment."""

    def __init__(self, on_segment):
        self.vad = core.vad_model()
        self.on_segment = on_segment
        self.preroll = collections.deque(maxlen=config.PREROLL_MS * SR // 1000 // CHUNK)
        self.end_chunks = config.END_SILENCE_MS * SR // 1000 // CHUNK
        self.max_chunks = config.MAX_SEGMENT_SEC * SR // CHUNK
        self.speech = None  # chunks of the utterance in progress
        self.silent = 0     # trailing non-speech chunks in it
        self.pos = 0        # samples consumed so far

    def feed(self, chunk):
        p = self.vad(torch.from_numpy(chunk), SR).item()
        self.pos += CHUNK
        if self.speech is None:
            self.preroll.append(chunk)
            if p >= config.VAD_THRESHOLD:
                self.speech, self.silent = list(self.preroll), 0
                self.start = self.pos - len(self.speech) * CHUNK
            return
        self.speech.append(chunk)
        # lower bar to *stay* in speech than to enter it, so quiet word endings aren't cut
        self.silent = 0 if p >= config.VAD_THRESHOLD - 0.15 else self.silent + 1
        if self.silent >= self.end_chunks or len(self.speech) >= self.max_chunks:
            self.flush()

    def flush(self):
        if self.speech:
            kept = self.speech[:len(self.speech) - max(0, self.silent - 3)]  # trim trailing silence
            if len(kept) * CHUNK >= config.MIN_SEGMENT_SEC * SR:
                self.on_segment(self.start, np.concatenate(kept))
        self.speech = None
        self.preroll.clear()


class Pipeline:
    def __init__(self):
        print("Loading models (first run downloads them)...")
        core.encoder(), core.whisper()
        self.speakers = Directory()
        if "me" not in self.speakers.speakers:
            print(f"{YELLOW}No voiceprint for you yet - run enroll.py, or you'll show up as an Unknown.{RESET}")
        self.t0 = datetime.datetime.now()

    def process(self, start_sample, audio):
        text = core.transcribe(audio)
        if not text:
            return  # cough, music, noise: don't let it create speakers
        self.speakers.reload_if_changed()
        status, sid, score = self.speakers.identify(core.embed(audio), audio)

        sure = status != "unsure"
        name = self.speakers.speakers[sid]["name"] if sid else "?"
        start = self.t0 + datetime.timedelta(seconds=start_sample / SR)
        rec = {
            "time": start.isoformat(timespec="seconds"),
            "duration": round(len(audio) / SR, 1),
            "speaker_id": sid if sure else None,
            "speaker": name if sure else "?",
            "is_user": sure and self.speakers.speakers[sid]["is_user"],
            "score": round(score, 3),
            "text": text,
        }
        if not sure:
            rec["guess"] = sid
        config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.TRANSCRIPTS_DIR / f"{start:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        color = GREEN if rec["is_user"] else {"new": YELLOW, "unsure": GREY}.get(status, CYAN)
        label = {"new": f"+{name}", "unsure": f"{name}?"}.get(status, name)
        print(f"{color}[{start:%H:%M:%S}] {label:<12} {score:5.2f} | {text}{RESET}")


def run_live(pipe):
    audio_q, seg_q = queue.Queue(), queue.Queue()
    seg = Segmenter(lambda start, audio: seg_q.put((start, audio)))

    def worker():  # transcription is slower than VAD, so it gets its own thread
        while True:
            pipe.process(*seg_q.get())
            if seg_q.qsize() > 3:
                print(f"{GREY}(falling behind: {seg_q.qsize()} utterances queued){RESET}")

    threading.Thread(target=worker, daemon=True).start()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK,
                        device=config.MIC_DEVICE,
                        callback=lambda indata, *_: audio_q.put(indata[:, 0].copy())):
        pipe.t0 = datetime.datetime.now()
        mic = sd.query_devices(config.MIC_DEVICE, "input")["name"]
        print(f"Listening on '{mic}'... Ctrl+C to stop")
        try:
            while True:
                seg.feed(audio_q.get())
        except KeyboardInterrupt:
            print("Stopped.")


def run_file(pipe, path):
    audio = core.load_audio(path)
    seg = Segmenter(pipe.process)
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        seg.feed(audio[i:i + CHUNK])
    seg.flush()


if __name__ == "__main__":
    os.system("")  # enables ANSI colours in the classic Windows console
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    args = ap.parse_args()
    pipeline = Pipeline()
    run_file(pipeline, args.file) if args.file else run_live(pipeline)
