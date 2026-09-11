"""Speaker directory: every voice heard so far, persisted in data/speakers.json.

{"next_unknown": 4,
 "speakers": {"me":    {"name": "Me", "is_user": true, "embeddings": [[...192 floats], ...], ...},
              "unk_3": {"name": "Unknown 3", "is_user": false, ...}}}
"""
import datetime
import json
import os
import re
import wave

import numpy as np

import config


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def centroid(spk):
    c = np.mean(spk["embeddings"], axis=0)
    return c / np.linalg.norm(c)


def _vec(e):
    return [round(float(x), 4) for x in e]


def save_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(config.SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


class Directory:
    def __init__(self, path=config.SPEAKERS_FILE):
        self.path = path
        self.load()

    def load(self):
        data = json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}
        self.next_unknown = data.get("next_unknown", 1)
        self.speakers = data.get("speakers", {})
        self._mtime = self._stat()

    def _stat(self):
        return self.path.stat().st_mtime if self.path.exists() else 0

    def reload_if_changed(self):
        """Pick up renames/merges made with manage.py while listen.py is running."""
        if self._stat() != self._mtime:
            self.load()

    def save(self):
        text = json.dumps({"next_unknown": self.next_unknown, "speakers": self.speakers},
                          indent=2, ensure_ascii=False)
        # one voiceprint per line instead of 192 lines of numbers
        text = re.sub(r"\[\s*([-0-9.eE,\s]+?)\s*\]", lambda m: "[" + re.sub(r"\s+", "", m[1]) + "]", text)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(text, "utf-8")
        os.replace(tmp, self.path)  # atomic, so Ctrl+C can't leave a half-written file
        self._mtime = self._stat()

    def _new(self, sid, name, audio, is_user=False):
        sample = f"samples/{sid}.wav"
        save_wav(config.DATA_DIR / sample, audio)
        self.speakers[sid] = {"name": name, "is_user": is_user, "first_seen": now(), "last_seen": now(),
                              "segments": 0, "sample": sample, "embeddings": []}
        return self.speakers[sid]

    def enroll_user(self, embs, audio):
        me = self.speakers.get("me") or self._new("me", "Me", audio, is_user=True)
        me["embeddings"] += [_vec(e) for e in embs]
        self.save()

    def identify(self, emb, audio):
        """Decide who is speaking. Returns (status, speaker_id, score):
        "match"  - a stored voice; for non-user speakers the clip also refines their voiceprint
        "new"    - unlike everyone stored, saved as a new "Unknown N"
        "unsure" - grey zone or clip too short to trust; speaker_id is only the best guess
        """
        best, score = None, -1.0
        for sid, spk in self.speakers.items():
            s = float(centroid(spk) @ emb)
            if s > score:
                best, score = sid, s
        long_enough = len(audio) / config.SAMPLE_RATE >= config.MIN_NEW_SPEAKER_SEC

        if best and score >= config.MATCH_THRESHOLD:
            spk = self.speakers[best]
            spk["segments"] += 1
            spk["last_seen"] = now()
            if not spk["is_user"] and long_enough:  # your print only changes via enroll.py
                spk["embeddings"] = (spk["embeddings"] + [_vec(emb)])[-config.MAX_EMBEDDINGS:]
            self.save()
            return "match", best, score

        if score < config.NEW_SPEAKER_THRESHOLD and long_enough:
            n = self.next_unknown
            self.next_unknown += 1
            spk = self._new(f"unk_{n}", f"Unknown {n}", audio)
            spk["segments"] = 1
            spk["embeddings"].append(_vec(emb))
            self.save()
            return "new", f"unk_{n}", score

        return "unsure", best, score
