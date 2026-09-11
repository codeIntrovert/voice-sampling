"""Look after the speaker directory. listen.py picks up changes without a restart.

  python manage.py list                  every speaker + their closest other voice (spot duplicates)
  python manage.py rename unk_3 "Rahul"
  python manage.py merge unk_4 unk_3     unk_4 was really unk_3: fold it in
  python manage.py delete unk_5

Rename and merge also update past transcripts. Hear who someone is: data/samples/<id>.wav
"""
import json
import sys

import config
from speakers import Directory, centroid


def rewrite_transcripts(fn):
    for path in config.TRANSCRIPTS_DIR.glob("*.jsonl"):
        recs = [fn(json.loads(line)) for line in path.read_text("utf-8").splitlines() if line.strip()]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), "utf-8")


def cmd_list(d):
    cents = {sid: centroid(spk) for sid, spk in d.speakers.items()}
    print(f"{'id':<8} {'name':<14} {'heard':>5}  {'last seen':<19}  closest other voice")
    for sid, spk in d.speakers.items():
        others = [(float(cents[sid] @ c), o) for o, c in cents.items() if o != sid]
        close = "-"
        if others:
            score, other = max(others)
            close = f"{other} ({score:.2f})"
        print(f"{sid:<8} {spk['name']:<14} {spk['segments']:>5}  {spk['last_seen']:<19}  {close}")
    print(f"\nclosest >= {config.MATCH_THRESHOLD} usually means one person split in two -> merge them")


def cmd_rename(d, sid, name):
    d.speakers[sid]["name"] = name
    d.save()

    def fix(r):
        if r.get("speaker_id") == sid:
            r["speaker"] = name
        return r
    rewrite_transcripts(fix)


def cmd_merge(d, src, dst):
    a, b = d.speakers.pop(src), d.speakers[dst]
    b["embeddings"] += a["embeddings"]
    if not b["is_user"]:
        b["embeddings"] = b["embeddings"][-config.MAX_EMBEDDINGS:]
    b["segments"] += a["segments"]
    b["first_seen"] = min(a["first_seen"], b["first_seen"])
    b["last_seen"] = max(a["last_seen"], b["last_seen"])
    d.save()
    (config.DATA_DIR / a["sample"]).unlink(missing_ok=True)

    def fix(r):
        if r.get("speaker_id") == src:
            r.update(speaker_id=dst, speaker=b["name"], is_user=b["is_user"])
        if r.get("guess") == src:
            r["guess"] = dst
        return r
    rewrite_transcripts(fix)


def cmd_delete(d, sid):
    spk = d.speakers.pop(sid)
    d.save()
    (config.DATA_DIR / spk["sample"]).unlink(missing_ok=True)


if __name__ == "__main__":
    commands = {"list": (cmd_list, 0), "rename": (cmd_rename, 2), "merge": (cmd_merge, 2), "delete": (cmd_delete, 1)}
    if len(sys.argv) < 2 or sys.argv[1] not in commands or len(sys.argv) - 2 != commands[sys.argv[1]][1]:
        sys.exit(__doc__)
    directory = Directory()
    for sid in sys.argv[2:3] + (sys.argv[3:4] if sys.argv[1] == "merge" else []):
        if sid not in directory.speakers:
            sys.exit(f"No speaker '{sid}'. Run: python manage.py list")
    commands[sys.argv[1]][0](directory, *sys.argv[2:])
