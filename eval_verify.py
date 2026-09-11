"""Standard "me vs not me" benchmark: speaker verification on VoxCeleb1-O (cleaned).
37,611 official trials of "are these two clips the same person?" over 40 speakers the model never saw.

  python eval_verify.py               full run (first time ~1.5 h on CPU to embed 4.7k clips, then cached)
  python eval_verify.py --limit 2000  quick run on the first N trials

Two systems are scored on the same trials:
  standard  whole-clip embedding on both sides - the protocol papers report
  ours      enrollment side prepared exactly like enroll.py (VAD speech, 3 s windows, centroid),
            test side a whole utterance like listen.py, plus the MATCH/NEW thresholds in config.py

Dataset: ~/datasets/voxceleb1/{veri_test2.txt, wav/idXXXXX/...}
"""
import argparse
import time
from pathlib import Path

import numpy as np

import config
import core
from enroll import voiceprints

SR = config.SAMPLE_RATE


def embed_all(files, root, cache_path):
    """{clip: (whole-clip embedding, enroll.py-style embedding)}, cached so reruns are instant."""
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path)
        cache = {n: (s, o) for n, s, o in zip(z["names"], z["standard"], z["ours"])}

    def save():
        names = list(cache)
        np.savez(cache_path, names=np.array(names), standard=np.array([cache[n][0] for n in names]),
                 ours=np.array([cache[n][1] for n in names]))

    todo = [f for f in files if f not in cache]
    start = time.time()
    for k, f in enumerate(todo, 1):
        audio = core.load_audio(root / f)
        speech = core.speech_only(audio)
        if len(speech) < SR:  # VAD found almost nothing: use the raw clip
            speech = audio
        c = voiceprints(speech).mean(axis=0)
        cache[f] = (core.embed(audio), c / np.linalg.norm(c))
        if k % 100 == 0 or k == len(todo):
            left = (time.time() - start) / k * (len(todo) - k) / 60
            print(f"  embedded {k}/{len(todo)} clips, ~{left:.0f} min left", flush=True)
        if k % 500 == 0 or k == len(todo):
            save()  # an interrupted run resumes from here
    return cache


def metrics(scores, labels, p_target=0.01):
    tgt, non = np.sort(scores[labels]), np.sort(scores[~labels])
    thr = np.sort(scores)
    frr = np.searchsorted(tgt, thr) / len(tgt)      # same person, scored below threshold (missed you)
    far = 1 - np.searchsorted(non, thr) / len(non)  # different person, scored at/above (accepted as you)
    i = np.argmin(np.abs(frr - far))
    min_dcf = ((p_target * frr + (1 - p_target) * far) / p_target).min()
    return {"eer": (frr[i] + far[i]) / 2, "eer_thr": thr[i], "min_dcf": min_dcf,
            "thr_far1": thr[np.argmax(far <= 0.01)], "thr_far01": thr[np.argmax(far <= 0.001)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "datasets" / "voxceleb1")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    trials = [line.split() for line in (args.data / "veri_test2.txt").read_text().splitlines() if line.strip()]
    trials = trials[:args.limit]
    root = args.data / "wav" if (args.data / "wav").is_dir() else args.data
    files = sorted({f for _, a, b in trials for f in (a, b)})
    print(f"VoxCeleb1-O: {len(trials)} trials over {len(files)} clips")
    emb = embed_all(files, root, args.data / "embeddings.npz")

    labels = np.array([t[0] == "1" for t in trials])
    print(f"\n{'system':<10} {'EER':>6}  {'minDCF':>6}  {'EER thr':>7}  {'thr @1% FA':>10}  {'thr @0.1% FA':>12}")
    for i, name in enumerate(["standard", "ours"]):
        scores = np.array([float(emb[a][i] @ emb[b][0]) for _, a, b in trials])
        m = metrics(scores, labels)
        print(f"{name:<10} {m['eer']:6.2%}  {m['min_dcf']:6.3f}  {m['eer_thr']:7.2f}  {m['thr_far1']:10.2f}  {m['thr_far01']:12.2f}")

    # what identify() in speakers.py would actually decide ("ours" scores from the loop above)
    match, new = config.MATCH_THRESHOLD, config.NEW_SPEAKER_THRESHOLD
    print(f"\nDecisions with config.py thresholds (match >= {match}, new voice < {new}, unsure between):")
    for label, mask in [("same person (should be ME)", labels), ("different person (NOT me)", ~labels)]:
        s = scores[mask]
        print(f"  {label:<28} me {np.mean(s >= match):7.2%} | unsure {np.mean((s >= new) & (s < match)):7.2%}"
              f" | not me {np.mean(s < new):7.2%}")
    print("\nEER = error rate where missed-you equals accepted-as-you; minDCF at p_target=0.01. Lower is better.")


if __name__ == "__main__":
    main()
