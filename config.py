"""Every tunable setting lives here."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SPEAKERS_FILE = DATA_DIR / "speakers.json"   # every voice ever heard + its voiceprints
SAMPLES_DIR = DATA_DIR / "samples"           # one clip per speaker, so you can hear who "unk_3" is
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"   # one .jsonl file per day, one line per utterance
MODELS_DIR = ROOT / "models"

SAMPLE_RATE = 16000
CHUNK = 512        # Silero VAD takes exactly 512 samples (32 ms) per call at 16 kHz
MIC_DEVICE = None  # None = system default. List devices with: python -m sounddevice

# --- Speech detection (Silero VAD) ---
VAD_THRESHOLD = 0.5
END_SILENCE_MS = 500   # this much silence ends an utterance
PREROLL_MS = 300       # audio kept from before speech onset so the first syllable isn't clipped
MIN_SEGMENT_SEC = 0.5  # ignore blips shorter than this
MAX_SEGMENT_SEC = 15   # force-cut long monologues

# --- Who is speaking (cosine similarity between ECAPA voiceprints, -1..1) ---
MATCH_THRESHOLD = 0.45        # >= this: it's that stored speaker (you or someone known)
NEW_SPEAKER_THRESHOLD = 0.30  # <  this vs everyone: a new voice -> saved as "Unknown N"
                              # in between: "unsure", logged with a best guess, nothing saved
MIN_NEW_SPEAKER_SEC = 2.0     # clips shorter than this never create or refine a speaker
MAX_EMBEDDINGS = 30           # voiceprints kept per non-user speaker (newest win)

# --- Transcription (faster-whisper) ---
WHISPER_MODEL = "small"   # tiny / base / small / medium / large-v3: bigger = more accurate, slower
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"
LANGUAGE = "en"           # None = auto-detect each utterance (unreliable on short clips), "hi" = Hindi
