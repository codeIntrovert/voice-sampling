"""Models shared by enroll.py and listen.py. Each loads once, on first use."""
import os
from functools import cache

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet model downloads stall on this network; plain HTTP works
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch

import config


@cache
def vad_model():
    from silero_vad import load_silero_vad
    return load_silero_vad()


@cache
def encoder():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(config.MODELS_DIR / "spkrec-ecapa-voxceleb"),
        local_strategy=LocalStrategy.COPY,  # default symlinks need admin rights on Windows
    )


@cache
def whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                        compute_type=config.WHISPER_COMPUTE)


def embed(audio):
    """Voiceprint of a float32 16 kHz mono clip: L2-normalised 192-dim vector."""
    with torch.no_grad():
        e = encoder().encode_batch(torch.from_numpy(audio).unsqueeze(0)).squeeze().numpy()
    return e / np.linalg.norm(e)


def transcribe(audio):
    segments, _ = whisper().transcribe(audio, language=config.LANGUAGE,
                                       condition_on_previous_text=False)
    # dropping low-confidence segments filters Whisper's classic noise hallucinations ("Thank you.")
    return " ".join(s.text.strip() for s in segments
                    if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0).strip()


def load_audio(path):
    """Any audio/video file -> float32 16 kHz mono (decoded by PyAV, bundled with faster-whisper)."""
    from faster_whisper import decode_audio
    return decode_audio(str(path), sampling_rate=config.SAMPLE_RATE)
