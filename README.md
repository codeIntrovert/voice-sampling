1. Silero VAD [listens if theres speech]
2. faster-whisper [voice to text]
3. SpeechBrain ECAPA [speaker identification]
4. speakers.py [unknown speaker directory, ECAPA voiceprints matched/clustered in data/speakers.json]

## Run

Open a terminal in this folder and activate the venv (it lives outside OneDrive):

```
& "$env:USERPROFILE\.venvs\voice-sampling\Scripts\Activate.ps1"
```

1. Enroll your voice (speak ~20 s into the laptop mic):
   `python enroll.py`
   Or from a recording: `python enroll.py --file myvoice.m4a`
   Run it again in other conditions (noisy room, further away) to add samples.
2. Start listening: `python listen.py`
   Green = you, cyan = known voice, yellow `+Unknown N` = new voice saved, grey `?` = unsure.
   Test with a recording instead of the mic: `python listen.py --file clip.wav`
3. Manage speakers (works while listen.py runs):
   `python manage.py list`, `rename unk_3 "Rahul"`, `merge unk_4 unk_3`, `delete unk_5`
   Hear who an Unknown is: `data/samples/unk_3.wav`

Output: `data/transcripts/YYYY-MM-DD.jsonl` (one line per utterance), `data/speakers.json`.
Settings (thresholds, whisper model, mic, language): `config.py`.
