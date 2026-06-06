"""
In whisper
Standard encoder-decoder transformer. Encoder consumes log-mel spectrograms. Decoder produces text tokens autoregressively. No vocoder, no CTC, no HMM.
Train on everything. 680,000 hours of weakly-labeled audio scraped from the internet across 97 languages. No clean academic corpus. No phoneme labels.
Multi-task single model. One decoder trained jointly on transcription, translation, voice activity detection, language ID, and timestamping via task tokens.


!Whisper pipeline: audio → mel → encoder → decoder → text
Audio at 16 kHz. Clip/pad to 30 seconds. Compute log-mel spectrogram: 80 mel bins, 10 ms stride → ~3,000 frames × 80 features. This is the "input image" that Whisper sees.
Two Conv1D layers with kernel 3 and stride 2 reduce the 3,000 frames to 1,500. Halves sequence length without adding a lot of parameters.
A 24-layer (for large) transformer encoder over 1,500 timesteps. Sinusoidal positional encoding, self-attention, GELU FFN. Produces 1,500 × 1,280 hidden states.
A 24-layer transformer decoder. It autoregressively produces tokens from a BPE vocabulary that is a superset of GPT-2's with a few audio-specific special tokens.
The decoder prompt starts with control tokens that tell the model what to do:
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>

output
Beam search (width 5) with a log-prob threshold. Timestamps are predicted every 0.02 seconds of audio when the <|notimestamps|> token is absent.

What Whisper does not do
No diarization (who is speaking). Pair with pyannote for that.
No real-time streaming natively — the 30-second window is fixed. Modern wrappers (faster-whisper, WhisperX) bolt on streaming via VAD + overlap.
(Whisper always processes 30 second chunks)
"""

import whisper
from pyannote.audio import Pipeline

audio_file = "meeting.wav"

whisper_model = whisper.load_model("base")
transcript = whisper_model.transcribe(audio_file)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

diarization = pipeline(audio_file)

# pyannote
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(
        f"{turn.start:.1f}s - {turn.end:.1f}s : {speaker}"
    )

# whisper
for seg in transcript["segments"]:
    print(seg["start"], seg["end"], seg["text"])


# combined diarization
def get_speaker(diarization, start, end):
    mid = (start + end) / 2

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= mid <= turn.end:
            return speaker

    return "Unknown"


for seg in transcript["segments"]:
    speaker = get_speaker(
        diarization,
        seg["start"],
        seg["end"]
    )

    print(f"[{speaker}] {seg['text']}")