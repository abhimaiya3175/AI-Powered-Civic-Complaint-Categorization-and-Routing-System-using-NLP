# pipeline/transcriber.py
"""
Wraps OpenAI Whisper.
Accepts a numpy float32 array so no temp files are needed.
"""

import numpy as np
import whisper


class Transcriber:
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16_000) -> dict:
        """
        Transcribe and translate audio to English.

        Parameters
        ----------
        audio_array : np.ndarray   float32, any sample rate
        sample_rate : int          original sample rate (Whisper resamples internally)

        Returns
        -------
        {
            "english_text"  : str,
            "detected_lang" : str,
            "no_speech_prob": float   (0–1, higher = probably silence)
        }

        Raises
        ------
        TranscriptionError on Whisper failure.
        """
        try:
            # Whisper needs 16 kHz mono float32
            audio = whisper.pad_or_trim(audio_array)
            mel   = whisper.log_mel_spectrogram(audio).to(self.model.device)

            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)

            result = self.model.transcribe(
                audio_array,
                language=None,      # auto-detect
                task="translate",   # always output English
                fp16=False,
            )

            # Average no-speech probability across all segments
            segs = result.get("segments", [])
            no_speech_prob = (
                sum(s.get("no_speech_prob", 0) for s in segs) / len(segs)
                if segs else 1.0
            )

            return {
                "english_text":   result["text"].strip(),
                "detected_lang":  detected_lang,
                "no_speech_prob": round(no_speech_prob, 4),
            }

        except Exception as exc:
            raise TranscriptionError(str(exc)) from exc


class TranscriptionError(RuntimeError):
    pass