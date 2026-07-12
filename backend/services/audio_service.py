"""
backend/services/audio_service.py
==================================
Audio transcription pipeline: Google STT (primary) → Whisper (fallback).
"""

import asyncio
import os

import speech_recognition as sr
from pydub import AudioSegment
from fastapi import HTTPException

from backend.config import SUPPORTED_LANGUAGES, logger


async def transcribe_audio(
    audio_path: str,
    recording_language: str,
    submitted_text: str,
    whisper_model,
) -> tuple:
    """Transcribe the uploaded audio file.

    Returns (transcribed_text, detected_language, audio_duration_seconds, error_stage, error_message).
    Uses Google STT for kn/hi/en, falls back to Whisper on failure.
    """
    transcribed_text = submitted_text
    detected_language = recording_language
    audio_duration = None
    error_stage = None
    error_message = None

    if recording_language in ["kn", "hi", "en"]:
        logger.info("Using Google STT for highly accurate transcription.")
        try:
            lang_code_map = {"kn": "kn-IN", "hi": "hi-IN", "en": "en-IN"}
            google_lang = lang_code_map.get(recording_language, "en-IN")

            # Convert webm to wav for SpeechRecognition
            wav_path = audio_path + ".wav"
            audio_segment = AudioSegment.from_file(audio_path)
            audio_duration = audio_segment.duration_seconds
            audio_segment.export(wav_path, format="wav")

            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)

            audio_text = await asyncio.to_thread(
                recognizer.recognize_google,
                audio_data,
                language=google_lang,
            )

            transcribed_text = f"{audio_text} {submitted_text}".strip() if submitted_text else audio_text
            logger.info("Google STT Transcribed: %s | Language: %s", audio_text, detected_language)

            if os.path.exists(wav_path):
                os.remove(wav_path)

        except sr.UnknownValueError:
            error_stage = "transcription"
            error_message = f"Google STT could not understand audio for {audio_path}"
            logger.warning(error_message)
            raise HTTPException(
                status_code=400,
                detail="No clear speech detected in audio. Please record again and speak closer to the microphone.",
            )
        except Exception as e:
            logger.error("Google STT failed: %s. Falling back to Whisper...", e)
            # Fallback to Whisper
            try:
                result = await asyncio.to_thread(
                    whisper_model.transcribe,
                    audio_path,
                    task="transcribe",
                    fp16=False,
                    condition_on_previous_text=False,
                )
                audio_text = (result.get("text") or "").strip()
                whisper_detected_language = (result.get("language") or recording_language).strip().lower()
                if whisper_detected_language in SUPPORTED_LANGUAGES:
                    detected_language = whisper_detected_language

                if not audio_text and not submitted_text:
                    error_stage = "transcription"
                    error_message = f"No speech detected in uploaded audio: {audio_path}"
                    logger.warning(error_message)
                    raise HTTPException(
                        status_code=400,
                        detail="No clear speech detected in audio. Please record again and speak closer to the microphone.",
                    )
                transcribed_text = f"{audio_text} {submitted_text}".strip() if submitted_text else audio_text
                logger.info("Whisper Transcribed: %s | Language: %s", audio_text, detected_language)

                if audio_duration is None:
                    try:
                        _seg = AudioSegment.from_file(audio_path)
                        audio_duration = _seg.duration_seconds
                    except Exception:
                        pass
            except HTTPException:
                raise
            except Exception as ex:
                error_stage = "transcription"
                error_message = str(ex)
                raise HTTPException(status_code=500, detail="Audio transcription failed. Please try again.")
    else:
        # Whisper-only path for other/unsupported language codes
        try:
            try:
                _seg = AudioSegment.from_file(audio_path)
                audio_duration = _seg.duration_seconds
            except Exception:
                pass

            result = await asyncio.to_thread(
                whisper_model.transcribe,
                audio_path,
                task="transcribe",
                fp16=False,
                condition_on_previous_text=False,
            )
            audio_text = (result.get("text") or "").strip()
            whisper_detected_language = (result.get("language") or recording_language).strip().lower()
            if whisper_detected_language in SUPPORTED_LANGUAGES:
                detected_language = whisper_detected_language

            if not audio_text and not submitted_text:
                error_stage = "transcription"
                error_message = f"No speech detected in uploaded audio: {audio_path}"
                logger.warning(error_message)
                raise HTTPException(
                    status_code=400,
                    detail="No clear speech detected in audio. Please record again and speak closer to the microphone.",
                )
            transcribed_text = f"{audio_text} {submitted_text}".strip() if submitted_text else audio_text
            logger.info("Transcribed: %s | Language: %s", audio_text, detected_language)
        except HTTPException:
            raise
        except Exception as e:
            error_stage = "transcription"
            error_message = str(e)
            logger.error("Whisper transcription failed for %s: %s", audio_path, e)
            if "WinError 2" in str(e) or "ffmpeg" in str(e).lower():
                logger.warning("FFmpeg not found. Cannot transcribe audio without FFmpeg.")
                raise HTTPException(
                    status_code=500,
                    detail="Audio transcription failed: FFmpeg is not installed or not found in PATH. Please install FFmpeg and retry.",
                )
            raise HTTPException(status_code=500, detail="Audio transcription failed. Please try again.")

    return transcribed_text, detected_language, audio_duration, error_stage, error_message
