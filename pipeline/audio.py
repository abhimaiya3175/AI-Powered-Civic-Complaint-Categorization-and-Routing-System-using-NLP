# pipeline/audio.py
"""
Converts a HuggingFace audio sample to a numpy float32 array.
Handles both decoded (array) and undecoded (file path) samples.
Uses soundfile for reading — no torchcodec dependency.
"""

import io
import numpy as np
import soundfile as sf


def sample_to_numpy(sample: dict) -> tuple[np.ndarray, int]:
    """
    Extract the raw float32 array and sampling rate from a dataset sample.

    Handles two formats:
      - Decoded: sample["audio"] = {"array": ..., "sampling_rate": ...}
      - Undecoded: sample["audio"] = {"path": "/path/to/file.wav", "bytes": b"..."}

    Returns
    -------
    audio_array : np.ndarray  (float32, shape [N])
    sample_rate : int
    """
    audio = sample["audio"]

    # Already decoded by datasets
    if "array" in audio and audio["array"] is not None:
        return np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"]

    # Undecoded — read from bytes or path using soundfile
    if audio.get("bytes"):
        buf = io.BytesIO(audio["bytes"])
        data, sr = sf.read(buf, dtype="float32")
        return data, sr

    if audio.get("path"):
        data, sr = sf.read(audio["path"], dtype="float32")
        return data, sr

    raise ValueError(f"Cannot extract audio from sample: {list(audio.keys())}")


def sample_to_wav_bytes(sample: dict) -> bytes:
    """
    Encode audio to WAV bytes in memory (used if a tool needs a file-like object).
    Whisper can also accept the numpy array directly – prefer sample_to_numpy.
    """
    arr, sr = sample_to_numpy(sample)
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV")
    buf.seek(0)
    return buf.read()