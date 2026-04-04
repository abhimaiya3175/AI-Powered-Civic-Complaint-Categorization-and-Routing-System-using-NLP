# pipeline/pipeline.py
"""
Main orchestrator – wires audio → transcriber → classifier → extractor.
All results are stored in a list of dicts and returned for export.
"""

import datasets
from datasets import load_dataset

from .audio       import sample_to_numpy
from .transcriber import Transcriber, TranscriptionError
from .classifier  import BBMPClassifier, ClassificationError
from .extractor   import LocationExtractor
from . import config


def build_pipeline() -> tuple:
    """Load all models once; return (transcriber, classifier, extractor)."""
    print("⏳ Loading models…")
    transcriber = Transcriber(config.WHISPER_MODEL_SIZE)
    classifier  = BBMPClassifier(config.MODEL_PATH)
    extractor   = LocationExtractor()
    print("✅ Models ready (Whisper + 98.5% BBMP + spaCy)\n")
    return transcriber, classifier, extractor


def load_data() -> object:
    print("⏳ Loading Kannada dataset…")
    dataset = load_dataset(
        config.DATASET_NAME,
        split="train",
        cache_dir=config.CACHE_DIR,
        trust_remote_code=True,
    )
    # Disable default audio decoding (avoids torchcodec requirement)
    dataset = dataset.cast_column("audio", datasets.Audio(decode=False))
    print(f"✅ Loaded {len(dataset)} samples\n")
    return dataset


def run(n_samples: int | None = None) -> list[dict]:
    """
    Run the full pipeline on the first `n_samples` samples.

    Returns
    -------
    List of result dicts – one per sample.
    """
    transcriber, classifier, extractor = build_pipeline()
    dataset = load_data()
    n       = n_samples or config.NUM_SAMPLES

    results = []
    print(f"Running pipeline on {n} samples…")
    print("=" * 70)

    for i in range(n):
        sample       = dataset[i]
        kannada_text = sample.get("text", "")

        result: dict = {
            "sample_id":    i + 1,
            "kannada_text": kannada_text,
            "english_text": "",
            "detected_lang": "",
            "no_speech_prob": None,
            "category":    "Error",
            "confidence":  0.0,
            "location": {
                "raw_places": [],
                "display":    "Not detected",
                "lat":        None,
                "lon":        None,
            },
            "error": None,
        }

        try:
            # ── Step 1: in-memory audio ──────────────────────────────────
            audio_arr, sr = sample_to_numpy(sample)

            # ── Step 2: Whisper STT + translate ──────────────────────────
            transcription = transcriber.transcribe(audio_arr, sr)
            result.update({
                "english_text":   transcription["english_text"],
                "detected_lang":  transcription["detected_lang"],
                "no_speech_prob": transcription["no_speech_prob"],
            })

            # ── Step 3: BBMP classification ──────────────────────────────
            prediction = classifier.predict(transcription["english_text"])
            result.update({
                "category":   prediction.category,
                "confidence": prediction.confidence,
            })

            # ── Step 4: Location extraction ──────────────────────────────
            loc = extractor.extract(transcription["english_text"])
            result["location"] = {
                "raw_places": loc.raw_places,
                "display":    loc.display,
                "lat":        loc.lat,
                "lon":        loc.lon,
            }

        except TranscriptionError as exc:
            result["error"] = f"Transcription failed: {exc}"
        except ClassificationError as exc:
            result["error"] = f"Classification failed: {exc}"
        except Exception as exc:
            result["error"] = f"Unexpected error: {exc}"

        # ── Print summary ────────────────────────────────────────────────
        _print_result(result)
        results.append(result)

    print("=" * 70)
    print(f"\n🎉 Pipeline complete – {len(results)} samples processed.")
    return results


def _print_result(r: dict) -> None:
    sid   = r["sample_id"]
    error = r.get("error")
    conf  = f"{r['confidence']:.0%}" if r["confidence"] else "n/a"
    print(f"Sample {sid:02d}")
    print(f"  Kannada  : {r['kannada_text'][:80]}…")
    if error:
        print(f"  ⚠ Error  : {error}")
    else:
        print(f"  English  : {r['english_text']}")
        print(f"  Category : {r['category']}  (confidence: {conf})")
        print(f"  Location : {r['location']['display']}")
        nsp = r.get("no_speech_prob")
        if nsp is not None and nsp > 0.5:
            print(f"  ⚠ Warning: high no-speech probability ({nsp:.0%})")
    print("-" * 60)