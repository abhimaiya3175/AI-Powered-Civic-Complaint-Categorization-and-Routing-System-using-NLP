"""
backend/services/translation_service.py
=========================================
IndicTrans2 / NLLB translation logic.
"""

import asyncio
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from backend.config import INDIC_LANG_TAGS, logger
from backend.utils.helpers import apply_civic_translation_glossary


def _translate_batch_sync(
    text_batch: list,
    src_lang_tag: str,
    tgt_lang_tag: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    processor: Any,
    translation_backend: str,
) -> list:
    """Run one CPU translation batch with backend-specific preprocessing."""

    if translation_backend == "indictrans2":
        model_inputs = processor.preprocess_batch(text_batch, src_lang=src_lang_tag, tgt_lang=tgt_lang_tag)
        encoded_inputs = tokenizer(
            model_inputs,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        encoded_inputs = {key: value.to("cpu") for key, value in encoded_inputs.items()}
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_inputs,
                max_length=256,
                num_beams=5,
                do_sample=False,
                use_cache=False,
            )
        decoded_batch = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return processor.postprocess_batch(decoded_batch, lang=tgt_lang_tag)

    if translation_backend == "nllb":
        try:
            tokenizer.src_lang = src_lang_tag
        except Exception:
            pass
        encoded_inputs = tokenizer(
            text_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        encoded_inputs = {key: value.to("cpu") for key, value in encoded_inputs.items()}
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_tag)
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=256,
                num_beams=5,
                do_sample=False,
                use_cache=True,
            )
        decoded_batch = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [(text or "").strip() for text in decoded_batch]

    # Fallback / rotary path: language-tagged inputs
    tagged_inputs = [
        f"{src_lang_tag} {tgt_lang_tag} {(text or '').strip()}".strip()
        for text in text_batch
    ]
    encoded_inputs = tokenizer(
        tagged_inputs,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )
    encoded_inputs = {key: value.to("cpu") for key, value in encoded_inputs.items()}
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded_inputs,
            max_length=256,
            num_beams=5,
            do_sample=False,
            use_cache=False,
        )
    decoded_batch = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return [(text or "").strip() for text in decoded_batch]


async def translate_text_with_indictrans2(
    text: str,
    source_language: str,
    target_language: str,
    app_state,
) -> str:
    """Mandatory step 2: dedicated NLP translation after Whisper transcription.

    Uses IndicTrans2 with IndicTransToolkit preprocessing/postprocessing.
    """
    clean_text = (text or "").strip()
    if not clean_text or source_language == target_language:
        return clean_text

    if target_language != "en":
        logger.warning(
            "Requested translation to '%s' with Indic→English model. Returning original text.",
            target_language,
        )
        return clean_text

    if source_language not in INDIC_LANG_TAGS or target_language not in INDIC_LANG_TAGS:
        return clean_text

    if source_language == "en":
        return clean_text

    tokenizer = getattr(app_state, "indictrans_tokenizer", None)
    model = getattr(app_state, "indictrans_model", None)
    processor = getattr(app_state, "indic_processor", None)
    translation_backend = getattr(app_state, "translation_backend", "unknown")
    translation_lock = getattr(app_state, "translation_lock", None)

    if tokenizer is None or model is None or processor is None or translation_lock is None:
        logger.warning(
            "Translation assets are not available (backend=%s). Returning original text.",
            translation_backend,
        )
        return clean_text

    src_lang_tag = INDIC_LANG_TAGS[source_language]
    tgt_lang_tag = INDIC_LANG_TAGS[target_language]

    try:
        async with translation_lock:
            translated_batch = await asyncio.to_thread(
                _translate_batch_sync,
                [clean_text],
                src_lang_tag,
                tgt_lang_tag,
                tokenizer,
                model,
                processor,
                translation_backend,
            )

        translated_text = (translated_batch[0] if translated_batch else "").strip()
        if not translated_text:
            return clean_text

        corrected_text = apply_civic_translation_glossary(
            clean_text, translated_text, source_language, target_language
        )
        if corrected_text != translated_text:
            logger.info(
                "Applied Kannada civic glossary correction: %s -> %s",
                translated_text, corrected_text,
            )

        return corrected_text or clean_text
    except Exception as exc:
        logger.error(
            "IndicTrans2 translation failed (%s->%s): %s",
            source_language, target_language, exc,
        )
        return clean_text
