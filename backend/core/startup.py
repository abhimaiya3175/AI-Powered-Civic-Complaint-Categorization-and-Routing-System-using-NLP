"""
backend/core/startup.py
========================
FastAPI lifespan / startup event: preloads IndicTrans2 translation models.
"""

import asyncio
import logging
from typing import Any

import sys
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Monkey-patch transformers.onnx to avoid crash in IndicTrans2 remote code on newer transformers versions
if not hasattr(transformers, 'onnx'):
    import types
    transformers.onnx = types.ModuleType('transformers.onnx')
    sys.modules['transformers.onnx'] = transformers.onnx

from backend.config import (
    INDICTRANS2_MODEL_NAME,
    NLLB_FALLBACK_MODEL_NAME,
    ROTARY_INDICTRANS2_FALLBACK_MODEL_NAME,
    logger,
)

# ── IndicProcessor import with fallback ──────────────────────────────
try:
    from IndicTransToolkit import IndicProcessor  # type: ignore
except Exception:
    try:
        from IndicTransToolkit.processor import IndicProcessor  # type: ignore
    except Exception:
        IndicProcessor = None


class IndicProcessorFallback:
    """Fallback processor when IndicTransToolkit cannot be imported on the host."""

    def __init__(self, inference: bool = True):
        self.inference = inference

    def preprocess_batch(self, text_batch: list, src_lang: str, tgt_lang: str) -> list:
        return [f"{src_lang} {tgt_lang} {(text or '').strip()}".strip() for text in text_batch]

    def postprocess_batch(self, text_batch: list, lang: str) -> list:
        return [(text or "").strip() for text in text_batch]


def _load_indictrans2_assets(model_name: str):
    # Try local cache first to avoid network HEAD checks that may fail
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32,
            local_files_only=True,
        )
    except OSError:
        # Model not cached — fall back to network download
        logger.info("Model %s not in local cache, downloading…", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32
        )
    model.to("cpu")
    model.eval()
    processor_class = IndicProcessor if IndicProcessor is not None else IndicProcessorFallback
    processor = processor_class(inference=True)
    return tokenizer, model, processor


def _load_nllb_assets(model_name: str):
    # Try local cache first to avoid network HEAD checks that may fail
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, use_safetensors=False, torch_dtype=torch.float32,
            local_files_only=True,
        )
    except OSError:
        # Model not cached — fall back to network download
        logger.info("Model %s not in local cache, downloading…", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, use_safetensors=False, torch_dtype=torch.float32
        )
    model.to("cpu")
    model.eval()
    return tokenizer, model, IndicProcessorFallback(inference=True)


async def preload_translation_models(app_state) -> None:
    """Preload IndicTrans2 translation assets once at startup for CPU inference."""
    app_state.indictrans_tokenizer = None
    app_state.indictrans_model = None
    app_state.indic_processor = None
    app_state.translation_backend = "unavailable"
    app_state.translation_lock = asyncio.Lock()

    try:
        logger.info("Loading IndicTrans2 distilled CPU model: %s", INDICTRANS2_MODEL_NAME)
        tokenizer, model, processor = await asyncio.to_thread(
            _load_indictrans2_assets, INDICTRANS2_MODEL_NAME
        )
        app_state.indictrans_tokenizer = tokenizer
        app_state.indictrans_model = model
        app_state.indic_processor = processor
        app_state.translation_backend = "indictrans2"
        if IndicProcessor is None:
            logger.warning("IndicTransToolkit not installed; running IndicTrans2 with basic IndicProcessorFallback.")
        
        logger.info("IndicTrans2 + IndicProcessor loaded successfully (backend=indictrans2).")
        return
    except Exception as primary_exc:
        logger.error(
            "Failed to preload primary IndicTrans2 assets (%s): %s",
            INDICTRANS2_MODEL_NAME, primary_exc,
        )

    try:
        logger.info("Loading fallback translation model: %s", NLLB_FALLBACK_MODEL_NAME)
        tokenizer, model, processor = await asyncio.to_thread(
            _load_nllb_assets, NLLB_FALLBACK_MODEL_NAME
        )
        app_state.indictrans_tokenizer = tokenizer
        app_state.indictrans_model = model
        app_state.indic_processor = processor
        app_state.translation_backend = "nllb"
        logger.info("Fallback translation model loaded successfully (backend=nllb).")
        return
    except Exception as nllb_exc:
        logger.error(
            "Failed to preload fallback translation assets (%s): %s",
            NLLB_FALLBACK_MODEL_NAME, nllb_exc,
        )

    try:
        logger.info(
            "Loading tertiary fallback translation model: %s",
            ROTARY_INDICTRANS2_FALLBACK_MODEL_NAME,
        )
        tokenizer, model, processor = await asyncio.to_thread(
            _load_indictrans2_assets, ROTARY_INDICTRANS2_FALLBACK_MODEL_NAME
        )
        app_state.indictrans_tokenizer = tokenizer
        app_state.indictrans_model = model
        app_state.indic_processor = processor
        app_state.translation_backend = "rotary_indictrans2"
        logger.info(
            "Tertiary IndicTrans2-compatible model loaded successfully (backend=rotary_indictrans2)."
        )
    except Exception as rotary_exc:
        logger.error(
            "Failed to preload tertiary fallback translation assets (%s): %s",
            ROTARY_INDICTRANS2_FALLBACK_MODEL_NAME, rotary_exc,
        )
