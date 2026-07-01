import re


def normalize_feature_text(text: str) -> str:
    """Normalize whitespace while preserving multilingual scripts."""
    return re.sub(r"\s+", " ", str(text or "").strip())


def build_multilingual_classification_text(
    original_text: str,
    translated_english_text: str,
) -> str:
    """
    Build the text consumed by the classifier.

    English translation stays first to preserve the legacy signal. The original
    complaint is appended when it adds a distinct multilingual signal.
    """
    translated = normalize_feature_text(translated_english_text)
    original = normalize_feature_text(original_text)

    if not original:
        return translated
    if not translated:
        return original
    if original.casefold() == translated.casefold():
        return translated

    return translated
