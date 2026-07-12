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

    English translation comes first (strongest signal for the TF-IDF model).
    The native-script original is appended so the model also sees civic
    keywords like 'ಬೀದಿ ದೀಪ', 'ಗುಂಡಿ', 'गड्ढा', 'नाली' that survive
    even when IndicTrans2 produces a garbled English output.

    This makes the ML classifier consistent with the keyword-override layer
    which already checks the original `transcribed_text` separately.
    """
    translated = normalize_feature_text(translated_english_text)
    original = normalize_feature_text(original_text)

    if not original:
        return translated
    if not translated:
        return original
    # If translation equals original (English input), no need to duplicate
    if original.casefold() == translated.casefold():
        return translated

    # Append native original — gives the classifier the benefit of both signals
    return f"{translated} {original}"
