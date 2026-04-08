# input_module/input_handler.py
# Handles text, optional speech (Whisper), and optional image (Tesseract)

import os

# ── Audio (optional) ────────────────────────────────────────────────
_whisper = None
def _load_whisper():
    global _whisper
    if _whisper is not None:
        return _whisper
    try:
        import openai_whisper as w
        _whisper = w.load_model("tiny")
        print("[input] Whisper loaded ✓")
    except Exception:
        pass
    return _whisper


def transcribe_audio(audio_path: str) -> str:
    model = _load_whisper()
    if model is None:
        return ""
    try:
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except Exception as e:
        print(f"[input] Audio transcription failed: {e}")
        return ""


# ── Image / OCR (optional) ─────────────────────────────────────────
def ocr_image(image_path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"[input] OCR failed: {e}")
        return ""


# ── Main entry point ────────────────────────────────────────────────
def get_input(
    text: str = None,
    audio_path: str = None,
    image_path: str = None
) -> str:
    """
    Returns cleaned student input string from any modality.
    Priority: audio > image > text > stdin prompt
    """
    if audio_path and os.path.exists(audio_path):
        transcribed = transcribe_audio(audio_path)
        if transcribed.strip():
            print(f"[input] Audio transcribed: {transcribed[:80]}…")
            return transcribed.strip()

    if image_path and os.path.exists(image_path):
        ocr_text = ocr_image(image_path)
        if ocr_text.strip():
            print(f"[input] OCR extracted: {ocr_text[:80]}…")
            return ocr_text.strip()

    if text and text.strip():
        return text.strip()

    # Fall back to CLI
    return input("Enter student answer: ").strip()
