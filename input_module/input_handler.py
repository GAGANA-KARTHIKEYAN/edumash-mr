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
        import whisper as w
        _whisper = w.load_model("tiny")
        print("[input] Local Whisper loaded ✓")
    except Exception as e:
        print(f"[input] Failed to load local whisper: {e}")
        pass
    return _whisper


def transcribe_audio(audio_path: str) -> str:
    # Attempt 1: Ultra-fast Groq API Transcription
    try:
        from explanation_module.engine import _groq_client
        if _groq_client is not None:
            print("[input] Sending audio to Groq Whisper API...")
            with open(audio_path, "rb") as file:
                transcription = _groq_client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model="whisper-large-v3-turbo",
                    prompt="Student answering a curriculum question.",
                    response_format="text",
                    temperature=0.0
                )
            
            # Anti-hallucination for silent audio files
            clean_text = transcription.strip().lower()
            hallucinations = ["thank you.", "thank you", "thanks for watching", "thanks for watching.", "please subscribe"]
            if clean_text in hallucinations:
                print(f"[input] Filtered Whisper silence hallucination: '{transcription.strip()}'")
                return ""
                
            return transcription
    except Exception as e:
        print(f"[input] Groq Whisper failed, falling back to local: {e}")

    # Attempt 2: Local CPU Transcription Fallback
    model = _load_whisper()
    if model is None:
        return ""
    try:
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except Exception as e:
        print(f"[input] Local audio transcription failed: {e}")
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

    # No valid input found
    return ""
