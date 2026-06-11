"""Local, offline punctuation restoration — the safety-block fallback.

When Gemini's non-configurable core filter rejects a transcript (e.g.
PROHIBITED_CONTENT on extreme-but-newsworthy content), the LLM cannot restore
punctuation no matter the safety_settings. This module provides a purely local
alternative (deepmultilingualpunctuation, a HuggingFace token-classification
model) that runs on-device with NO content filtering, so the blocked group can
still advance through the pipeline instead of dead-ending.

Caveats vs. the LLM:
  - restores sentence punctuation (. , ? -) but NOT casing;
  - no speaker-diarization cues are added.
The model is loaded lazily and cached as a process-wide singleton so the (heavy)
weights are only paid for if a block actually occurs.
"""

from utils.logger import get_logger
from config.constants import LOCAL_PUNCTUATION_MODEL

logger = get_logger(__name__)

# Speaker-turn marker used throughout the pipeline; preserved across restoration.
_SPEAKER_MARKER = " >> "

_model = None          # cached PunctuationModel instance
_load_failed = False   # True once a load attempt has failed, to avoid retry storms


def _get_model():
    """Lazily construct and cache the local punctuation model. Returns None if the
    dependency/model is unavailable (caller then falls back to raw text)."""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    try:
        from deepmultilingualpunctuation import PunctuationModel
        logger.info(f"Loading local punctuation model '{LOCAL_PUNCTUATION_MODEL}' "
                    f"(safety-block fallback)...")
        _model = PunctuationModel(model=LOCAL_PUNCTUATION_MODEL)
        logger.info("Local punctuation model loaded.")
        return _model
    except Exception as e:
        _load_failed = True
        logger.warning(f"Local punctuation model unavailable ({e}); "
                       f"safety-blocked rows will keep raw text.")
        return None


def restore_punctuation(text: str):
    """Restore punctuation locally, preserving ' >> ' speaker markers.

    Returns (restored_text, succeeded). On any failure returns (original_text, False)
    so the caller can record PunctuationSource.RAW rather than crash.
    """
    raw = text or ""
    if not raw.strip():
        return raw, False

    model = _get_model()
    if model is None:
        return raw, False

    try:
        # Punctuate each speaker segment independently so the markers survive.
        segments = raw.split(_SPEAKER_MARKER)
        restored = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                restored.append(seg)
                continue
            restored.append(model.restore_punctuation(seg))
        return _SPEAKER_MARKER.join(restored), True
    except Exception as e:
        logger.warning(f"Local punctuation restoration failed ({e}); keeping raw text.")
        return raw, False
