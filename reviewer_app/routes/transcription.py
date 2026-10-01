"""
LakeRCM Transcription Route — POST /api/transcribe

Accepts a short WAV audio phrase (multipart) and returns its transcript, for
the live microphone in the assistant chat composer. Audio is brokered to a
managed FM serving endpoint (services/transcription.py) and is NEVER logged or
persisted.

Guards:
  - 403 : the unauthenticated demo fallback identity may not transcribe
  - 429 : per-user sliding-window rate limit (in-memory, per pod)
  - 413 : audio chunk over the 5 MB cap (well under Model Serving's 16 MB limit)
  - 400 : payload is not WAV/RIFF bytes
  - 502 : transcription engine error
"""

import logging
import time
from collections import defaultdict

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from config import settings
from dependencies import get_current_user_email
from services.transcription import transcribe_audio

router = APIRouter(prefix="/api/transcribe")
logger = logging.getLogger(__name__)

MAX_BYTES = 5 * 1024 * 1024  # 5 MB

# Per-user sliding window (in-memory; per pod — fine for a single-replica app.
# If the app scales horizontally, back this with Lakebase/Redis).
_hits: dict[str, list[float]] = defaultdict(list)


def _rate_ok(email: str) -> bool:
    now = time.time()
    window = _hits[email] = [t for t in _hits[email] if now - t < 60]
    if len(window) >= settings.transcription_rate_per_min:
        return False
    window.append(now)
    return True


@router.post("")
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
    seq_id: int = Form(0),
):
    email = get_current_user_email(request)
    if email == "demo@example.com":
        raise HTTPException(
            status_code=403, detail="Transcription requires an authenticated user"
        )

    data = await audio.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio chunk too large")
    if data[:4] != b"RIFF":
        raise HTTPException(status_code=400, detail="Expected WAV (RIFF) bytes")

    # Rate-limit only well-formed requests — malformed ones (413/400) must not
    # burn a user's per-minute budget.
    if not _rate_ok(email):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        text = await transcribe_audio(data)  # no audio in logs/spans
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — log the message only, never the bytes
        logger.error("transcription failed: %s", e)
        raise HTTPException(status_code=502, detail="Transcription service error")

    return {"text": text, "seq_id": seq_id}
