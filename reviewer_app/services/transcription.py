"""
Speech-to-text transcription service.

Default engine ("fm") brokers audio to a managed Databricks multimodal
Foundation Model serving endpoint via the OpenAI-compatible chat-completions
API, passing the audio as an `audio_url` content block. This keeps audio inside
Databricks-brokered serving (no browser SpeechRecognition).

Validated live (2026-07-05) against `databricks-gemini-3-5-flash` on both AWS
and Azure workspaces:
  - the endpoint accepts `{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,…"}}`
    (the `input_audio` shape is rejected with 400);
  - the transcript is at `choices[0].message.content`, which for this reasoning
    model comes back as a LIST of blocks (`[{"type":"text","text":…}]`), NOT a
    plain string — hence the shape-tolerant parsing in `_extract_text` below.

Auth mirrors services/agent_client.py: a fresh header per call from
WorkspaceClient (Databricks Apps tokens rotate, so we never cache a client).
Audio bytes / base64 are NEVER logged.
"""

import logging

import httpx
from databricks.sdk import WorkspaceClient

from config import settings

logger = logging.getLogger(__name__)

# Comfortable ceiling — this is a reasoning model, so "thinking" tokens share
# the budget with the transcript; 256 truncated real speech in testing.
_MAX_TOKENS = 1024

# Defensive: some models prepend a spoken-preamble despite the instruction.
_PREAMBLES = (
    "here is the transcription:",
    "here is the transcript:",
    "the audio says:",
    "transcript:",
    "transcription:",
)


def _host_headers() -> tuple[str, dict]:
    """Fresh workspace host + auth headers (token rotates — never cache)."""
    w = WorkspaceClient()
    host = (w.config.host or "").rstrip("/")
    headers = dict(w.config._header_factory())
    headers["Content-Type"] = "application/json"
    return host, headers


def _extract_text(content) -> str:
    """Normalize a chat-completions `message.content` to plain text.

    Handles both shapes returned by Databricks FM endpoints:
      - reasoning models  -> list of blocks: [{"type":"text","text":"…"}, …]
      - non-reasoning     -> a plain string
    """
    if isinstance(content, list):
        parts = [
            (b.get("text") or "")  # a block may carry text=null; coerce to ""
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "".join(parts)
    if isinstance(content, str):
        return content
    return ""


def _strip_preamble(text: str) -> str:
    low = text.lower()
    for p in _PREAMBLES:
        if low.startswith(p):
            return text[len(p) :].strip()
    return text


async def transcribe_audio(audio_bytes: bytes, fmt: str = "wav") -> str:
    """Transcribe raw audio bytes to text. Raises on empty/failed transcription.

    NEVER logs the audio bytes or their base64 encoding.
    """
    if settings.transcription_engine != "fm":
        # PART G upgrade path — self-hosted Whisper. Lazy import so the default
        # FM path never requires the module to exist.
        from services.transcription_whisper import transcribe_whisper

        return await transcribe_whisper(audio_bytes, fmt)

    import base64

    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    host, headers = _host_headers()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the audio verbatim. Output ONLY the "
                            "transcript text — no preamble, no quotes."
                        ),
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/{fmt};base64,{b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": _MAX_TOKENS,
    }

    url = f"{host}/serving-endpoints/{settings.transcription_endpoint}/invocations"
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        resp = await client.post(url, headers=headers, json=payload)
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    text = _strip_preamble(_extract_text(content).strip())

    if not text:
        # Fail loud, but never log the audio.
        logger.warning("FM transcription returned empty content")
        raise RuntimeError("empty transcript")
    return text
