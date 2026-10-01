"""
Content hashing for eval reproducibility.

An MLflow run id tells you *that* an eval happened; it doesn't tell you whether
two runs scored the same data with the same scorer definitions. When a scorer
is reworded or the dataset is re-curated, old and new runs become
incommensurable with no marker to detect it. These hashes are that marker:
log them on every eval/optimize run and tag the promoted prompt version, so a
production output traces back to exact train data + exact scorer definitions.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

__all__ = ["content_hash", "scorer_set_hash"]


def content_hash(records: Any, length: int = 16) -> str:
    """Stable short hash of dataset records (order-independent).

    Accepts a list of dict-like records or a DataFrame-convertible object.
    Canonicalizes each record to sorted-key JSON, sorts the resulting lines,
    and hashes — so row order and key order don't change the digest.
    """
    rows = _coerce_rows(records)
    canon = sorted(
        json.dumps(r, sort_keys=True, default=str, ensure_ascii=False) for r in rows
    )
    digest = hashlib.sha256("\n".join(canon).encode("utf-8")).hexdigest()
    return digest[:length]


def scorer_set_hash(scorers: list[Any], length: int = 16) -> str:
    """Stable short hash of a scorer set's identity AND definitions.

    Captures each scorer's name plus a fingerprint of its behavior: for
    Guidelines scorers the guideline text, for code scorers the source of the
    callable. Rewording a guideline or editing a scorer body changes the hash,
    flagging that prior evals are no longer comparable.
    """
    fingerprints = sorted(_scorer_fingerprint(s) for s in scorers)
    digest = hashlib.sha256("\n".join(fingerprints).encode("utf-8")).hexdigest()
    return digest[:length]


def _coerce_rows(records: Any) -> list[dict]:
    if hasattr(records, "to_dict"):
        # pandas DataFrame
        return records.to_dict(orient="records")
    return [dict(r) if not isinstance(r, dict) else r for r in records]


def _scorer_fingerprint(scorer: Any) -> str:
    name = getattr(scorer, "name", None) or getattr(
        scorer, "__name__", scorer.__class__.__name__
    )
    # Guidelines scorers carry their prompt in `.guidelines`.
    guidelines = getattr(scorer, "guidelines", None)
    if guidelines:
        body = (
            guidelines
            if isinstance(guidelines, str)
            else json.dumps(guidelines, sort_keys=True, default=str)
        )
        return f"{name}:guidelines:{body}"
    # Code scorers (the @scorer-decorated functions) — hash their source.
    target = getattr(scorer, "__wrapped__", scorer)
    try:
        src = inspect.getsource(target)
        return f"{name}:code:{hashlib.sha256(src.encode()).hexdigest()[:12]}"
    except (OSError, TypeError):
        # Built-in judges (Correctness, Safety, …) — identity is the name +
        # class; their definitions live in the mlflow version, captured
        # separately by the run's library logging.
        return f"{name}:builtin:{scorer.__class__.__name__}"
