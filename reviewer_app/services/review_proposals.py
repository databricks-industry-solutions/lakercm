"""Orchestration for agent-assisted review of held documents.

Keeps the I/O out of ``services.remediation``, which stays a pure function of
(reason, observed code, terminology) so it can be tested exhaustively. This
module supplies the inputs — the document's review reasons, the codes the
pipeline flagged, and the curated terminology — and decides the control-group
split.

Read path is the warehouse, not Lakebase: ``review_reasons`` rides the synced
gold table but ``validated_codes`` does not, and both are needed to say which
specific code caused the hold. One joined read gets both.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from databricks.sdk import WorkspaceClient

from config import settings
from services.remediation import (
    CPT_HCPCS,
    ICD10,
    REASON_MISSING_MEMBER_ID,
    Remediation,
    Terminology,
    TerminologyRow,
    remediate_document,
)
from services.warehouse import warehouse_rows

logger = logging.getLogger(__name__)

# Terminology is a few hundred curated rows that change only when
# seed_reference_data.py runs, so it is cached process-wide. Short enough that a
# re-seed is picked up within a review session without a redeploy.
_TERMINOLOGY_TTL_S = 900.0
_terminology_cache: Optional[Tuple[float, Terminology]] = None

# Share of held documents whose proposals are computed but never shown. Without
# a control group, "the agent made review faster" cannot be distinguished from
# reviewers getting faster at the corpus, so the comparison has to be built in
# from the start rather than reconstructed later.
DEFAULT_HOLDOUT_RATE = 0.2

SOURCE_PANE = "agent_pane"
SOURCE_TRIAGE = "agent_triage"


def _qualified(table: str) -> str:
    return f"{settings.catalog}.{settings.lakercm_schema}.{table}"


def _as_list(value: Any) -> List[Any]:
    """Coerce a warehouse column to a list.

    JSON_ARRAY format returns ARRAY/STRUCT columns as JSON text, but a Lakebase
    read of the same column gives a parsed list. Accept both so callers do not
    have to care which tier answered.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


# ── terminology ─────────────────────────────────────────────────────────────

_TERMINOLOGY_SQL = """
SELECT code, description, code_system, is_billable
FROM {icd}
UNION ALL
SELECT code, description, code_system, TRUE AS is_billable
FROM {cpt}
"""


def load_terminology(
    workspace_client: WorkspaceClient,
    warehouse_id: str,
    force: bool = False,
) -> Terminology:
    """The curated terminology, cached.

    ``ref_cpt_hcpcs`` has no ``is_billable`` column — billability is an
    ICD-10-CM concept — so procedure rows are selected as billable.
    """
    global _terminology_cache
    now = time.monotonic()
    if (
        not force
        and _terminology_cache
        and now - _terminology_cache[0] < _TERMINOLOGY_TTL_S
    ):
        return _terminology_cache[1]

    sql = _TERMINOLOGY_SQL.format(
        icd=_qualified("ref_icd10_cm"), cpt=_qualified("ref_cpt_hcpcs")
    )
    rows = warehouse_rows(workspace_client, warehouse_id, sql)
    terminology = Terminology(
        TerminologyRow(
            code=str(r.get("code") or ""),
            description=str(r.get("description") or ""),
            code_system=str(r.get("code_system") or "") or ICD10,
            # Statement Execution returns booleans as 'true'/'false' strings.
            is_billable=str(r.get("is_billable")).lower() != "false",
        )
        for r in rows
    )
    _terminology_cache = (now, terminology)
    logger.info("terminology loaded: %d codes", len(terminology))
    return terminology


def reset_terminology_cache() -> None:
    """Drop the cache (tests, and after a re-seed)."""
    global _terminology_cache
    _terminology_cache = None


# ── the held document's flagged items ───────────────────────────────────────

_HELD_DOCUMENT_SQL = """
SELECT g.review_reasons AS review_reasons,
       c.validated_codes AS validated_codes
FROM {labels} g
LEFT JOIN {codes} c ON c.document_path = g.document_path
WHERE g.document_path = :document_path
LIMIT 1
"""


def fetch_flagged_items(
    workspace_client: WorkspaceClient,
    warehouse_id: str,
    document_path: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """``(review_reasons, validated_codes)`` for one document from gold.

    Codes come from the base ``gold_claim_codes`` table (only ``validated_codes``
    is selected — no PHI columns). The PHI-safe ``gold_claim_codes_secure`` view,
    which Genie uses, drops ``document_path`` and pseudonymizes ``document_name``,
    so it has no identity key that joins back to ``gold_extraction_labels``. This
    is the reviewer app's own PHI-authorized backend — the same surface that
    already reads real document names from ``gold_extraction_labels`` — so the
    base table is the correct source here, and the reviewer SP holds SELECT on it.
    """
    sql = _HELD_DOCUMENT_SQL.format(
        labels=_qualified("gold_extraction_labels"),
        codes=_qualified("gold_claim_codes"),
    )
    rows = warehouse_rows(
        workspace_client, warehouse_id, sql, {"document_path": document_path}
    )
    if not rows:
        return [], []
    row = rows[0]
    reasons = [str(r) for r in _as_list(row.get("review_reasons")) if r]
    codes = [c for c in _as_list(row.get("validated_codes")) if isinstance(c, dict)]
    return reasons, codes


# ── extraction fidelity (did a stored code stop matching the page?) ──────────

_FIDELITY_SQL = """
SELECT fidelity_status AS fidelity_status,
       codes_rewritten AS codes_rewritten,
       rewritten_and_auto_verified AS rewritten_and_auto_verified
FROM {fidelity}
WHERE document_path = :document_path
LIMIT 1
"""


def fetch_extraction_fidelity(
    workspace_client: WorkspaceClient,
    warehouse_id: str,
    document_path: str,
) -> Optional[Dict[str, Any]]:
    """Whether this document's stored codes still match the source page.

    Returns ``None`` — not an error — whenever there is nothing to say, which is
    the normal case in three different ways, all of which must degrade to "show
    no badge" rather than to a 500:

    * ``gold_extraction_fidelity`` does not exist yet. It is a new dataset in the
      analytics pipeline, so every target has a window between this code
      deploying and that pipeline's first update.
    * The document is real. Fidelity is scored against the synthetic generator's
      manifest, the only ground truth that survives ai_parse_document's
      normalisation, so a real document has no row (or scores
      ``no_ground_truth``).
    * The extraction was faithful, which is the overwhelming majority.

    Only a genuine divergence returns a dict, so the caller can treat a truthy
    result as "there is something to tell the reviewer".
    """
    sql = _FIDELITY_SQL.format(fidelity=_qualified("gold_extraction_fidelity"))
    try:
        rows = warehouse_rows(
            workspace_client, warehouse_id, sql, {"document_path": document_path}
        )
    except Exception as exc:  # noqa: BLE001 - badge-only; never break the review
        logger.info("extraction fidelity unavailable for %r: %s", document_path, exc)
        return None
    if not rows:
        return None

    row = rows[0]
    status = str(row.get("fidelity_status") or "")
    if status not in ("rewritten", "dropped"):
        return None

    rewritten = [
        {
            "source_code": str(r.get("source_code") or ""),
            "stored_as": str(r.get("stored_as") or ""),
        }
        for r in _as_list(row.get("codes_rewritten"))
        if isinstance(r, dict) and r.get("source_code")
    ]
    return {
        "fidelity_status": status,
        "codes_rewritten": rewritten,
        # Statement Execution returns booleans as 'true'/'false' strings.
        "rewritten_and_auto_verified": str(
            row.get("rewritten_and_auto_verified")
        ).lower()
        == "true",
    }


# ── the routing decision, with everything needed to explain it ───────────────

_ROUTING_SQL = """
SELECT g.is_automated AS is_automated,
       g.confidence_score AS confidence_score,
       g.review_reasons AS review_reasons,
       c.validated_codes AS validated_codes
FROM {labels} g
LEFT JOIN {codes} c ON c.document_path = g.document_path
WHERE g.document_path = :document_path
LIMIT 1
"""


def fetch_routing_decision(
    workspace_client: WorkspaceClient,
    warehouse_id: str,
    document_path: str,
) -> Optional[Dict[str, Any]]:
    """The pipeline's routing decision plus the detail that explains it.

    One read for all four columns. ``is_automated`` is fetched rather than
    recomputed from ``confidence_score``: the decision is made once, in
    ``gold_extraction_labels``, and an app that re-derives it will disagree with
    the pipeline the first time either definition moves.

    Returns ``None`` when the document has no gold row — it has not been through
    the pipeline, so there is no decision to report. That is distinct from a
    failed read, which raises: an unreadable warehouse must not look like an
    unprocessed document.

    ``is_automated`` is returned RAW. Statement Execution serialises booleans as
    strings, and ``services.hold_reasons.coerce_bool`` is the one place that
    knows how to read them while keeping "unknown" distinguishable from "false".
    """
    sql = _ROUTING_SQL.format(
        labels=_qualified("gold_extraction_labels"),
        codes=_qualified("gold_claim_codes"),
    )
    rows = warehouse_rows(
        workspace_client, warehouse_id, sql, {"document_path": document_path}
    )
    if not rows:
        return None
    row = rows[0]
    confidence = row.get("confidence_score")
    try:
        confidence = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence = None
    return {
        "is_automated": row.get("is_automated"),
        "confidence_score": confidence,
        "review_reasons": [str(r) for r in _as_list(row.get("review_reasons")) if r],
        "validated_codes": [
            c for c in _as_list(row.get("validated_codes")) if isinstance(c, dict)
        ],
    }


# ── uncaptured codes (did a code on the page never reach the claim?) ─────────

_UNCAPTURED_SQL = """
SELECT uncaptured_codes AS uncaptured_codes,
       uncaptured_count AS uncaptured_count,
       uncaptured_billable_count AS uncaptured_billable_count
FROM {uncaptured}
WHERE document_path = :document_path
LIMIT 1
"""


def fetch_uncaptured_codes(
    workspace_client: WorkspaceClient,
    warehouse_id: str,
    document_path: str,
) -> Optional[Dict[str, Any]]:
    """Diagnosis codes printed on the page that never reached the claim.

    Deliberately shaped like :func:`fetch_extraction_fidelity`: ``None`` for
    "nothing to say", and any failure degrades to ``None`` rather than breaking
    the review. ``silver_uncaptured_codes`` is a new dataset, so every target has
    a window between this code deploying and the documents pipeline's next
    update where the table is simply absent.

    Unlike fidelity this needs no ground truth — it reads the parsed page text —
    so it works on real documents. It is ADVISORY: measured 87.4% precision,
    where the 12.6% are codes printed in a payer policy table rather than
    assigned to the patient. That is why it informs a reviewer instead of
    gating auto-verification.
    """
    sql = _UNCAPTURED_SQL.format(uncaptured=_qualified("silver_uncaptured_codes"))
    try:
        rows = warehouse_rows(
            workspace_client, warehouse_id, sql, {"document_path": document_path}
        )
    except Exception as exc:  # noqa: BLE001 - advisory only; never break review
        logger.info("uncaptured codes unavailable for %r: %s", document_path, exc)
        return None
    if not rows:
        return None

    row = rows[0]
    codes = [
        str(c.get("code") or "")
        for c in _as_list(row.get("uncaptured_codes"))
        if isinstance(c, dict) and c.get("code")
    ]
    if not codes:
        return None
    # Statement Execution returns booleans as 'true'/'false' strings, so the
    # billable split has to be read as text.
    billable = [
        str(c.get("code") or "")
        for c in _as_list(row.get("uncaptured_codes"))
        if isinstance(c, dict)
        and c.get("code")
        and str(c.get("is_billable")).lower() == "true"
    ]
    return {
        "uncaptured_codes": codes,
        "uncaptured_billable": billable,
        "uncaptured_count": len(codes),
    }


_HELD_DOCUMENTS_SQL = """
SELECT g.document_path AS document_path,
       g.review_reasons AS review_reasons,
       c.validated_codes AS validated_codes
FROM {labels} g
LEFT JOIN {codes} c ON c.document_path = g.document_path
WHERE g.is_automated = FALSE
  AND size(g.review_reasons) > 0
ORDER BY g.extracted_at DESC
LIMIT {limit}
"""


def fetch_held_documents(
    workspace_client: WorkspaceClient,
    warehouse_id: str,
    limit: int = 1000,
) -> List[Tuple[str, List[str], List[Dict[str, Any]]]]:
    """Every held document with its reasons and flagged codes, in one read.

    The batch form of fetch_flagged_items. Triage over a whole corpus one
    document at a time would be one warehouse round trip per document, which on
    a demo corpus is thousands of queries for data that fits in a single result.
    """
    sql = _HELD_DOCUMENTS_SQL.format(
        labels=_qualified("gold_extraction_labels"),
        codes=_qualified("gold_claim_codes"),
        limit=int(limit),
    )
    out: List[Tuple[str, List[str], List[Dict[str, Any]]]] = []
    for row in warehouse_rows(workspace_client, warehouse_id, sql):
        path = str(row.get("document_path") or "")
        if not path:
            continue
        reasons = [str(r) for r in _as_list(row.get("review_reasons")) if r]
        codes = [c for c in _as_list(row.get("validated_codes")) if isinstance(c, dict)]
        out.append((path, reasons, codes))
    return out


def document_remediations(
    workspace_client: WorkspaceClient,
    warehouse_id: str,
    document_path: str,
) -> List[Remediation]:
    """Every remediation for one held document. Empty when it is not held."""
    reasons, codes = fetch_flagged_items(workspace_client, warehouse_id, document_path)
    if not reasons:
        return []
    terminology = load_terminology(workspace_client, warehouse_id)
    return remediate_document(reasons, codes, terminology)


# ── control group ───────────────────────────────────────────────────────────


def should_withhold(document_id: str, rate: float = DEFAULT_HOLDOUT_RATE) -> bool:
    """Whether this document's proposals are withheld from the reviewer.

    Derived from the document id, not drawn at random: a document has to stay
    on the same side of the split for its whole life, or a re-run of triage
    would move it between arms and the turnaround comparison would be
    meaningless. Same id, same answer, on any host.
    """
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha256((document_id or "").encode("utf-8")).digest()
    # First 8 bytes as a fraction of the range.
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < rate


# ── recording ───────────────────────────────────────────────────────────────


def record_remediations(
    db,
    document_id: str,
    remediations: Sequence[Remediation],
    source: str,
    model: Optional[str] = None,
    holdout_rate: float = DEFAULT_HOLDOUT_RATE,
) -> List[Dict[str, Any]]:
    """Persist remediations as proposals.

    Only the CANDIDATES are recorded here — no ``proposed_value``. Choosing a
    candidate is the agent's judgment call and arrives later through the pane,
    so what this writes is the shortlist and the reason, which is what the
    reviewer needs in front of them even if the agent never runs.

    A ``not_resolvable`` remediation is written too, as a ``declined`` row. That
    is deliberate: a document nobody could fix is a fact worth counting, and it
    keeps ``missing_member_id`` out of the pending queue where it would look
    like work waiting to happen.
    """
    withheld = should_withhold(document_id, holdout_rate)
    written: List[Dict[str, Any]] = []
    for rem in remediations:
        row = db.insert_review_proposal(
            document_id=document_id,
            review_reason=rem.review_reason,
            resolution=rem.resolution,
            source=source,
            field_name=rem.field_name,
            observed_value=rem.observed_code,
            rationale=rem.guidance,
            candidates=[c.as_dict() for c in rem.candidates],
            model=model,
            # A refusal is never part of the experiment: there is nothing to
            # withhold, and counting it in the control arm would dilute the
            # comparison with documents the agent was never going to help.
            withheld=False if rem.resolution == "not_resolvable" else withheld,
        )
        written.append(row)
    return written


__all__ = [
    "CPT_HCPCS",
    "fetch_held_documents",
    "DEFAULT_HOLDOUT_RATE",
    "ICD10",
    "REASON_MISSING_MEMBER_ID",
    "SOURCE_PANE",
    "SOURCE_TRIAGE",
    "document_remediations",
    "fetch_flagged_items",
    "load_terminology",
    "record_remediations",
    "reset_terminology_cache",
    "should_withhold",
]
