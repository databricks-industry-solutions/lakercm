"""Agent-assisted review endpoints: remediation candidates and proposals.

Held documents arrive with a reason (``gold_extraction_labels.review_reasons``)
but no suggested fix. These routes supply the deterministic shortlist of
possible fixes, and record what the agent proposed and what the reviewer did
with it.

All writes stay on this path. The agent app never writes: it stages a card, the
reviewer approves, and the resulting call lands here under the reviewer's
identity — so there is one audit trail and the agent needs no write grant.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from schemas import (
    DocumentFidelityResponse,
    DocumentHoldReasonsResponse,
    DocumentRemediationResponse,
    HoldReasonItem,
    ProposalDispositionSubmission,
    RemediationCandidate,
    RemediationItem,
    ReviewProposal,
    ReviewProposalListResponse,
    ReviewProposalSubmission,
    RewrittenCode,
)
from dependencies import (
    get_workspace_client,
    get_lakercm_db,
    get_current_user_email,
)
from config import settings
from services import hold_reasons
from services import review_proposals as rp
from services.pipeline_trigger import trigger_analytics_pipeline
from services.remediation import (
    KNOWN_REASONS as REMEDIABLE_REASONS,
    NOT_RESOLVABLE,
    remediate_document,
)
from services.warehouse import WarehouseReadError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# What a reviewer can do to a pending proposal. 'declined' and 'superseded' are
# system transitions, not reviewer verdicts, so they are not accepted here.
_REVIEWER_DISPOSITIONS = frozenset({"accepted", "modified", "rejected"})

_VALID_RESOLUTIONS = frozenset({"deterministic", "needs_judgment", NOT_RESOLVABLE})


def _document_or_404(db, document_id: str) -> dict:
    document = db.get_document_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get(
    "/documents/{document_id}/remediation",
    response_model=DocumentRemediationResponse,
)
async def get_document_remediation(
    document_id: str,
    db=Depends(get_lakercm_db),
    workspace_client=Depends(get_workspace_client),
):
    """Deterministic candidate fixes for a held document.

    Computed on demand from the curated terminology, so it reflects a re-seed
    without waiting for a pipeline run. A document that is not held returns
    ``is_held=false`` and no items rather than 404 — the pane calls this for
    whatever is open.
    """
    document = _document_or_404(db, document_id)
    document_path = document.get("file_path") or ""
    if not document_path:
        return DocumentRemediationResponse(document_id=document_id, is_held=False)

    try:
        warehouse_id = settings.get_warehouse_id()
        reasons, codes = rp.fetch_flagged_items(
            workspace_client, warehouse_id, document_path
        )
        if not reasons:
            return DocumentRemediationResponse(document_id=document_id, is_held=False)
        terminology = rp.load_terminology(workspace_client, warehouse_id)
        from services.remediation import remediate_document

        items = remediate_document(reasons, codes, terminology)
        return DocumentRemediationResponse(
            document_id=document_id,
            is_held=True,
            review_reasons=reasons,
            items=[RemediationItem(**item.as_dict()) for item in items],
        )
    except WarehouseReadError as e:
        # The gold tier is the only source for which code was flagged. Say so
        # rather than returning an empty shortlist, which would read as "no fix
        # is available" — the opposite of "we could not look".
        logger.warning("remediation read failed for %s: %s", document_id, e)
        raise HTTPException(
            status_code=503,
            detail=f"Remediation unavailable: {e}",
        )
    except Exception as e:
        logger.error("Failed to build remediation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to build remediation: {str(e)}"
        )


@router.get(
    "/documents/{document_id}/hold-reasons",
    response_model=DocumentHoldReasonsResponse,
)
async def get_document_hold_reasons(
    document_id: str,
    db=Depends(get_lakercm_db),
    workspace_client=Depends(get_workspace_client),
):
    """Everything that is wrong with one document, in one payload.

    The verdict is assembled here rather than in the browser because the browser
    got it wrong in the most expensive way available: ``ReviewPanel`` decided
    held-vs-auto-verified from two React props, the only routed page never passed
    them, and so every document rendered as held with a fabricated explanation.
    A verdict computed from props a caller can omit will eventually be the wrong
    verdict, stated confidently.

    Error handling is split by what an empty answer would MEAN, following the
    reasoning already argued for ``/remediation`` and ``/fidelity``:

    * the blocking read failing is reported as 503 — an empty blocking list reads
      as "nothing is wrong with this document", which is the opposite of "we
      could not look";
    * the advisory reads failing return 200 with ``degraded``, because "could not
      check" and "nothing to report" render identically for an advisory, and a
      503 would error every document the moment the analytics pipeline lagged.
    """
    document = _document_or_404(db, document_id)
    document_path = document.get("file_path") or ""
    threshold = settings.auto_verdict_threshold

    if not document_path:
        return DocumentHoldReasonsResponse(
            document_id=document_id,
            state=hold_reasons.STATE_UNKNOWN,
            auto_verdict_threshold=threshold,
            blocking=[],
            advisory=[],
        )

    warehouse_id = settings.get_warehouse_id()
    try:
        routing = rp.fetch_routing_decision(
            workspace_client, warehouse_id, document_path
        )
    except WarehouseReadError as e:
        logger.warning("hold-reasons read failed for %s: %s", document_id, e)
        raise HTTPException(status_code=503, detail=f"Hold reasons unavailable: {e}")

    if routing is None:
        # No gold row at all: the document has not been through the pipeline, so
        # there is no routing decision to report. Not an error, and not "held".
        return DocumentHoldReasonsResponse(
            document_id=document_id,
            state=hold_reasons.STATE_UNKNOWN,
            auto_verdict_threshold=threshold,
            blocking=[],
            advisory=[],
        )

    reasons = routing["review_reasons"]
    remediations = []
    # Only the code and member-id reasons have candidate fixes. A document held
    # purely on confidence has nothing for the terminology to offer, so loading
    # it would be a warehouse round trip that cannot change the answer.
    if any(r in REMEDIABLE_REASONS for r in reasons):
        try:
            terminology = rp.load_terminology(workspace_client, warehouse_id)
            remediations = remediate_document(
                reasons, routing["validated_codes"], terminology
            )
        except WarehouseReadError as e:
            # The reason is already known; only the candidate fixes are missing.
            # Report the reason without them rather than 503 the whole answer.
            logger.warning("terminology unavailable for %s: %s", document_id, e)

    degraded: list[str] = []
    uncaptured = fidelity = None
    try:
        uncaptured = rp.fetch_uncaptured_codes(
            workspace_client, warehouse_id, document_path
        )
        fidelity = rp.fetch_extraction_fidelity(
            workspace_client, warehouse_id, document_path
        )
    except Exception as e:  # noqa: BLE001 - advisory only; see the docstring
        logger.info("advisory lookups failed for %s: %s", document_id, e)
        degraded.append("advisory_unavailable")

    assessment = hold_reasons.assess_document(
        is_automated=routing["is_automated"],
        confidence_score=routing["confidence_score"],
        threshold=threshold,
        review_reasons=reasons,
        remediations=remediations,
        uncaptured=uncaptured,
        fidelity=fidelity,
        degraded=degraded,
    )
    return _hold_response(document_id, assessment)


def _hold_item(item: hold_reasons.HoldReason) -> HoldReasonItem:
    return HoldReasonItem(
        code=item.code,
        severity=item.severity,
        source=item.source,
        title=item.title,
        detail=item.detail,
        resolution=item.resolution,
        guidance=item.guidance,
        field_name=item.field_name,
        observed_code=item.observed_code,
        code_system=item.code_system,
        candidates=[RemediationCandidate(**c.as_dict()) for c in item.candidates],
        candidates_truncated=item.candidates_truncated,
        confidence_score=item.confidence_score,
        threshold=item.threshold,
        count=item.count,
    )


def _hold_response(
    document_id: str, assessment: hold_reasons.HoldAssessment
) -> DocumentHoldReasonsResponse:
    return DocumentHoldReasonsResponse(
        document_id=document_id,
        state=assessment.state,
        is_automated=assessment.is_automated,
        confidence_score=assessment.confidence_score,
        auto_verdict_threshold=assessment.threshold,
        reasons_recorded=assessment.reasons_recorded,
        unexplained=assessment.unexplained,
        blocking=[_hold_item(i) for i in assessment.blocking],
        advisory=[_hold_item(i) for i in assessment.advisory],
        degraded=list(assessment.degraded),
    )


@router.get(
    "/documents/{document_id}/fidelity",
    response_model=DocumentFidelityResponse,
)
async def get_document_fidelity(
    document_id: str,
    db=Depends(get_lakercm_db),
    workspace_client=Depends(get_workspace_client),
):
    """Has a stored clinical code stopped matching what the page printed?

    Deliberately never raises. Unlike remediation — where failing to read the
    gold tier has to be reported, because an empty shortlist would read as "no
    fix exists" — the honest rendering of "we could not check fidelity" and "the
    extraction was faithful" is the same: no badge. A 503 here would put an error
    state on every document the moment the analytics pipeline lagged, on a signal
    that is advisory by construction.

    Always 200 with ``has_finding=false`` unless there is a real divergence.
    """
    empty = DocumentFidelityResponse(document_id=document_id)
    try:
        document = _document_or_404(db, document_id)
    except HTTPException:
        raise
    document_path = document.get("file_path") or ""
    if not document_path:
        return empty

    try:
        finding = rp.fetch_extraction_fidelity(
            workspace_client, settings.get_warehouse_id(), document_path
        )
    except Exception as e:  # noqa: BLE001 - advisory badge; see the docstring
        logger.info("fidelity lookup failed for %s: %s", document_id, e)
        return empty
    if not finding:
        return empty

    return DocumentFidelityResponse(
        document_id=document_id,
        has_finding=True,
        fidelity_status=finding["fidelity_status"],
        codes_rewritten=[RewrittenCode(**c) for c in finding["codes_rewritten"]],
        rewritten_and_auto_verified=finding["rewritten_and_auto_verified"],
    )


@router.get(
    "/documents/{document_id}/proposals",
    response_model=ReviewProposalListResponse,
)
async def list_document_proposals(
    document_id: str,
    db=Depends(get_lakercm_db),
):
    """Proposals recorded for one document, newest first.

    Withheld (control-slice) proposals are excluded: showing one would put the
    document in both arms of the comparison it exists to support.
    """
    _document_or_404(db, document_id)
    try:
        rows = db.list_review_proposals(document_id, include_withheld=False)
        return ReviewProposalListResponse(
            document_id=document_id,
            proposals=[ReviewProposal(**row) for row in rows],
        )
    except Exception as e:
        logger.error("Failed to list proposals: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to list proposals: {str(e)}"
        )


@router.post(
    "/documents/{document_id}/proposals",
    response_model=ReviewProposal,
    status_code=201,
)
async def create_document_proposal(
    document_id: str,
    submission: ReviewProposalSubmission,
    db=Depends(get_lakercm_db),
):
    """Record a proposal the agent staged for this document.

    Called when the card is presented, not when it is approved, so a proposal
    the reviewer ignores or rejects is still counted. Counting only the
    approved ones would make the acceptance rate 100% by construction.
    """
    _document_or_404(db, document_id)

    if submission.resolution not in _VALID_RESOLUTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"resolution must be one of {sorted(_VALID_RESOLUTIONS)}",
        )
    proposed_value = submission.proposed_value
    if submission.resolution == NOT_RESOLVABLE:
        # A refusal carries no value. Dropped here as well as constrained in
        # the table, so the caller gets a stored row instead of an integrity
        # error it cannot act on.
        proposed_value = None
    elif not (proposed_value or "").strip():
        raise HTTPException(
            status_code=422,
            detail="proposed_value is required unless resolution is "
            f"'{NOT_RESOLVABLE}'",
        )

    try:
        # One pending card per target: a second proposal for the same field
        # supersedes the first rather than queueing a rival Approve button.
        db.supersede_pending_proposals(document_id, submission.correction_key)
        row = db.insert_review_proposal(
            document_id=document_id,
            review_reason=submission.review_reason,
            resolution=submission.resolution,
            source=rp.SOURCE_PANE,
            field_name=submission.field_name,
            correction_key=submission.correction_key,
            observed_value=submission.observed_value,
            proposed_value=proposed_value,
            rationale=submission.rationale,
            candidates=[c.model_dump() for c in submission.candidates],
            model=submission.model,
            withheld=False,
        )
        return ReviewProposal(**row)
    except Exception as e:
        logger.error("Failed to record proposal: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to record proposal: {str(e)}"
        )


@router.patch("/proposals/{proposal_id}", response_model=ReviewProposal)
async def set_proposal_disposition(
    request: Request,
    proposal_id: str,
    submission: ProposalDispositionSubmission,
    background_tasks: BackgroundTasks,
    db=Depends(get_lakercm_db),
    workspace_client=Depends(get_workspace_client),
):
    """Record what the reviewer did with a staged proposal.

    This is the measurement. Without it an approved proposal is
    indistinguishable from a correction the reviewer typed, and the acceptance
    rate cannot be computed at all.
    """
    if submission.disposition not in _REVIEWER_DISPOSITIONS:
        raise HTTPException(
            status_code=422,
            detail=f"disposition must be one of {sorted(_REVIEWER_DISPOSITIONS)}",
        )
    if (
        submission.disposition == "modified"
        and not (submission.human_value or "").strip()
    ):
        raise HTTPException(
            status_code=422,
            detail="human_value is required when disposition is 'modified'",
        )

    user_email = get_current_user_email(request)
    try:
        row = db.set_proposal_disposition(
            proposal_id=proposal_id,
            disposition=submission.disposition,
            disposition_by=user_email,
            human_value=submission.human_value,
        )
    except Exception as e:
        logger.error("Failed to set disposition: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to set disposition: {str(e)}"
        )
    if not row:
        # Either the id is unknown or it has already been dispositioned. Both
        # are 409 rather than 404: the second Approve click is the common case,
        # and it must not restamp the timestamp the turnaround metric uses.
        raise HTTPException(
            status_code=409,
            detail="Proposal is not pending (unknown id, or already resolved)",
        )
    # A disposition changed a proposal's outcome → refresh review analytics
    # event-driven (only reached on a real transition, never on the 409).
    background_tasks.add_task(trigger_analytics_pipeline, workspace_client)
    return ReviewProposal(**row)
