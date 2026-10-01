"""
LakeRCM Pydantic Models

Request and response schemas for document management and analytics.
"""

from pydantic import BaseModel, Field, BeforeValidator
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime
from enum import Enum

StrUUID = Annotated[str, BeforeValidator(lambda v: str(v))]


class ProcessingStatus(str, Enum):
    # Three-state lifecycle for the reviewer UI:
    #   PROCESSING  -> file landed, extraction in flight
    #   PENDING     -> extraction done, awaiting human review
    #   AUTO_VERIFIED / FAILED -> terminal
    # "Reviewed" is derived (presence of a document_extraction_reviews row),
    # not a stored value here.
    PROCESSING = "processing"
    PENDING = "pending"
    AUTO_VERIFIED = "auto_verified"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    id: StrUUID
    filename: str
    volume_path: str
    file_size: int
    status: ProcessingStatus
    message: Optional[str] = None


class DocumentMetadata(BaseModel):
    id: StrUUID
    user_email: str
    document_name: str
    file_path: str
    file_size: int
    document_type: Optional[str]
    notes: Optional[str]
    processing_status: ProcessingStatus
    processing_error: Optional[str] = None
    num_pages: Optional[int] = None
    element_count: Optional[int] = None
    has_medical_entities: bool = False
    upload_timestamp: Optional[datetime] = None
    processing_timestamp: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    review_verdict: Optional[str] = None
    review_reviewer_email: Optional[str] = None
    review_reviewer_name: Optional[str] = None
    # Why a pending document is pending, for the card badge. `processing_status`
    # keeps its existing values — the Postgres CHECK constraint and every saved
    # filter still work — and these carry the specificity that "Pending" lacked.
    # NOTE: these must be DECLARED here, not merely selected: pydantic defaults
    # to extra='ignore', so an undeclared field is silently dropped on the way
    # out (the same trap documented on ExtractionComparisonItem).
    hold_primary_code: Optional[str] = None
    hold_primary_label: Optional[str] = None
    # Held with nothing accounting for it. Worth showing distinctly rather than
    # folding into the others: it is a defect signal, and it should be zero.
    hold_unexplained: bool = False


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class ExtractionElement(BaseModel):
    """One parsed element from ai_parse_document — a text/table/image region
    with a bounding box on the rasterized page."""

    page_number: Optional[int] = None
    element_type: Optional[str] = None
    text_content: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    confidence_score: Optional[float] = None


class DocumentElement(BaseModel):
    element_id: Optional[int] = None
    element_type: str
    content: str
    bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = None
    page_number: Optional[int] = None


class MedicalEntity(BaseModel):
    entity_type: str
    text: str
    confidence: float
    date: Optional[str] = None
    element_id: Optional[int] = None


class DocumentExtraction(BaseModel):
    dates: List[Dict[str, Any]] = Field(default_factory=list)
    medical_entities: List[MedicalEntity] = Field(default_factory=list)
    patient_info: Dict[str, Any] = Field(default_factory=dict)
    document_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total_count: int
    limit: int
    offset: int


class DocumentStatusCountsResponse(BaseModel):
    processing: int = 0
    pending: int = 0
    reviewed: int = 0
    auto_verified: int = 0
    failed: int = 0
    total: int = 0


class DocumentDetailResponse(BaseModel):
    document: DocumentMetadata
    elements: List[DocumentElement] = Field(default_factory=list)
    extractions: DocumentExtraction


class ExtractionComparisonItem(BaseModel):
    document_path: str
    document_name: str
    label: Optional[str] = None
    # Each identifier is {name, value, confidence, citations[]} per ai_extract
    # v2.1. Typed loosely as Dict[str, Any] so float confidence and the
    # nested citations array don't trip Pydantic's str-only coercion.
    identifiers: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    elements: Optional[List[ExtractionElement]] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    extracted_at: Optional[datetime] = None
    # Rendered per-page image URIs (ai_parse_document imageOutputPath), ordered
    # by page id; empty for docs that predate page rendering.
    #
    # List[Optional[str]], not List[str]: an element is NULL when that page has no
    # render, and the position still means "page N". ai_parse_document populates
    # one entry per page whether or not it produced an image, so
    # ["...page1.jpg", null] is a two-page document whose second page did not
    # render -- and routes/documents.py relies on that alignment
    # (`page_images[page]`, guarded by `if page_images[idx]` to fall back to the
    # raw upload).
    #
    # Typed List[str], pydantic rejected the null and failed validation for the
    # WHOLE comparison payload, so /comparisons 500'd and the entire review form
    # vanished -- not just one page image. Seen live on dev: 2 of 3,419 documents.
    #
    # Do NOT "fix" this by filtering nulls out of the array in silver. That shifts
    # every later page down an index, so the overlay would serve the wrong page's
    # render behind the bounding boxes -- a silent correctness bug, worse than the
    # loud failure it replaces.
    page_images: Optional[List[Optional[str]]] = Field(default_factory=list)
    # Whether the pipeline auto-verified this document, and if not, why. These
    # must be declared: the model ignores undeclared keys, so selecting them in
    # the query is not enough to get them to the UI. The reviewer UI needs both
    # because confidence alone does not decide auto-verification — it is
    # `confidence >= threshold AND size(review_reasons) = 0` — and inferring it
    # from confidence told reviewers that every high-confidence document had
    # been auto-accepted, including the ones being shown to them precisely
    # because the pipeline declined to.
    is_automated: Optional[bool] = None
    review_reasons: Optional[List[str]] = Field(default_factory=list)
    # ai_classify v2.1. Both are None on a document classified before the upgrade
    # AND during the window where the synced table has not picked the columns up
    # yet (the db layer substitutes NULL). Optional-with-None rather than a
    # default_factory, so the UI can tell "no rationale" from an empty one.
    #
    # classify_confidence is the RAW model score, on ai_classify's own scale:
    # measured 0.55-0.78 over 60 real documents, never near 1.0. It is NOT
    # comparable to confidence_score above, which gold computes by rescaling this
    # against classify_confidence_ceiling and blending it with three other
    # signals. Do not render the two through the same widget.
    classify_confidence: Optional[float] = None
    classify_rationale: Optional[str] = None


class ExtractionComparisonResponse(BaseModel):
    document_id: str
    document_name: str
    total_items: int
    items: List[ExtractionComparisonItem] = Field(default_factory=list)


class ExtractionVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class HumanReviewSubmission(BaseModel):
    verdict: ExtractionVerdict
    reasoning: Optional[str] = None
    # Field-level corrections keyed by stable identifier (e.g. "id:0", "id:7")
    # produced by parseIdentifiers in the reviewer UI. Sent only when the
    # reviewer touched the inline-editable extracted values.
    corrections: Optional[Dict[str, str]] = None


class HumanReviewResponse(BaseModel):
    id: StrUUID
    document_id: StrUUID
    reviewer_email: str
    verdict: ExtractionVerdict
    reasoning: Optional[str] = None
    is_automated: bool = False
    corrections: Optional[Dict[str, str]] = None
    created_at: datetime
    updated_at: datetime


class DocumentNoteSubmission(BaseModel):
    # Full replacement of the reviewer's private notepad for one document.
    # The client sends the whole text (debounced autosave), not a delta.
    note_text: str = ""


class DocumentNoteResponse(BaseModel):
    document_id: StrUUID
    user_email: str
    note_text: str = ""
    updated_at: Optional[datetime] = None


class ReviewDraftSubmission(BaseModel):
    """An UNSUBMITTED review, autosaved as the reviewer works.

    Every field is optional, unlike HumanReviewSubmission: a draft is partial by
    definition (verdict chosen but reasoning still empty, or corrections made
    before any verdict), and that partiality is exactly why it cannot be stored
    as a row in document_extraction_reviews.

    `verdict` is the same vocabulary as a real review, so a draft can never hold
    a value that would be rejected at submit time -- but it may be absent.
    """

    verdict: Optional[ExtractionVerdict] = None
    reasoning: Optional[str] = None
    corrections: Optional[Dict[str, str]] = None


class ReviewDraftResponse(BaseModel):
    document_id: StrUUID
    user_email: str
    verdict: Optional[ExtractionVerdict] = None
    reasoning: Optional[str] = None
    # Always an object, never null: the GET returns an empty draft rather than a
    # 404 when none exists, so the client has one shape to handle.
    corrections: Dict[str, str] = {}
    updated_at: Optional[datetime] = None
    # False on the empty-draft response, so the client can tell "no draft" from
    # "a draft that happens to be blank" without inspecting every field.
    exists: bool = False


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    databricks_connected: bool
    database_connected: bool
    details: Optional[Dict[str, Any]] = None


class AnalyticsSummaryResponse(BaseModel):
    total_reviews: int
    total_documents_reviewed: int
    correct_count: int
    partially_correct_count: int
    incorrect_count: int
    accuracy_pct: float
    auto_reviewed: int = 0
    human_reviewed: int = 0
    auto_accuracy_pct: float = 0.0
    human_accuracy_pct: float = 0.0
    human_documents_reviewed: int = 0
    auto_documents_reviewed: int = 0
    human_correct_count: int = 0
    human_partially_correct_count: int = 0
    human_incorrect_count: int = 0


class DurationStats(BaseModel):
    total: int
    avg_seconds: Optional[float] = None
    min_seconds: Optional[float] = None
    max_seconds: Optional[float] = None
    median_seconds: Optional[float] = None


class ProcessingMetricsResponse(BaseModel):
    pipeline: DurationStats
    review: DurationStats


class MonthlyTrendItem(BaseModel):
    review_month: str
    correct: int = 0
    partially_correct: int = 0
    incorrect: int = 0
    total: int = 0
    accuracy_pct: float = 0.0


class AnalyticsTrendResponse(BaseModel):
    trend: List[MonthlyTrendItem] = Field(default_factory=list)


class RecentReviewItem(BaseModel):
    id: StrUUID
    document_id: StrUUID
    document_name: str
    reviewer_email: str
    reviewer_display_name: Optional[str] = None
    verdict: str
    reasoning: Optional[str] = None
    is_automated: bool = False
    created_at: datetime


class RecentReviewsResponse(BaseModel):
    reviews: List[RecentReviewItem] = Field(default_factory=list)
    total_count: int = 0
    limit: int = 20
    offset: int = 0


class ReviewerOption(BaseModel):
    value: str
    label: str


class ReviewerListResponse(BaseModel):
    reviewers: List[ReviewerOption] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int


# ── agent-assisted review of held documents ────────────────────────────────


class RemediationCandidate(BaseModel):
    """One possible replacement for a flagged code."""

    code: str
    description: str = ""
    method: str


class RemediationItem(BaseModel):
    """What can be done about one flagged item on a held document.

    `candidates` is a shortlist, never a decision: `resolution` says how much
    judgment is still required, and `not_resolvable` means no value should be
    proposed at all.
    """

    review_reason: str
    resolution: str
    guidance: str
    field_name: Optional[str] = None
    observed_code: Optional[str] = None
    code_system: Optional[str] = None
    candidates: List[RemediationCandidate] = Field(default_factory=list)
    candidates_truncated: bool = False


class DocumentRemediationResponse(BaseModel):
    document_id: StrUUID
    is_held: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    items: List[RemediationItem] = Field(default_factory=list)


class HoldReasonItem(BaseModel):
    """One specific thing that is wrong with a document.

    ``severity`` is what a consumer branches on, never ``code``: an advisory
    finding is surfaced but does not hold the document, and promoting one to
    blocking must not require every reader to learn a new code.

    ``source`` says who established it — the pipeline, this app deriving it from
    the score, or nobody (``unexplained``). A reviewer being told *why we think
    so* is the difference between this and the banner it replaces, which asserted
    a cause it had not established.
    """

    code: str
    severity: str
    source: str
    title: str
    detail: str
    resolution: Optional[str] = None
    guidance: Optional[str] = None
    field_name: Optional[str] = None
    observed_code: Optional[str] = None
    code_system: Optional[str] = None
    candidates: List[RemediationCandidate] = Field(default_factory=list)
    candidates_truncated: bool = False
    confidence_score: Optional[float] = None
    threshold: Optional[float] = None
    count: Optional[int] = None


class DocumentHoldReasonsResponse(BaseModel):
    """Why this document is where it is — stated, never inferred.

    ``state`` is the pipeline's own routing decision. ``unknown`` is a real
    answer, returned when ``is_automated`` cannot be read, and the UI renders no
    verdict for it: the predecessor collapsed unknown into "held" and then
    invented a reason to match.

    ``blocking`` is never empty for a held document. When nothing accounts for
    the hold, it carries a single ``reason_not_recorded`` item saying exactly
    that, and ``unexplained`` is true — which makes the gap countable instead of
    papering over it with a plausible sentence.
    """

    document_id: StrUUID
    state: str
    is_automated: Optional[bool] = None
    confidence_score: Optional[float] = None
    auto_verdict_threshold: float
    reasons_recorded: bool = False
    unexplained: bool = False
    blocking: List[HoldReasonItem] = Field(default_factory=list)
    advisory: List[HoldReasonItem] = Field(default_factory=list)
    # Which optional reads failed, e.g. ["advisory_unavailable"]. An advisory
    # signal that could not be checked and one with nothing to report look
    # identical otherwise, and only one of them is reassuring.
    degraded: List[str] = Field(default_factory=list)


class RewrittenCode(BaseModel):
    """One code that stopped matching the page: what was printed, what we stored."""

    source_code: str
    stored_as: str


class DocumentFidelityResponse(BaseModel):
    """Whether this document's stored codes still match the source page.

    ``has_finding`` is false in every ordinary case — a faithful extraction, a
    real document with no planted ground truth, or an analytics pipeline that has
    not produced gold_extraction_fidelity yet — so the UI can render the badge on
    that single flag without distinguishing "fine" from "cannot tell".
    """

    document_id: StrUUID
    has_finding: bool = False
    fidelity_status: Optional[str] = None
    codes_rewritten: List[RewrittenCode] = Field(default_factory=list)
    # A clinical code was altered AND the document auto-verified anyway. This is
    # the combination worth interrupting a reviewer for.
    rewritten_and_auto_verified: bool = False


class ReviewProposal(BaseModel):
    id: StrUUID
    document_id: StrUUID
    review_reason: str
    resolution: str
    source: str
    disposition: str
    field_name: Optional[str] = None
    correction_key: Optional[str] = None
    observed_value: Optional[str] = None
    proposed_value: Optional[str] = None
    rationale: Optional[str] = None
    candidates: List[RemediationCandidate] = Field(default_factory=list)
    model: Optional[str] = None
    withheld: bool = False
    disposition_at: Optional[datetime] = None
    disposition_by: Optional[str] = None
    human_value: Optional[str] = None
    proposed_at: Optional[datetime] = None


class ReviewProposalListResponse(BaseModel):
    document_id: StrUUID
    proposals: List[ReviewProposal] = Field(default_factory=list)


class ReviewProposalSubmission(BaseModel):
    """A proposal the agent staged, recorded when the card is shown.

    `proposed_value` is ignored for a `not_resolvable` resolution — a refusal
    cannot carry a value, and the table constrains that too.
    """

    review_reason: str
    resolution: str
    field_name: Optional[str] = None
    correction_key: Optional[str] = None
    observed_value: Optional[str] = None
    proposed_value: Optional[str] = None
    rationale: Optional[str] = None
    candidates: List[RemediationCandidate] = Field(default_factory=list)
    model: Optional[str] = None


class ProposalDispositionSubmission(BaseModel):
    """What the reviewer did with a staged proposal.

    `human_value` is required for `modified` — the point of that disposition is
    to record what the reviewer used instead.
    """

    disposition: str
    human_value: Optional[str] = None
