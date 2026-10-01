"""What is wrong with one document, stated rather than inferred.

The reviewer app used to work out why a document was held by looking at what was
MISSING: an empty ``review_reasons`` was taken to mean "held on confidence
alone", and the banner said "confidence was below the threshold". Two things went
wrong with that, and this module exists to make both unrepresentable.

1. The inference was unfounded. ``review_reasons`` is empty whenever the pipeline
   had no *rule* to name, which is not the same as "the score was low". A
   document measuring 99.96% was told it was below 92%.
2. The verdict was computed from React props. ``ReviewPanel`` took
   ``reviewReasons`` and ``pipelineAutoVerified``; the only routed page never
   passed them, so they defaulted to ``[]`` and ``False`` and EVERY document
   rendered as held with the fabricated sentence. A component that can render a
   verdict from a prop a caller may forget will eventually render the wrong one.

So the verdict is assembled here, server-side, once, from the pipeline's own
columns — and the arithmetic sentence is reachable only when the arithmetic
actually holds. ``assess_document`` never returns an empty explanation for a held
document: if nothing accounts for the hold it says exactly that
(``reason_not_recorded``) instead of choosing a plausible cause.

Severity, not code, is what callers branch on. ``uncaptured_code`` and
``code_rewritten`` are real findings that do NOT gate auto-verification today;
promoting one to blocking is a change to :data:`ADVISORY_FINDINGS` here plus a
``review_reasons`` entry in ``gold_extraction_labels.sql``, and nothing else.

Candidate fixes are NOT derived here. :mod:`reviewer_app.services.remediation`
owns that, and its ``Remediation`` objects are mapped onto these items so the
guidance a reviewer reads and the guidance the agent proposes from cannot drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .remediation import (
    NEEDS_JUDGMENT,
    NOT_RESOLVABLE,
    REASON_INVALID_CODE,
    REASON_MISSING_MEMBER_ID,
    REASON_NON_BILLABLE,
    Candidate,
    Remediation,
)

# ── blocking reasons ────────────────────────────────────────────────────────
# The first three mirror gold_extraction_labels.sql as it already was. The next
# two are added there by this change, so that "held" means "carries at least one
# named reason" by construction rather than by coincidence.
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_CONFIDENCE_UNAVAILABLE = "confidence_unavailable"

# App-only. Never emitted by the pipeline; both are refusals to guess.
#   reason_not_recorded  — held, nothing explains it. A defect worth surfacing,
#                          not a blank space and not an invented cause.
#   unrecognized_reason  — the pipeline named something this build predates.
#                          Echoed verbatim; dropping it would hide a real flag.
REASON_NOT_RECORDED = "reason_not_recorded"
REASON_UNRECOGNIZED = "unrecognized_reason"

# ── advisory findings ───────────────────────────────────────────────────────
FINDING_UNCAPTURED_CODE = "uncaptured_code"
FINDING_CODE_REWRITTEN = "code_rewritten"

# The promotion seam. Membership here is the ONLY thing that keeps a finding out
# of the blocking list, so promoting one is a one-line change plus the pipeline.
ADVISORY_FINDINGS = frozenset({FINDING_UNCAPTURED_CODE, FINDING_CODE_REWRITTEN})

SEVERITY_BLOCKING = "blocking"
SEVERITY_ADVISORY = "advisory"

# Provenance, so the UI can distinguish "the pipeline said so" from "this app
# worked it out from the score" from "nothing did".
SOURCE_PIPELINE = "pipeline"
SOURCE_DERIVED_CONFIDENCE = "derived_from_confidence"
SOURCE_UNEXPLAINED = "unexplained"
SOURCE_PIPELINE_ADVISORY = "pipeline_advisory"
SOURCE_ANALYTICS = "analytics"

STATE_AUTO_VERIFIED = "auto_verified"
STATE_HELD = "held"
STATE_UNKNOWN = "unknown"

# Queue-chip precedence: hardest blocker first. A claim with no member id cannot
# be adjudicated at all, whereas an invalid code has a candidate fix; so the
# chip names the thing that most constrains what a reviewer can do.
#
# The defensible alternative is "most fixable first" (invalid_code before
# missing_member_id), which optimises reviewer throughput instead of severity.
# Changing the ordering is changing this tuple and nothing else.
_PRECEDENCE: Tuple[str, ...] = (
    REASON_MISSING_MEMBER_ID,
    REASON_INVALID_CODE,
    REASON_NON_BILLABLE,
    REASON_CONFIDENCE_UNAVAILABLE,
    REASON_LOW_CONFIDENCE,
    REASON_UNRECOGNIZED,
    REASON_NOT_RECORDED,
)

# Short labels. Also the queue chip's text, so they have to read well at two
# words: this is what replaces a flat "Pending" on a document card.
TITLES: Dict[str, str] = {
    REASON_INVALID_CODE: "Invalid code",
    REASON_NON_BILLABLE: "Non-billable code",
    REASON_MISSING_MEMBER_ID: "Missing member ID",
    REASON_LOW_CONFIDENCE: "Low confidence",
    REASON_CONFIDENCE_UNAVAILABLE: "No confidence score",
    REASON_NOT_RECORDED: "Reason not recorded",
    REASON_UNRECOGNIZED: "Unrecognised reason",
    FINDING_UNCAPTURED_CODE: "Code on the page was not captured",
    FINDING_CODE_REWRITTEN: "A code was silently rewritten",
}

KNOWN_BLOCKING = (
    REASON_INVALID_CODE,
    REASON_NON_BILLABLE,
    REASON_MISSING_MEMBER_ID,
    REASON_LOW_CONFIDENCE,
    REASON_CONFIDENCE_UNAVAILABLE,
)


def coerce_bool(value) -> Optional[bool]:
    """Coerce a possibly-stringified boolean, PRESERVING unknown as ``None``.

    Two hazards in one helper. The Statement Execution API serialises booleans
    as the strings ``"true"`` / ``"false"``, and ``bool("false")`` is True. And
    the frontend's ``record?.is_automated === true`` collapsed NULL, absent and
    ``"true"`` all to "held" — so a document whose routing was simply unknown
    was reported as held, with a reason invented to match.

    ``None`` in means ``None`` out: "we do not know" is a distinct answer from
    "no", and the caller renders it as no verdict rather than a wrong one.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "t", "1"):
            return True
        if text in ("false", "f", "0"):
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def format_pct(score: Optional[float]) -> Optional[str]:
    """Format a 0–1 score as a percentage that never overstates.

    FLOOR, not round, to two decimals. ``Math.round(0.9996 * 100)`` is 100, and
    that is how a 99.96% extraction came to be described as "100%" on the same
    line as "below the 92% threshold". This number answers "did it clear the
    bar", so rounding it up is the one direction that must not happen.
    """
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    floored = math.floor(value * 10000) / 100
    text = f"{floored:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


@dataclass(frozen=True)
class HoldReason:
    """One thing that is wrong with a document, ready to render."""

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
    candidates: Tuple[Candidate, ...] = field(default_factory=tuple)
    candidates_truncated: bool = False
    confidence_score: Optional[float] = None
    threshold: Optional[float] = None
    count: Optional[int] = None


@dataclass(frozen=True)
class HoldAssessment:
    """The whole answer to "what is wrong with this document"."""

    state: str
    is_automated: Optional[bool]
    confidence_score: Optional[float]
    threshold: float
    reasons_recorded: bool
    unexplained: bool
    blocking: Tuple[HoldReason, ...] = field(default_factory=tuple)
    advisory: Tuple[HoldReason, ...] = field(default_factory=tuple)
    degraded: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def primary(self) -> Optional[HoldReason]:
        """The one reason a queue chip shows. Blocking only — an advisory
        finding must never be what a card reports as the hold."""
        return _highest(self.blocking)


def _rank(code: str) -> int:
    try:
        return _PRECEDENCE.index(code)
    except ValueError:
        return len(_PRECEDENCE)


def _highest(items: Sequence[HoldReason]) -> Optional[HoldReason]:
    if not items:
        return None
    return sorted(items, key=lambda i: _rank(i.code))[0]


def _confidence_detail(
    confidence: Optional[float], threshold: float
) -> Tuple[str, str]:
    """Sentence for a low-confidence hold, plus the source that earned it.

    The ONLY place in this codebase that says "below the ... threshold". It is
    reachable only when the comparison literally holds; when a pipeline-recorded
    ``low_confidence`` disagrees with the stored score, this reports the
    disagreement rather than repeating the claim. That is what makes the
    contradiction in the original banner unrepresentable rather than merely
    fixed.
    """
    threshold_text = format_pct(threshold)
    if confidence is None:
        return (
            f"No confidence score was recorded, so the {threshold_text} "
            "auto-verify threshold could not be applied.",
            SOURCE_DERIVED_CONFIDENCE,
        )
    measured = format_pct(confidence)
    if confidence < threshold:
        return (
            f"Blended extraction confidence was {measured}, below the "
            f"{threshold_text} auto-verify threshold.",
            SOURCE_DERIVED_CONFIDENCE,
        )
    return (
        f"The pipeline recorded a low-confidence hold, but the stored score is "
        f"{measured} — at or above the {threshold_text} threshold. The routing "
        "decision and the score disagree; treat the score as unverified.",
        SOURCE_PIPELINE,
    )


def _from_remediation(rem: Remediation) -> HoldReason:
    """Map a Remediation onto a HoldReason, keeping its guidance verbatim.

    Deliberately no re-wording: the reviewer's sentence and the agent's
    proposal come from the same string, so they cannot drift apart.
    """
    code = rem.review_reason
    if code == REASON_MISSING_MEMBER_ID:
        detail = "No member or subscriber ID was found on this document."
    else:
        subject = rem.observed_code or "a code"
        system = rem.code_system or ""
        where = f" in {rem.field_name}" if rem.field_name else ""
        if code == REASON_INVALID_CODE:
            detail = (
                f"{subject} ({system}){where} is not in the reference " "terminology."
            )
        else:
            detail = f"{subject} ({system}){where} is a non-billable parent code."
    return HoldReason(
        code=code,
        severity=SEVERITY_BLOCKING,
        source=SOURCE_PIPELINE,
        title=TITLES.get(code, code),
        detail=detail,
        resolution=rem.resolution,
        guidance=rem.guidance,
        field_name=rem.field_name,
        observed_code=rem.observed_code,
        code_system=rem.code_system,
        candidates=tuple(rem.candidates),
        candidates_truncated=rem.candidates_truncated,
    )


def _generic_blocking(code: str) -> HoldReason:
    """A recorded reason with no per-code detail behind it.

    Happens when ``validated_codes`` is unavailable or disagrees with
    ``review_reasons``. The reason is still shown: a flag the pipeline raised
    must not vanish because the supporting detail could not be loaded.
    """
    details = {
        REASON_INVALID_CODE: (
            "The pipeline found a code that is not in the reference "
            "terminology. The specific code could not be resolved."
        ),
        REASON_NON_BILLABLE: (
            "The pipeline found a non-billable parent code. The specific code "
            "could not be resolved."
        ),
        REASON_MISSING_MEMBER_ID: (
            "No member or subscriber ID was found on this document."
        ),
    }
    return HoldReason(
        code=code,
        severity=SEVERITY_BLOCKING,
        source=SOURCE_PIPELINE,
        title=TITLES.get(code, code),
        detail=details.get(code, "The pipeline flagged this document."),
        resolution=(
            NOT_RESOLVABLE if code == REASON_MISSING_MEMBER_ID else NEEDS_JUDGMENT
        ),
    )


def _advisory_items(
    uncaptured: Optional[Dict], fidelity: Optional[Dict]
) -> List[HoldReason]:
    """Findings that are surfaced but do not gate auto-verification."""
    out: List[HoldReason] = []

    codes = list((uncaptured or {}).get("uncaptured_codes") or [])
    if codes:
        shown = ", ".join(str(c) for c in codes[:5])
        more = f" (+{len(codes) - 5} more)" if len(codes) > 5 else ""
        out.append(
            HoldReason(
                code=FINDING_UNCAPTURED_CODE,
                severity=SEVERITY_ADVISORY,
                source=SOURCE_PIPELINE_ADVISORY,
                title=TITLES[FINDING_UNCAPTURED_CODE],
                detail=(
                    f"{len(codes)} diagnosis code(s) appear in the page text but "
                    f"are not on the claim: {shown}{more}. Roughly one in eight "
                    "of these is printed for reference rather than assigned to "
                    "the patient, so confirm against the document."
                ),
                resolution=NEEDS_JUDGMENT,
                guidance=(
                    "Check whether each code was assigned to this patient. If it "
                    "was, add it; if it appears in a payer policy or reference "
                    "table, leave it off."
                ),
                count=len(codes),
            )
        )

    pairs = [
        p
        for p in ((fidelity or {}).get("codes_rewritten") or [])
        if isinstance(p, dict) and p.get("source_code") and p.get("stored_as")
    ]
    if pairs:
        shown = ", ".join(f"{p['source_code']} → {p['stored_as']}" for p in pairs[:5])
        out.append(
            HoldReason(
                code=FINDING_CODE_REWRITTEN,
                severity=SEVERITY_ADVISORY,
                source=SOURCE_ANALYTICS,
                title=TITLES[FINDING_CODE_REWRITTEN],
                detail=(
                    f"The document parser resolved a confusable character and "
                    f"stored a different code than the page shows: {shown}. It "
                    "reported full confidence in the character it changed."
                ),
                resolution=NEEDS_JUDGMENT,
                guidance=(
                    "Read the code off the page image and confirm which form is "
                    "correct before accepting the extraction."
                ),
                count=len(pairs),
            )
        )

    return out


def assess_document(
    *,
    is_automated,
    confidence_score: Optional[float],
    threshold: float,
    review_reasons: Optional[Sequence[str]] = None,
    remediations: Sequence[Remediation] = (),
    uncaptured: Optional[Dict] = None,
    fidelity: Optional[Dict] = None,
    degraded: Sequence[str] = (),
) -> HoldAssessment:
    """Assemble the full verdict for one document.

    ``is_automated`` is the pipeline's routing decision and the only thing that
    decides held vs auto-verified — never the confidence score on its own. An
    unknown value yields ``STATE_UNKNOWN`` and no blocking list, because the
    honest rendering of "we do not know" is no verdict at all.
    """
    automated = coerce_bool(is_automated)
    recorded = [str(r) for r in (review_reasons or []) if r]
    advisory = _advisory_items(uncaptured, fidelity)

    if automated is None:
        return HoldAssessment(
            state=STATE_UNKNOWN,
            is_automated=None,
            confidence_score=confidence_score,
            threshold=threshold,
            reasons_recorded=bool(recorded),
            unexplained=False,
            advisory=tuple(advisory),
            degraded=tuple(degraded),
        )

    if automated:
        return HoldAssessment(
            state=STATE_AUTO_VERIFIED,
            is_automated=True,
            confidence_score=confidence_score,
            threshold=threshold,
            reasons_recorded=bool(recorded),
            unexplained=False,
            advisory=tuple(advisory),
            degraded=tuple(degraded),
        )

    blocking: List[HoldReason] = []
    by_reason: Dict[str, List[Remediation]] = {}
    for rem in remediations or ():
        by_reason.setdefault(rem.review_reason, []).append(rem)

    for code in recorded:
        if code in (REASON_LOW_CONFIDENCE, REASON_CONFIDENCE_UNAVAILABLE):
            detail, source = _confidence_detail(confidence_score, threshold)
            blocking.append(
                HoldReason(
                    code=code,
                    severity=SEVERITY_BLOCKING,
                    source=source,
                    title=TITLES[code],
                    detail=detail,
                    resolution=NEEDS_JUDGMENT,
                    guidance=(
                        "Re-read every extracted value against the document "
                        "before confirming."
                    ),
                    confidence_score=confidence_score,
                    threshold=threshold,
                )
            )
        elif code in by_reason:
            blocking.extend(_from_remediation(r) for r in by_reason[code])
        elif code in KNOWN_BLOCKING:
            blocking.append(_generic_blocking(code))
        else:
            # Echo it. A reason this build does not know is still a reason the
            # pipeline raised, and hiding it is how a real flag goes missing.
            blocking.append(
                HoldReason(
                    code=REASON_UNRECOGNIZED,
                    severity=SEVERITY_BLOCKING,
                    source=SOURCE_PIPELINE,
                    title=TITLES[REASON_UNRECOGNIZED],
                    detail=(
                        f'The pipeline held this document for "{code}", which '
                        "this version of the app does not recognise. Review the "
                        "extraction manually and report the unknown reason."
                    ),
                    resolution=NOT_RESOLVABLE,
                    observed_code=code,
                )
            )

    unexplained = False
    if not blocking:
        # Held with nothing recorded. Derive ONLY what the numbers support, and
        # otherwise refuse: this is the branch that used to assert "below the
        # threshold" about a 99.96% document.
        if confidence_score is None:
            detail, source = _confidence_detail(None, threshold)
            blocking.append(
                HoldReason(
                    code=REASON_CONFIDENCE_UNAVAILABLE,
                    severity=SEVERITY_BLOCKING,
                    source=SOURCE_DERIVED_CONFIDENCE,
                    title=TITLES[REASON_CONFIDENCE_UNAVAILABLE],
                    detail=detail,
                    resolution=NEEDS_JUDGMENT,
                    threshold=threshold,
                )
            )
        elif confidence_score < threshold:
            detail, source = _confidence_detail(confidence_score, threshold)
            blocking.append(
                HoldReason(
                    code=REASON_LOW_CONFIDENCE,
                    severity=SEVERITY_BLOCKING,
                    source=SOURCE_DERIVED_CONFIDENCE,
                    title=TITLES[REASON_LOW_CONFIDENCE],
                    detail=detail,
                    resolution=NEEDS_JUDGMENT,
                    guidance=(
                        "Re-read every extracted value against the document "
                        "before confirming."
                    ),
                    confidence_score=confidence_score,
                    threshold=threshold,
                )
            )
        else:
            unexplained = True
            measured = format_pct(confidence_score)
            blocking.append(
                HoldReason(
                    code=REASON_NOT_RECORDED,
                    severity=SEVERITY_BLOCKING,
                    source=SOURCE_UNEXPLAINED,
                    title=TITLES[REASON_NOT_RECORDED],
                    detail=(
                        f"The pipeline routed this document to a reviewer but "
                        f"recorded no reason, and its confidence of {measured} "
                        f"is at or above the {format_pct(threshold)} threshold. "
                        "Nothing here explains the hold — review the extraction "
                        "on its merits and report this document."
                    ),
                    resolution=NOT_RESOLVABLE,
                    confidence_score=confidence_score,
                    threshold=threshold,
                )
            )

    return HoldAssessment(
        state=STATE_HELD,
        is_automated=False,
        confidence_score=confidence_score,
        threshold=threshold,
        reasons_recorded=bool(recorded),
        unexplained=unexplained,
        blocking=tuple(sorted(blocking, key=lambda i: _rank(i.code))),
        advisory=tuple(advisory),
        degraded=tuple(degraded),
    )


def primary_for_list(
    *,
    effective_status: Optional[str],
    is_automated,
    confidence_score: Optional[float],
    review_reasons: Optional[Sequence[str]],
    threshold: float,
) -> Optional[HoldReason]:
    """The queue chip's reason, from columns Lakebase already has.

    Same precedence and the same derivation as the detail page, so a card and
    the document it opens can never name different reasons. No warehouse read:
    ``is_automated``, ``confidence_score`` and ``review_reasons`` all live on
    the synced gold table, which the document list already joins.
    """
    if (effective_status or "").lower() != "pending":
        return None
    return assess_document(
        is_automated=is_automated,
        confidence_score=confidence_score,
        threshold=threshold,
        review_reasons=review_reasons,
    ).primary
