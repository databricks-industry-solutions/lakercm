"""Deterministic remediation candidates for a held document's review reasons.

The documents pipeline decides *why* a document is held
(``gold_extraction_labels.review_reasons``: ``invalid_code``,
``non_billable_code``, ``missing_member_id``) but never proposes a fix. This
module derives the CANDIDATE fixes, and only the candidates — it deliberately
does not choose between them.

That split is the whole design. Candidate generation is a lookup against the
curated terminology and belongs in code, where it is deterministic and
testable. *Choosing* a candidate needs the document in front of you
(``M25.561`` is the right knee, ``M25.562`` the left) and belongs to the agent,
which proposes, and to the reviewer, who signs off. A resolver that guessed
here would manufacture confidence it has not earned.

The three reasons are not equally tractable, and the return value says so
rather than flattening them:

``invalid_code``
    An OCR corruption: a letter ``O`` read for a zero, or a truncated code.
    Recovering it is close to mechanical — normalize the confusable
    characters, or prefix-match the truncation.

``non_billable_code``
    A parent code billed where one of its billable children belongs. The
    candidate set is exact (the children are the terminology rows that extend
    the parent), but picking one usually needs clinical detail from the
    document.

``missing_member_id``
    Not resolvable, ever, and the only honest answer is to say so. A member ID
    cannot be derived from the terminology or inferred from context; inventing
    one is a compliance problem, not a convenience. Callers get
    ``NOT_RESOLVABLE`` and guidance to source it, never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ── review reasons (mirrors gold_extraction_labels.sql) ─────────────────────
REASON_INVALID_CODE = "invalid_code"
REASON_NON_BILLABLE = "non_billable_code"
REASON_MISSING_MEMBER_ID = "missing_member_id"

KNOWN_REASONS = (
    REASON_INVALID_CODE,
    REASON_NON_BILLABLE,
    REASON_MISSING_MEMBER_ID,
)

# ── resolution classes ──────────────────────────────────────────────────────
# How much judgment the caller has to supply. DETERMINISTIC does NOT mean
# "apply it": every proposal still goes to a human. It means the terminology
# left exactly one option, so the agent is confirming rather than choosing.
DETERMINISTIC = "deterministic"
NEEDS_JUDGMENT = "needs_judgment"
NOT_RESOLVABLE = "not_resolvable"

# ── candidate methods ───────────────────────────────────────────────────────
METHOD_OCR_NORMALIZE = "ocr_normalize"
METHOD_TRUNCATION = "truncation_prefix"
METHOD_OVERLONG = "overlong_prefix"
METHOD_BILLABLE_CHILD = "billable_child"

ICD10 = "ICD-10-CM"
CPT_HCPCS = "CPT/HCPCS"

# Characters an OCR pass confuses with digits. Applied only from the SECOND
# character onward: an ICD-10-CM code legitimately STARTS with a letter, so a
# blanket letter->digit pass turns the real code I10 into 110. The synthetic
# corpus plants exactly this case ("I1O" for "I10"), which a position-blind
# normalizer gets wrong in both directions.
_LETTER_TO_DIGIT = {
    "O": "0",
    "o": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "i": "1",
    "l": "1",
    "L": "1",
    "Z": "2",
    "z": "2",
    "S": "5",
    "s": "5",
    "G": "6",
    "b": "6",
    "T": "7",
    "B": "8",
    "g": "9",
    "q": "9",
}

# The reverse, for the first character only, where a digit was read for the
# letter that every ICD-10-CM and HCPCS code begins with.
_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}

# A truncated code can prefix a great many terminology rows. Past this, the
# candidate list stops being a shortlist a human can scan and becomes noise —
# callers get the truncated list plus `candidates_truncated`, so the UI can say
# "too ambiguous to shortlist" instead of rendering 40 options.
MAX_CANDIDATES = 8


@dataclass(frozen=True)
class TerminologyRow:
    """One curated terminology row.

    ``is_billable`` is meaningful for ICD-10-CM only; ``ref_cpt_hcpcs`` has no
    such column, so procedure rows are constructed with the default ``True``.
    """

    code: str
    description: str
    code_system: str
    is_billable: bool = True


@dataclass(frozen=True)
class Candidate:
    """One possible replacement for an observed code."""

    code: str
    description: str
    method: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "description": self.description,
            "method": self.method,
        }


@dataclass(frozen=True)
class Remediation:
    """What can be done about ONE flagged item on a held document."""

    review_reason: str
    resolution: str
    guidance: str
    field_name: Optional[str] = None
    observed_code: Optional[str] = None
    code_system: Optional[str] = None
    candidates: Tuple[Candidate, ...] = field(default_factory=tuple)
    candidates_truncated: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "review_reason": self.review_reason,
            "resolution": self.resolution,
            "guidance": self.guidance,
            "field_name": self.field_name,
            "observed_code": self.observed_code,
            "code_system": self.code_system,
            "candidates": [c.as_dict() for c in self.candidates],
            "candidates_truncated": self.candidates_truncated,
        }


# ── terminology indexing ────────────────────────────────────────────────────


class Terminology:
    """Indexed terminology, built once and reused across documents.

    Codes are matched case-insensitively and whitespace-trimmed, because the
    observed value comes from OCR. The dotted form is authoritative (the
    ``ref_icd10_cm`` DDL calls it "normalized upper-case, dot form"), so a
    dotless observation is also indexed to the same row.
    """

    def __init__(self, rows: Iterable[TerminologyRow]):
        self._by_system: Dict[str, List[TerminologyRow]] = {}
        self._lookup: Dict[Tuple[str, str], TerminologyRow] = {}
        for row in rows:
            code = (row.code or "").strip().upper()
            if not code:
                continue
            system = (row.code_system or "").strip() or ICD10
            canonical = TerminologyRow(
                code=code,
                description=row.description or "",
                code_system=system,
                is_billable=bool(row.is_billable),
            )
            self._by_system.setdefault(system, []).append(canonical)
            self._lookup[(system, code)] = canonical
            # A dotless spelling of the same code resolves to the same row.
            self._lookup.setdefault((system, code.replace(".", "")), canonical)
        for rows_for_system in self._by_system.values():
            rows_for_system.sort(key=lambda r: r.code)

    def get(self, code: str, code_system: str) -> Optional[TerminologyRow]:
        key = (code_system, (code or "").strip().upper())
        return self._lookup.get(key)

    def rows(self, code_system: str) -> Sequence[TerminologyRow]:
        return self._by_system.get(code_system, ())

    def __len__(self) -> int:  # pragma: no cover - diagnostics only
        return len(self._lookup)


# ── candidate derivation ────────────────────────────────────────────────────


def _normalize_observed(raw: str) -> str:
    """Strip the decoration OCR and claim forms add around a code."""
    value = (raw or "").strip().upper()
    # A modifier-suffixed procedure ("99213-25") is the base code plus a
    # modifier; silver_validate_codes looks up the base, so match that here.
    if "-" in value:
        value = value.split("-", 1)[0].strip()
    return value


def _truthy(value) -> bool:
    """Coerce a possibly-stringified boolean to bool.

    The Databricks Statement Execution API serialises struct-field booleans as
    the STRINGS ``"true"`` / ``"false"`` (JSON_ARRAY disposition), and
    ``bool("false")`` is True because any non-empty string is truthy. Reading
    validated_codes straight from the warehouse therefore made every code look
    valid AND non-billable, which silently suppressed every invalid_code
    remediation. Real Python bools (the unit tests, a Lakebase read) pass
    through unchanged.
    """
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _ocr_variants(code: str) -> List[str]:
    """Position-aware OCR corrections of `code`, most likely first.

    Two variants, in order: letters read for digits from the second character
    on (the common direction — ``I1O`` -> ``I10``), then a digit read for the
    leading letter (``0M54`` -> ``OM54``). Only variants that actually differ
    from the input are returned.
    """
    if not code:
        return []
    variants: List[str] = []

    head, tail = code[0], code[1:]
    fixed_tail = "".join(_LETTER_TO_DIGIT.get(ch, ch) for ch in tail)
    if fixed_tail != tail:
        variants.append(head + fixed_tail)

    if head in _DIGIT_TO_LETTER:
        variants.append(_DIGIT_TO_LETTER[head] + fixed_tail)

    # De-duplicate while keeping order.
    seen = set()
    ordered = []
    for v in variants:
        if v != code and v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _billable_children(
    parent: str, terminology: Terminology, code_system: str = ICD10
) -> List[Candidate]:
    """Billable terminology codes that extend `parent`.

    ICD-10-CM specificity is expressed in the code string, so a child is any
    billable code that starts with the parent and is longer — the same
    relationship the synthetic generator uses when it plants the quirk. There
    is no parent/child column to join on; `ref_icd10_cm` carries only
    ``is_billable``.
    """
    parent = _normalize_observed(parent)
    if not parent:
        return []
    return [
        Candidate(
            code=row.code,
            description=row.description,
            method=METHOD_BILLABLE_CHILD,
        )
        for row in terminology.rows(code_system)
        if row.is_billable and row.code != parent and row.code.startswith(parent)
    ]


def _invalid_code_candidates(
    observed: str, terminology: Terminology, code_system: str
) -> List[Candidate]:
    """Candidates for a code the terminology does not contain.

    Ordered by how much interpretation each method requires: an OCR
    substitution that lands on a real code first, then a truncation that a
    real code extends, then the rarer overlong reading.
    """
    candidates: List[Candidate] = []
    seen: set = set()

    def _add(code: str, description: str, method: str) -> None:
        if code in seen:
            return
        seen.add(code)
        candidates.append(Candidate(code=code, description=description, method=method))

    # 1. Confusable characters. An exact hit after normalizing is the single
    #    most likely reading, so it leads.
    for variant in _ocr_variants(observed):
        row = terminology.get(variant, code_system)
        if row and row.is_billable:
            _add(row.code, row.description, METHOD_OCR_NORMALIZE)

    # 2. Truncation: a real code the observation is a prefix of.
    if observed:
        for row in terminology.rows(code_system):
            if not row.is_billable or row.code == observed:
                continue
            if row.code.startswith(observed):
                _add(row.code, row.description, METHOD_TRUNCATION)

    # 3. Overlong: a real code that is a prefix of the observation (a stray
    #    trailing character survived extraction).
    if len(observed) > 2:
        for row in terminology.rows(code_system):
            if not row.is_billable or row.code == observed:
                continue
            if observed.startswith(row.code):
                _add(row.code, row.description, METHOD_OVERLONG)

    return candidates


def _classify(candidates: Sequence[Candidate]) -> str:
    if not candidates:
        return NOT_RESOLVABLE
    if len(candidates) == 1:
        return DETERMINISTIC
    return NEEDS_JUDGMENT


def _shortlist(candidates: List[Candidate]) -> Tuple[Tuple[Candidate, ...], bool]:
    truncated = len(candidates) > MAX_CANDIDATES
    return tuple(candidates[:MAX_CANDIDATES]), truncated


# ── public entry points ─────────────────────────────────────────────────────


def remediate_code(
    review_reason: str,
    observed_code: str,
    code_system: str,
    terminology: Terminology,
    field_name: Optional[str] = None,
) -> Remediation:
    """Candidates for one flagged code. Never chooses between them."""
    observed = _normalize_observed(observed_code)

    if review_reason == REASON_NON_BILLABLE:
        found = _billable_children(observed, terminology, code_system or ICD10)
        candidates, truncated = _shortlist(found)
        resolution = _classify(candidates)
        if resolution == NOT_RESOLVABLE:
            guidance = (
                f"{observed} is not billable at this specificity and the "
                "terminology lists no billable child for it. The correct code "
                "has to come from the clinical documentation — do not "
                "substitute a sibling code."
            )
        elif resolution == DETERMINISTIC:
            guidance = (
                f"{observed} is a parent code and is not billable. The "
                f"terminology has exactly one billable child, "
                f"{candidates[0].code}. Confirm it against the document before "
                "proposing it."
            )
        else:
            guidance = (
                f"{observed} is a parent code and is not billable. Its "
                f"billable children differ by clinical detail (site, "
                f"laterality or severity), so read the document and cite the "
                f"evidence for whichever child you propose."
            )
        return Remediation(
            review_reason=review_reason,
            resolution=resolution,
            guidance=guidance,
            field_name=field_name,
            observed_code=observed,
            code_system=code_system,
            candidates=candidates,
            candidates_truncated=truncated,
        )

    if review_reason == REASON_INVALID_CODE:
        found = _invalid_code_candidates(observed, terminology, code_system)
        candidates, truncated = _shortlist(found)
        resolution = _classify(candidates)
        if resolution == NOT_RESOLVABLE:
            guidance = (
                f"{observed} is not in the terminology and no correction of it "
                "lands on a real code. Treat it as unreadable and source the "
                "code from the document or the provider."
            )
        elif resolution == DETERMINISTIC:
            only = candidates[0]
            how = (
                "a confusable character"
                if only.method == METHOD_OCR_NORMALIZE
                else "a truncation"
            )
            guidance = (
                f"{observed} is not a valid code; {how} explains it, and "
                f"{only.code} is the only reading that resolves. Confirm it "
                "against the document."
            )
        else:
            guidance = (
                f"{observed} is not a valid code and more than one correction "
                "resolves. Use the document's own description of the service "
                "or diagnosis to choose, and say which evidence decided it."
            )
        return Remediation(
            review_reason=review_reason,
            resolution=resolution,
            guidance=guidance,
            field_name=field_name,
            observed_code=observed,
            code_system=code_system,
            candidates=candidates,
            candidates_truncated=truncated,
        )

    # Unknown reason: report it rather than inventing a remedy for it.
    return Remediation(
        review_reason=review_reason,
        resolution=NOT_RESOLVABLE,
        guidance=(f"No remediation is defined for review reason {review_reason!r}."),
        field_name=field_name,
        observed_code=observed or None,
        code_system=code_system,
    )


def remediate_missing_member_id() -> Remediation:
    """The deliberate refusal.

    A member ID is not derivable. The pipeline flags its absence; the remedy is
    to find it, not to produce it. This returns NOT_RESOLVABLE with no
    candidates so no caller can mistake it for something to apply.
    """
    return Remediation(
        review_reason=REASON_MISSING_MEMBER_ID,
        resolution=NOT_RESOLVABLE,
        guidance=(
            "No member, subscriber or insured ID was extracted from this "
            "document. A member ID cannot be derived from the terminology or "
            "inferred from the rest of the claim — do not propose a value. "
            "Check whether the ID is present in the document image but was "
            "missed by extraction; otherwise the document needs the ID from "
            "intake or the payer before it can be adjudicated."
        ),
    )


def remediate_document(
    review_reasons: Sequence[str],
    validated_codes: Sequence[Dict[str, object]],
    terminology: Terminology,
) -> List[Remediation]:
    """Every remediation for one held document, one per flagged item.

    ``validated_codes`` is the ``silver_validate_codes`` struct array carried
    on the gold row: each entry has ``field_name``, ``code``, ``raw_code``,
    ``code_system``, ``code_valid`` and ``is_non_billable``. A code can be
    flagged for at most one of the two code reasons — ``code_valid`` false
    means it is not in the terminology at all, so billability does not apply.

    De-duplicated per (reason, code): ai_extract repeats a code across every
    field that mentions it, and the reviewer needs one decision per code, not
    one per mention. The field names that carried it are kept so the UI can
    still point at them.
    """
    reasons = [r for r in (review_reasons or []) if r]
    out: List[Remediation] = []

    if REASON_MISSING_MEMBER_ID in reasons:
        out.append(remediate_missing_member_id())

    wants_invalid = REASON_INVALID_CODE in reasons
    wants_non_billable = REASON_NON_BILLABLE in reasons
    if not (wants_invalid or wants_non_billable):
        return out

    seen: set = set()
    for entry in validated_codes or []:
        if not isinstance(entry, dict):
            continue
        code = _normalize_observed(
            str(entry.get("code") or entry.get("raw_code") or "")
        )
        if not code:
            continue
        system = str(entry.get("code_system") or "").strip() or ICD10
        code_valid = _truthy(entry.get("code_valid"))
        non_billable = _truthy(entry.get("is_non_billable"))

        if wants_invalid and not code_valid:
            reason = REASON_INVALID_CODE
        elif wants_non_billable and code_valid and non_billable:
            reason = REASON_NON_BILLABLE
        else:
            continue

        key = (reason, system, code)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            remediate_code(
                review_reason=reason,
                observed_code=code,
                code_system=system,
                terminology=terminology,
                field_name=(entry.get("field_name") or None),
            )
        )

    return out
