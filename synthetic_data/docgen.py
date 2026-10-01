"""Synthetic claims documents for LakeRCM: fictional facts, AI-written narrative.

Each document starts as a Scenario whose facts come from code, never from the
model, so they are fictional by construction and known in advance:

  * people, clinics and identifiers come from Faker. A clinic's specialty fits
    the service it bills, and addresses are in one of the 50 states or DC with
    a ZIP code from that state. Provider NPIs deliberately fail the NPI check
    digit, so none can be a real provider's, and member ids carry the payer's
    prefix;
  * payers come from FICTIONAL_PAYERS (scripts/payer_policy_content.py);
  * diagnosis and procedure codes come from the reference sets the pipeline
    validates against (scripts/seed_reference_data.py). A share of documents
    get a non-billable code, a malformed code or a missing member id on
    purpose, so they land in the human review queue;
  * prior authorizations and denials are built around one of the payer
    policies, and denials around a CARC reason the pipeline's denial coder maps.

Claude Opus 5.5 on the Foundation Model APIs writes the narrative around those
facts: the reason for referral, the clinical justification, the denial
rationale. Its output is rejected unless it is well-formed JSON and names no
real insurer. The Scenario is the ground truth; the notebook stores it next to
the PDFs so extraction can be scored against it.

Faker and reportlab (synthetic_data/requirements.txt) are imported only where
they are used, so this module imports without them.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import random
import re
from dataclasses import dataclass, field

from denial_code_content import CARC_RULES
from payer_policy_content import FICTIONAL_PAYERS, POLICIES
from seed_reference_data import CPT_HCPCS_ROWS, ICD10_ROWS

DEFAULT_MODEL = "databricks-claude-opus-5-5"

# Real insurers and public programs an invented policy must never be attributed
# to. The repository-wide hygiene test uses the same pattern.
REAL_INSURER_PATTERN = re.compile(
    r"\bblue ?(?:cross|shield)\b|\bbcbs\b|\bunited ?health\w*|\buhc\b|\boptum\b"
    r"|\baetna\b|\bcigna\b|\bhumana\b|\banthem\b|\belevance health\b"
    r"|\bkaiser permanente\b|\bcentene\b|\bmolina\b|\bmedicare\b|\bmedicaid\b",
    re.I,
)

# Document types, with the label silver_classify_label.sql should assign and
# the sections the model writes.
DOC_TYPES: dict[str, dict] = {
    "referral": {
        "label": "referral_workqueue",
        "title": "Specialist Referral",
        "issuer": "provider",
        "sections": ["Reason for referral", "Relevant history", "Requested service"],
    },
    "prior_authorization": {
        "label": "prior_authorization",
        "title": "Prior Authorization Request",
        "issuer": "provider",
        "sections": [
            "Requested service",
            "Clinical justification",
            "Supporting documentation",
        ],
    },
    "denial": {
        "label": "denial_management",
        "title": "Notice of Claim Denial",
        "issuer": "payer",
        "sections": ["Denial reason", "Policy basis", "Appeal rights"],
    },
    "explanation_of_benefits": {
        "label": "explanation_of_benefits",
        "title": "Explanation of Benefits",
        "issuer": "payer",
        "sections": ["Claim summary", "How we processed this claim", "Member notes"],
    },
    "lab_results": {
        "label": "lab_results",
        "title": "Laboratory Report",
        "issuer": "provider",
        "sections": ["Tests performed", "Results", "Interpretation"],
    },
    "clinical_notes": {
        "label": "clinical_notes",
        "title": "Clinical Progress Note",
        "issuer": "provider",
        "sections": ["Subjective", "Objective", "Assessment and plan"],
    },
}
DEFAULT_MIX: dict[str, float] = {
    "referral": 0.25,
    "prior_authorization": 0.2,
    "denial": 0.2,
    "explanation_of_benefits": 0.1,
    "lab_results": 0.1,
    "clinical_notes": 0.15,
}

# Procedure-code categories (seed_reference_data.CPT_HCPCS_ROWS) that fit a
# document not built around a policy; other types draw from every category.
_PROCEDURE_CATEGORIES = {
    "lab_results": ("Lab",),
    "clinical_notes": ("E&M",),
    "referral": ("E&M",),
}

# Deliberate problems for "review" documents: each one should lower the
# document's confidence or fail code validation, so it waits for a person.
#
# `missing_member_id` is deliberately NOT here. A claim document arriving with no
# member or subscriber ID at all is rare in practice, and generating it at the
# same rate as a coding error misrepresented the queue: it took 51% of the held
# documents on a 2,097-document corpus (293 of 575) — more than its 1-in-3 share,
# because it is also the fallback when a non-billable parent will not fit a
# document. Every one of those is unresolvable by construction, so the agent
# declined 293 of its 294 declines on a defect that is mostly a generator
# artifact, and the review queue looked far less tractable than the product is.
#
# The PIPELINE still detects it: gold_extraction_labels holds a document whose
# extraction has no member id, remediation still answers `not_resolvable`, and
# the agent is still told never to invent one. That path stays tested in
# agent_app/tests/test_review_remediation_tool.py. This is only about what the
# synthetic corpus manufactures.
QUIRKS = ("non_billable_code", "malformed_code")
# What a document takes instead when a non-billable parent code would not fit it
# (see _non_billable_target), so `review_share` still holds. With one quirk left
# this is deterministic rather than a draw, which is why the malformed share
# rises as the non-billable one falls.
_OTHER_QUIRKS = tuple(q for q in QUIRKS if q != "non_billable_code")
# OCR-style mistakes the reference sets can't contain: a letter O for a zero,
# and a truncated CPT code.
MALFORMED_CODES = (
    ("I1O", "Essential (primary) hypertension"),
    ("9921", "Office/outpatient visit, established patient"),
)
# The real code each corruption came from. Recorded in the manifest as
# expected_fix so a remediation PROPOSAL can be scored against the truth, not
# just against whether a reviewer clicked Approve — the two differ exactly where
# it matters, when a reviewer accepts a wrong suggestion. Keyed separately rather
# than as a third tuple element: MALFORMED_CODES entries are used directly as
# (code, description) pairs in the printed code lists.
MALFORMED_TRUE_CODE = {
    "I1O": "I10",  # letter O read for the zero
    "9921": "99213",  # truncated; the description is the low-complexity visit
}

# The kinds of practice that bill each procedure code, so the provider fits the
# service: the first batch had an emergency visit billed by a physical-therapy
# clinic. Keyed by code where a category is too broad (E&M covers both office and
# emergency visits), else by CPT_HCPCS_ROWS category.
_PRIMARY_CARE = ("Family Medicine", "Internal Medicine")
_SPECIALTIES_BY_CODE = {
    "99283": ("Emergency Physicians",),
    "99285": ("Emergency Physicians",),
    "45378": ("Gastroenterology Associates",),
    "20610": ("Orthopedic Associates",),
    "36415": ("Diagnostic Laboratory",),
}
_SPECIALTIES_BY_CATEGORY = {
    "E&M": _PRIMARY_CARE,
    "Cardiology": ("Cardiology Group",),
    "Radiology": ("Imaging Center",),
    "Lab": ("Diagnostic Laboratory",),
    "Immunization": _PRIMARY_CARE,
    "Therapy": ("Physical Therapy",),
    "Drug": ("Internal Medicine",),
    "Transport": ("Ambulance Services",),
}
_PROCEDURE_CATEGORY = {c: cat for c, _d, _t, cat in CPT_HCPCS_ROWS}


def _specialties_for(procedure_code: str) -> tuple[str, ...]:
    """The practices that fit a document's first procedure code."""
    if procedure_code in _SPECIALTIES_BY_CODE:
        return _SPECIALTIES_BY_CODE[procedure_code]
    category = _PROCEDURE_CATEGORY.get(procedure_code)
    # A malformed code ('9921') has no category; it is an office visit gone wrong.
    return _SPECIALTIES_BY_CATEGORY.get(category, _PRIMARY_CARE)


# The policies each document type can be built around: a prior authorization
# asks for a service a policy gates, and a denial may cite any policy.
_POLICY_TYPES_FOR = {
    "prior_authorization": ("prior_authorization", "medical_necessity"),
    "denial": ("prior_authorization", "medical_necessity", "coding_billing", "appeals"),
}
# The CARC reasons a denial under each kind of policy can give. Patient-
# responsibility adjustments (CARC 1-3: deductible, coinsurance, copay) are not
# denials, so they never appear.
_DENIAL_CATEGORIES = {
    "medical_necessity": (
        "medical_necessity",
        "frequency_benefit_limit",
        "non_covered",
        "documentation",
    ),
    "prior_authorization": ("authorization",),
    "coding_billing": (
        "coding_billing",
        "bundling",
        "eligibility",
        "coordination_of_benefits",
    ),
    "appeals": ("timely_filing", "duplicate"),
}


@dataclass(frozen=True)
class Scenario:
    """The facts of one synthetic document: its ground truth."""

    doc_id: str
    doc_type: str
    expected_label: str
    payer: str
    member_id: str | None
    patient_name: str
    date_of_birth: str
    # Surrogate keys tying a patient's documents (and one episode of care)
    # together. Randomly minted, NOT derived from any identifier, because these
    # are what reach the knowledge graph -- where the patient's name, date of
    # birth and member id must not appear. Excluded from facts() so they never
    # reach the model or get printed on a document: no real claim form carries
    # them.
    patient_key: str
    episode_key: str
    provider_name: str
    provider_npi: str
    facility_name: str
    facility_address: str
    payer_address: str
    service_date: str
    diagnosis_codes: tuple[tuple[str, str], ...]
    procedure_codes: tuple[tuple[str, str], ...]
    policy_id: str | None = None
    policy_citation: str | None = None
    denial_carc: str | None = None
    denial_reason: str | None = None
    difficulty: str = "clean"
    quirks: tuple[str, ...] = field(default_factory=tuple)
    # The code that SHOULD be on the document where a code quirk was planted:
    # the billable child the parent displaced, or the uncorrupted form of an OCR
    # corruption. None when the quirk has no code answer (missing_member_id
    # cannot be fixed) or nothing was planted.
    expected_fix: str | None = None

    def facts(self) -> dict:
        """The facts the model may use, as a JSON-ready dict."""
        out = dataclasses.asdict(self)
        out["diagnosis_codes"] = [
            {"code": c, "description": d} for c, d in self.diagnosis_codes
        ]
        out["procedure_codes"] = [
            {"code": c, "description": d} for c, d in self.procedure_codes
        ]
        for key in (
            "doc_id",
            "expected_label",
            "difficulty",
            "quirks",
            "patient_key",
            "episode_key",
        ):
            out.pop(key)
        return {k: v for k, v in out.items() if v not in (None, "", [])}


def fictional_npi(rng: random.Random) -> str:
    """A 10-digit NPI-shaped id whose check digit is deliberately wrong.

    Real NPIs end in a Luhn check digit computed over "80840" + the first nine
    digits. Using any other final digit guarantees the id belongs to nobody.
    """
    body = "1" + "".join(str(rng.randrange(10)) for _ in range(8))
    return body + str((_npi_check_digit(body) + 1) % 10)


def _npi_check_digit(first_nine: str) -> int:
    digits = [int(d) for d in "80840" + first_nine]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def is_valid_npi(npi: str) -> bool:
    return len(npi) == 10 and npi.isdigit() and _npi_check_digit(npi[:9]) == int(npi[9])


def _street_address(fake) -> str:
    """A street address in one of the 50 states or DC, with a ZIP from that state.

    Faker's address() also yields military APO boxes; state_abbr() includes the
    freely associated states unless told not to (the first batch had a Marshall
    Islands 'MH'); and zipcode() ignores the state.
    """
    state = fake.state_abbr(
        include_territories=False, include_freely_associated_states=False
    )
    return (
        f"{fake.street_address()}, {fake.city()}, "
        f"{state} {fake.zipcode_in_state(state)}"
    )


# A reference description can carry a note for the pipeline, not the page: M54.5
# is 'Low back pain (parent code, not billable)', which printed the planted
# problem on the document.
_REFERENCE_NOTE = re.compile(r"\s*\((?=[^)]*\b(?:billable|parent code)\b)[^)]*\)")


def _plain(description: str) -> str:
    """A code description as a clinic or payer would print it."""
    return _REFERENCE_NOTE.sub("", description).strip()


def _member_id(payer: str, rng: random.Random) -> str:
    letters = "".join(ch for ch in payer.upper() if ch.isalpha())[:3]
    return f"{letters}{rng.randrange(10**8, 10**9)}"


# Reason text for a CARC whose standard description the denial coder's keyword
# rule doesn't match ("...notification absent" has no auth term after "absent").
_REASON_TEXT = {"197": "Prior authorization was not obtained for this service"}


def _denial_rules(categories: tuple[str, ...] | None = None) -> list[tuple[str, str]]:
    """CARC (code, reason) pairs a denial can give, optionally only from the
    given categories, whose reason text the denial coder maps back."""
    out = []
    for code, desc, cat, pattern, _prio in CARC_RULES:
        reason = _REASON_TEXT.get(code, desc)
        if (
            cat != "patient_responsibility"
            and (categories is None or cat in categories)
            and re.search(pattern, reason.lower())
        ):
            out.append((code, reason))
    return out


def _billable_icd() -> list[tuple[str, str]]:
    return [(c, _plain(d)) for c, d, _cat, billable in ICD10_ROWS if billable]


def _non_billable_icd() -> list[tuple[str, str]]:
    return [(c, _plain(d)) for c, d, _cat, billable in ICD10_ROWS if not billable]


# Procedure codes whose service is about the region a non-billable parent code
# describes, so the parent reads as a real (if unspecific) reason for the
# service: lumbar spine MRI and therapeutic exercise for M54.5 "Low back pain".
# Deliberately narrow — an ICD category is too coarse, since knee pain and low
# back pain are both "Musculoskeletal".
_REGION_PROCEDURES = {
    "M54.5": ("72148", "97110"),  # lumbar MRI, therapeutic exercise
    "M25.56": ("20610", "97110"),  # major joint injection, therapeutic exercise
    "J45.9": ("71046",),  # chest imaging
}


def _non_billable_target(
    parent: str,
    diagnoses: list[tuple[str, str]],
    procedures: list[tuple[str, str]],
    policy: dict | None,
    *,
    dx_from_policy: bool = True,
) -> int | None:
    """Which diagnosis a non-billable parent code may replace, or None.

    Coding a parent instead of its child (M54.5 where M54.50 belongs) is a real
    coding error, but only on a document about that condition: low back pain on
    a colonoscopy prior authorization is not an error a reviewer would ever see.
    The parent fits when its own child is already there (the textbook case), when
    the governing policy names the parent, or when the service is about that
    region. None means this document should carry a different problem.

    `dx_from_policy=False` says the caller filled `diagnoses` by random sample
    because the policy contributed none (a few policies carry an empty
    related_codes). The textbook case is then withdrawn: a child of the parent
    being present is an accident of that sample, not something the document's
    policy implies, so planting the parent produces a document whose coding error
    contradicts the policy it cites.
    """
    for i, (code, _description) in enumerate(diagnoses):
        if dx_from_policy and code != parent and code.startswith(parent):
            return i
    named_by_policy = policy is not None and parent in {
        c.strip() for c in policy["related_codes"].split(",") if c.strip()
    }
    fits_service = any(
        code in _REGION_PROCEDURES.get(parent, ()) for code, _d in procedures
    )
    return 0 if (named_by_policy or fits_service) and diagnoses else None


def _codes_for_policy(policy: dict) -> tuple[list, list]:
    icd = {c: _plain(d) for c, d, _cat, _b in ICD10_ROWS}
    cpt = {c: _plain(d) for c, d, _t, _cat in CPT_HCPCS_ROWS}
    wanted = [c.strip() for c in policy["related_codes"].split(",") if c.strip()]
    return (
        [(c, icd[c]) for c in wanted if c in icd],
        [(c, cpt[c]) for c in wanted if c in cpt],
    )


# =============================================================================
# Patient families
# =============================================================================
# Documents are grouped into PATIENTS and, within a patient, into EPISODES of
# care. Before this, every document drew a fresh Faker name and a fresh member
# id, so 3,419 documents produced 3,418 distinct patients -- one document each.
# That makes a patient node in the knowledge graph worse than useless: every
# traversal from a patient finds exactly the document you started from, and a
# 1:1 patient:document node is k=1, so it carries more re-identification risk
# than having no patient node at all.
#
# Documents in one episode share the patient, the payer, the member id and a
# primary diagnosis, and carry ascending service dates -- so the graph has real
# structure to traverse, not just a label.

# Where each doc_type sits in the revenue-cycle lifecycle. Used only to ORDER an
# episode's documents, so a family reads as a sequence (referral, then notes,
# then the authorisation, then the denial, then the remittance) instead of an
# unordered bag. The mix itself is untouched: types are still drawn from
# DEFAULT_MIX exactly as before, then sorted.
_LIFECYCLE_RANK = {
    "referral": 0,
    "clinical_notes": 1,
    "lab_results": 2,
    "prior_authorization": 3,
    "denial": 4,
    "explanation_of_benefits": 5,
}

# Long tail, not uniform. Most patients have one or two documents and a few have
# many, which is how claim volume actually distributes -- a minority of patients
# drive most of it. A uniform 3-per-patient would be the wrong shape AND would
# hide the interesting case, which is the patient with a real history.
_EPISODES_PER_PATIENT = ((1, 0.70), (2, 0.22), (3, 0.08))
_DOCS_PER_EPISODE = ((1, 0.45), (2, 0.30), (3, 0.17), (4, 0.08))

# Patient 0 is a deterministic showcase: three episodes totalling eight
# documents. A pure long-tail draw over ~35 patients can easily top out at three,
# which would make the graph look thin for reasons that have nothing to do with
# the design. Having one guaranteed rich family also gives verification a known
# patient to point at.
_SHOWCASE_EPISODES = (3, 3, 2)


def _draw(rng: random.Random, table: tuple[tuple[int, float], ...]) -> int:
    return rng.choices([v for v, _ in table], weights=[w for _, w in table])[0]


def _plan_patient_episodes(count: int, rng: random.Random) -> list[tuple[int, int]]:
    """Assign each of `count` documents to a (patient index, episode index)."""
    shaped: list[list[int]] = []
    if count >= sum(_SHOWCASE_EPISODES):
        shaped.append(list(_SHOWCASE_EPISODES))
    while sum(sum(eps) for eps in shaped) < count:
        shaped.append(
            [
                _draw(rng, _DOCS_PER_EPISODE)
                for _ in range(_draw(rng, _EPISODES_PER_PATIENT))
            ]
        )
    plan: list[tuple[int, int]] = []
    for patient, episodes in enumerate(shaped):
        for episode, size in enumerate(episodes):
            for _ in range(size):
                if len(plan) >= count:
                    return plan
                plan.append((patient, episode))
    return plan


def _payer_policy_types(policies: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in policies:
        out.setdefault(p["payer"], set()).add(p["policy_type"])
    return out


def _payer_for_family(
    doc_types: list[str],
    by_payer: dict[str, set[str]],
    rng: random.Random,
) -> str:
    """Pick one payer the whole family can plausibly be billed to.

    A member id is payer-scoped, so every document in a family MUST share a
    payer -- otherwise the same patient appears under two member ids and the
    graph splits them back apart.

    Payer coverage is uneven (only 5 payers, and e.g. Solvara has neither a
    prior_authorization nor a medical_necessity policy), so the payer is chosen
    knowing the family's doc types: any payer that can support the most
    constrained type present. If none can, the caller still falls back to one of
    that payer's other policies -- a slightly off policy_type is a far smaller
    lie than a patient with two payers.
    """
    candidates = [p for p in by_payer if by_payer[p]]
    for doc_type in sorted(doc_types, key=lambda t: -len(_POLICY_TYPES_FOR.get(t, ()))):
        needed = _POLICY_TYPES_FOR.get(doc_type)
        if not needed:
            continue
        narrowed = [p for p in candidates if by_payer[p] & set(needed)]
        if narrowed:
            candidates = narrowed
    return rng.choice(candidates or list(by_payer))


def build_scenarios(
    count: int,
    *,
    seed: int,
    review_share: float = 0.3,
    mix: dict[str, float] | None = None,
    today: dt.date | None = None,
    id_prefix: str = "synthetic",
) -> list[Scenario]:
    """Deterministic scenarios for `count` documents (same seed, same facts)."""
    from faker import Faker

    rng = random.Random(seed)
    fake = Faker("en_US")
    fake.seed_instance(seed)
    today = today or dt.date.today()
    weights = mix or DEFAULT_MIX
    doc_types = [t for t in weights if t in DOC_TYPES]
    policies = [p for p in POLICIES if p["payer"] in FICTIONAL_PAYERS]
    billable, non_billable = _billable_icd(), _non_billable_icd()
    procedure_rows = CPT_HCPCS_ROWS

    # Policies that can carry a non-billable parent code, because they name one
    # in related_codes (_non_billable_target's `named_by_policy` case). Only two
    # of the fifteen do, which is why the quirk has to steer toward them; see the
    # steering note in the loop.
    _parents = {c for c, _d in non_billable}

    def _can_carry_a_parent(policy: dict) -> bool:
        """Whether _non_billable_target could plant a parent on this policy.

        Mirrors that function's first two cases: the policy names a parent
        outright, or it names one of a parent's children (the textbook swap,
        M54.50 -> M54.5). Checking only the first missed every policy that
        lists a child, which is most of them.
        """
        codes = {c.strip() for c in policy["related_codes"].split(",") if c.strip()}
        return any(
            parent in codes or any(c != parent and c.startswith(parent) for c in codes)
            for parent in _parents
        )

    nb_policy_ids = {p["policy_id"] for p in policies if _can_carry_a_parent(p)}

    # ---- Plan the families before building anything --------------------------
    # Three passes, because an episode's payer and its document ORDER both depend
    # on knowing every doc_type in it first.
    plan = _plan_patient_episodes(count, rng)
    # Types are still drawn from the mix, one per document slot, so DEFAULT_MIX is
    # preserved in expectation exactly as before -- families change the grouping
    # and the ORDER, never the distribution.
    slot_types = [
        rng.choices(doc_types, weights=[weights[t] for t in doc_types])[0] for _ in plan
    ]

    by_payer = _payer_policy_types(policies)
    episodes: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(plan):
        episodes.setdefault(key, []).append(i)

    # One identity per patient, reused by every document of theirs.
    patient_ids = sorted({pat for pat, _ in plan})
    patients: dict[int, dict] = {}
    for pat in patient_ids:
        mine = [slot_types[i] for i, (p_, _) in enumerate(plan) if p_ == pat]
        pay = _payer_for_family(mine, by_payer, rng)
        patients[pat] = {
            "payer": pay,
            "member_id": _member_id(pay, rng),
            "patient_name": f"{fake.first_name()} {fake.last_name()}",
            "date_of_birth": fake.date_of_birth(
                minimum_age=18, maximum_age=90
            ).isoformat(),
            # A RANDOM surrogate, never a hash of the name or member id. A code
            # derived from an identifier is pseudonymisation, not
            # de-identification -- and this key is what the knowledge graph
            # carries, where no patient identifier may appear.
            "patient_key": f"pt-{rng.getrandbits(48):012x}",
            # One condition threaded through the patient's documents, so their
            # graph neighbourhoods genuinely overlap on a DiagnosisCode node
            # rather than merely sharing a patient label.
            "primary_dx": rng.choice(billable),
        }

    # Order each episode by the revenue-cycle lifecycle and give it a surrogate
    # key plus a start date; documents then advance in time within it.
    order: list[dict] = []
    for key in sorted(episodes):
        idxs = sorted(
            episodes[key], key=lambda i: _LIFECYCLE_RANK.get(slot_types[i], 9)
        )
        episode_key = f"ep-{rng.getrandbits(48):012x}"
        start = today - dt.timedelta(days=rng.randint(20, 90))
        day = 0
        for seq, i in enumerate(idxs):
            if seq:
                day += rng.randint(2, 12)
            service = min(start + dt.timedelta(days=day), today - dt.timedelta(days=1))
            order.append(
                {
                    "doc_type": slot_types[i],
                    "patient": key[0],
                    "episode_key": episode_key,
                    "service_date": service.isoformat(),
                }
            )

    scenarios: list[Scenario] = []
    for n, slot in enumerate(order, 1):
        doc_type = slot["doc_type"]
        patient = patients[slot["patient"]]
        payer = patient["payer"]
        # The quirk is drawn BEFORE the policy so a non-billable parent can steer
        # which policy the document is built around. Drawn after (as it was), the
        # document's clinical content was already fixed, _non_billable_target
        # rejected the parent as implausible on most of them, and the quirk was
        # swapped for another — so non_billable_code survived on only 1-7% of
        # review documents instead of its intended third, and a 36-document batch
        # usually contained none at all. That silently dropped the most
        # instructive claims case (a parent code billed where its billable child
        # belongs) from the demo. Steering keeps the plausibility guard intact —
        # it changes which document carries the quirk, never where a parent code
        # may land.
        quirks: tuple[str, ...] = ()
        expected_fix: str | None = None
        if rng.random() < review_share:
            quirks = (rng.choice(QUIRKS),)

        policy = None
        if doc_type in _POLICY_TYPES_FOR:
            # The family's payer outranks the policy's, and is applied FIRST.
            # Previously the policy chose the payer, which cannot work once a
            # patient spans documents: their member id is payer-scoped, so a
            # second payer would split one patient into two.
            #
            # Order matters. Filtering by payer AFTER the non-billable steering
            # below silently undid it -- the steered policy would be dropped for
            # belonging to another payer, and a parent code then landed on a
            # policy that cannot plausibly carry one
            # (test_a_planted_parent_code_never_contradicts_its_policy catches
            # exactly this). Narrow to the payer, then steer within what is left.
            mine = [p for p in policies if p["payer"] == payer]
            fits = [
                p for p in mine if p["policy_type"] in _POLICY_TYPES_FOR[doc_type]
            ] or mine
            if "non_billable_code" in quirks:
                # Prefer a policy that names a non-billable parent. Empty when
                # this payer admits none, and then the swap below still runs.
                #
                # This must NOT be allowed to outrank the policy_type filter: a
                # prior authorization has to cite a policy that actually gates a
                # service (test_prior_authorizations_cite_a_policy_that_gates_a_
                # service), and letting the nb steering win breaks that.
                fits = [p for p in fits if p["policy_id"] in nb_policy_ids] or fits
            policy = rng.choice(fits or policies)

        diagnoses, procs = _codes_for_policy(policy) if policy else ([], [])
        # A policy may list a non-billable parent code (M54.5) to illustrate a
        # specificity rule; only a "review" document gets one, on purpose, below.
        diagnoses = [d for d in diagnoses if d in billable]
        # True when there is no policy to contradict at all, or when the policy
        # did contribute the diagnoses. False ONLY for the narrow case that
        # matters: a document that cites a policy whose related_codes yielded
        # nothing, so its clinical content is an unrelated random sample.
        #
        # Getting this wrong is expensive in a way that does not announce itself.
        # Setting it False for every policy-less document (referral, EOB, labs,
        # clinical notes -- 60% of the mix) withdrew the textbook plant from most
        # of the corpus and halved non_billable_code, from 31% of review
        # documents to 16%, quietly thinning out the most instructive claims
        # error in the demo.
        dx_from_policy = policy is None or bool(diagnoses)
        if not diagnoses:
            diagnoses = rng.sample(billable, k=rng.randint(1, 2))
        if not procs:
            wanted = _PROCEDURE_CATEGORIES.get(doc_type)
            fits = [
                (c, _plain(d))
                for c, d, _t, cat in procedure_rows
                if wanted is None or cat in wanted
            ]
            procs = rng.sample(
                fits, k=min(len(fits), 3 if doc_type == "lab_results" else 1)
            )

        if "non_billable_code" in quirks:
            # Try every parent, in random order, and plant the first one that
            # fits this document. Picking ONE parent up front and giving up when
            # it did not fit wasted the steering above: the policy was chosen to
            # carry *some* parent, so an independently drawn parent missed
            # roughly (n_parents - 1) / n_parents of the time, and widening the
            # terminology from one parent to three made the quirk RARER, not
            # commoner. Plausibility is unchanged — _non_billable_target still
            # decides where a parent may land.
            candidates = list(non_billable)
            rng.shuffle(candidates)
            planted: tuple[tuple[str, str], int] | None = None
            for candidate in candidates:
                at = _non_billable_target(
                    candidate[0],
                    diagnoses,
                    procs,
                    policy,
                    dx_from_policy=dx_from_policy,
                )
                if at is not None:
                    planted = (candidate, at)
                    break
            if planted is None:
                quirks = (rng.choice(_OTHER_QUIRKS),)
            else:
                parent, at = planted
                # Only the textbook case has a knowable answer. When
                # _non_billable_target displaced the parent's OWN child, that
                # child is what belongs and a remediation should arrive at it.
                # Its two other paths (the policy names the parent, or the
                # service is about that region) overwrite diagnosis 0 whatever
                # it was, so the displaced code is not the parent's child and
                # the document no longer determines which child is right — a
                # reviewer reading it could only pick one on clinical grounds.
                # expected_fix stays None there: those documents are still
                # scored for whether the proposal was ACCEPTED, just not for
                # whether it was CORRECT. Recording the displaced code anyway
                # would mark every correct proposal wrong.
                displaced = diagnoses[at][0]
                if displaced != parent and displaced.startswith(parent):
                    expected_fix = displaced
                diagnoses = [parent if i == at else d for i, d in enumerate(diagnoses)]
        if "malformed_code" in quirks:
            bad = rng.choice(MALFORMED_CODES)
            expected_fix = MALFORMED_TRUE_CODE.get(bad[0])
            if bad[0][0].isdigit():
                procs = [bad, *procs[1:]]
            else:
                diagnoses = [bad, *diagnoses[1:]]

        # Thread the patient's primary condition through every document of
        # theirs, AFTER the quirk block above so a planted bad code is never
        # displaced. This is what makes a family's graph neighbourhoods overlap
        # on a shared DiagnosisCode rather than only on the patient node.
        if patient["primary_dx"] not in diagnoses:
            diagnoses = [*diagnoses, patient["primary_dx"]][:4]

        carc = None
        if doc_type == "denial":
            allowed = _DENIAL_CATEGORIES.get(policy["policy_type"])
            carc = rng.choice(_denial_rules(allowed) or _denial_rules())
        clinic = f"{fake.last_name()} {rng.choice(_specialties_for(procs[0][0]))}"
        scenarios.append(
            Scenario(
                doc_id=f"{id_prefix}-{seed}-{n:04d}",
                doc_type=doc_type,
                expected_label=DOC_TYPES[doc_type]["label"],
                payer=payer,
                member_id=(
                    None if "missing_member_id" in quirks else patient["member_id"]
                ),
                patient_name=patient["patient_name"],
                date_of_birth=patient["date_of_birth"],
                patient_key=patient["patient_key"],
                episode_key=slot["episode_key"],
                provider_name=f"Dr. {fake.first_name()} {fake.last_name()}",
                provider_npi=fictional_npi(rng),
                facility_name=clinic,
                facility_address=_street_address(fake),
                payer_address=_street_address(fake),
                # Set by the episode plan, so a family advances in time.
                service_date=slot["service_date"],
                diagnosis_codes=tuple(diagnoses),
                procedure_codes=tuple(procs),
                policy_id=policy["policy_id"] if policy else None,
                policy_citation=policy["citation_label"] if policy else None,
                denial_carc=carc[0] if carc else None,
                denial_reason=carc[1] if carc else None,
                difficulty="review" if quirks else "clean",
                quirks=quirks,
                expected_fix=expected_fix,
            )
        )
    return scenarios


SYSTEM_PROMPT = (
    "You write realistic but entirely fictional US healthcare administrative "
    "documents for a software demonstration. Use only the people, organizations, "
    "identifiers, dates, codes and policy given to you. Never name a real insurer, "
    "hospital, clinician, patient or government program, and never invent new "
    "identifiers. Write in the plain, professional register of a real clinic or "
    "health plan. Reply with a single JSON object and nothing else."
)


def build_messages(scenario: Scenario) -> list[dict]:
    """Chat messages asking the model for the document's narrative sections."""
    spec = DOC_TYPES[scenario.doc_type]
    request = {
        "document_type": spec["title"],
        "facts": scenario.facts(),
        "sections": spec["sections"],
        "instructions": (
            "Write each section in 60 to 150 words, in the voice of the document's "
            "author: a clinician for referrals, requests and notes, the health plan for "
            "denials and explanations of benefits, the laboratory for lab reports. "
            "Refer to the patient, provider, clinic and payer by the names in `facts`. "
            "Mention every diagnosis and procedure code where it naturally belongs. Add "
            "the concrete, clinically plausible detail a real document carries, "
            "consistent with the codes: how long symptoms have lasted, treatments "
            "already tried, examination findings, vital signs, lab values with their "
            "reference ranges, and for an explanation of benefits the billed, allowed "
            "and paid amounts and the member's share. For a denial, explain the denial "
            "reason and cite the policy exactly as `policy_citation` gives it. Do not "
            "repeat the header facts as a list: they are printed separately. Write "
            "plain text, with no markdown; separate paragraphs with a blank line and "
            'put each list item on its own line starting with "- ".'
        ),
        "reply_format": {
            "sections": [{"heading": "<section name>", "body": "<text>"}],
            "summary": "<one sentence>",
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(request, indent=2)},
    ]


def reply_text(content) -> str:
    """A reply's text. Some models (Claude among them) return a list of content
    blocks instead of a string; only the text blocks count."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type", "text") == "text":
                parts.append(str(block.get("text") or ""))
        elif getattr(block, "type", "text") == "text":
            parts.append(str(getattr(block, "text", "") or ""))
    return "".join(parts)


def parse_narrative(text, scenario: Scenario) -> dict:
    """Validate the model's reply; raise ValueError if it can't be used."""
    raw = reply_text(text).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    if fence:
        raw = fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("reply has no JSON object")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not valid JSON: {exc}") from exc

    sections = data.get("sections")
    if not isinstance(sections, list) or len(sections) < 2:
        raise ValueError("reply needs at least two sections")
    cleaned = []
    for sec in sections:
        if not isinstance(sec, dict):
            raise ValueError("each section must be an object")
        heading, body = (
            str(sec.get("heading", "")).strip(),
            str(sec.get("body", "")).strip(),
        )
        if not heading or len(body) < 40:
            raise ValueError(f"section {heading!r} is empty or too short")
        cleaned.append({"heading": heading, "body": body})
    summary = str(data.get("summary", "")).strip()

    everything = " ".join([summary] + [s["heading"] + " " + s["body"] for s in cleaned])
    real = REAL_INSURER_PATTERN.search(everything)
    if real:
        raise ValueError(f"reply names a real insurer or program: {real.group(0)!r}")
    return {"sections": cleaned, "summary": summary}


def generate_narrative(
    workspace,
    scenario: Scenario,
    *,
    model: str = DEFAULT_MODEL,
    attempts: int = 3,
    temperature: float | None = None,
) -> dict:
    """Ask the model for the narrative; retry a reply that fails validation.

    temperature is sent only when given: Claude Opus 5.5 rejects the parameter,
    and the documents already differ through their facts.
    """
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    messages = [
        ChatMessage(role=ChatMessageRole(m["role"]), content=m["content"])
        for m in build_messages(scenario)
    ]
    options: dict = {"max_tokens": 2400}
    if temperature is not None:
        options["temperature"] = temperature
    last_error: Exception | None = None
    for _ in range(attempts):
        reply = workspace.serving_endpoints.query(
            name=model, messages=messages, **options
        )
        try:
            return parse_narrative(reply.choices[0].message.content, scenario)
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(
        f"{scenario.doc_id}: no usable reply after {attempts} attempts ({last_error})"
    )


_LIST_ITEM = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")

FOOTER = (
    "Synthetic document generated for the LakeRCM demo. All names, "
    "organizations and identifiers are fictional."
)


def render_pdf(scenario: Scenario, narrative: dict) -> bytes:
    """Lay the document out as a one- or two-page PDF and return its bytes."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from xml.sax.saxutils import escape

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle(
        "small",
        parent=body,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#444444"),
    )
    spec = DOC_TYPES[scenario.doc_type]

    bullet = ParagraphStyle("bullet", parent=body, leftIndent=16, bulletIndent=5)

    def para(text: str, style=body) -> Paragraph:
        return Paragraph(escape(text), style)

    def section_flowables(text: str) -> list:
        """Paragraphs split on blank lines; "- " / "* " / "1." lines as bullets."""
        out: list = []
        for block in re.split(r"\n\s*\n", text.strip()):
            prose: list[str] = []
            for line in block.splitlines():
                item = _LIST_ITEM.match(line)
                if item:
                    if prose:
                        out.append(para(" ".join(prose)))
                        prose = []
                    out.append(
                        Paragraph(
                            escape(line[item.end() :].strip()), bullet, bulletText="•"
                        )
                    )
                elif line.strip():
                    prose.append(line.strip())
            if prose:
                out.append(para(" ".join(prose)))
        return out

    def codes(pairs) -> Paragraph:
        return Paragraph(
            "<br/>".join(f"{escape(c)} — {escape(d)}" for c, d in pairs), body
        )

    rows = [
        ["Patient", para(scenario.patient_name)],
        ["Date of birth", para(scenario.date_of_birth)],
    ]
    if scenario.member_id:
        rows.append(["Member ID", para(scenario.member_id)])
    rows.append(["Payer", para(scenario.payer)])
    if spec["issuer"] == "payer":
        rows.append(["Provider", para(scenario.facility_name)])
    rows += [
        ["Rendering provider", para(scenario.provider_name)],
        ["Provider NPI", para(scenario.provider_npi)],
        ["Date of service", para(scenario.service_date)],
        ["Diagnosis codes", codes(scenario.diagnosis_codes)],
        ["Procedure codes", codes(scenario.procedure_codes)],
    ]
    if scenario.policy_citation:
        rows.append(["Payer policy", para(scenario.policy_citation)])
    if scenario.denial_reason:
        rows.append(
            [
                "Denial reason",
                para(f"CARC {scenario.denial_carc}: {scenario.denial_reason}"),
            ]
        )

    table = Table(rows, colWidths=[1.6 * inch, 5.0 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    if spec["issuer"] == "payer":
        letterhead = (f"{scenario.payer} Health Plan", scenario.payer_address)
    else:
        letterhead = (scenario.facility_name, scenario.facility_address)
    story = [
        Paragraph(escape(letterhead[0]), styles["Heading2"]),
        para(letterhead[1], small),
        Spacer(1, 10),
        Paragraph(escape(spec["title"]), styles["Title"]),
        table,
        Spacer(1, 12),
    ]
    for sec in narrative["sections"]:
        story.append(Paragraph(escape(sec["heading"]), styles["Heading3"]))
        story.extend(section_flowables(sec["body"]))
    if narrative.get("summary"):
        story += [Spacer(1, 8), para(f"Summary: {narrative['summary']}", small)]

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(0.75 * inch, 0.5 * inch, FOOTER)
        canvas.drawRightString(
            7.75 * inch, 0.5 * inch, f"{scenario.doc_id} · page {doc.page}"
        )
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(
        buf,
        pagesize=letter,
        title=spec["title"],
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.8 * inch,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def file_name(scenario: Scenario) -> str:
    return f"{scenario.doc_id}-{scenario.doc_type.replace('_', '-')}.pdf"


def lakehouse_document_path(path: str) -> str:
    """The key bronze_doc_parsed records for a document written to *path*.

    The generator writes to a UC volume path (``/Volumes/cat/schema/vol/x.pdf``),
    but the pipeline reads that volume with Spark, whose file listing returns the
    path WITH its scheme (``dbfs:/Volumes/...``). bronze_doc_parsed stores that
    string verbatim as ``document_path``, and it is the lakehouse's identity for
    the document all the way down: every silver/gold dataset joins on it, and the
    reviewer app derives each review id from ``uuid5(NAMESPACE_URL,
    document_path)``. It therefore cannot be reformatted to match the manifest.

    So the manifest records the pipeline's key alongside the path it actually
    wrote to, and grading extraction against ground truth is a plain equality
    join. Reconstructing it at query time instead (``replace(document_path,
    'dbfs:', '')``) is the trap this exists to remove: forget it and the join
    silently returns zero rows, which reads exactly like 0% extraction accuracy
    rather than like a broken join.
    """
    if "://" in path or path.startswith("dbfs:"):
        return path
    return f"dbfs:{path}" if path.startswith("/") else path


def manifest_row(scenario: Scenario, path: str, model: str) -> dict:
    """One ground-truth row for the manifest table."""
    return {
        "doc_id": scenario.doc_id,
        "file_path": path,
        "document_path": lakehouse_document_path(path),
        "doc_type": scenario.doc_type,
        "expected_label": scenario.expected_label,
        "difficulty": scenario.difficulty,
        "quirks": ",".join(scenario.quirks),
        # The code a remediation should arrive at where a code quirk was
        # planted, so a proposal can be scored for correctness and not only for
        # whether the reviewer accepted it.
        "expected_fix": scenario.expected_fix,
        "payer": scenario.payer,
        "member_id": scenario.member_id,
        "patient_name": scenario.patient_name,
        "date_of_birth": scenario.date_of_birth,
        # Ground truth for the patient graph. gold derives its own patient key
        # from the EXTRACTED member id (the only source that also works for a
        # real upload); these columns are what that derivation is checked against.
        "patient_key": scenario.patient_key,
        "episode_key": scenario.episode_key,
        "provider_npi": scenario.provider_npi,
        "service_date": scenario.service_date,
        "diagnosis_codes": ",".join(c for c, _ in scenario.diagnosis_codes),
        "procedure_codes": ",".join(c for c, _ in scenario.procedure_codes),
        "policy_id": scenario.policy_id,
        "denial_carc": scenario.denial_carc,
        "model": model,
    }
