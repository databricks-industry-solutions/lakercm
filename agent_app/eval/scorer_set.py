"""
Shared scorer definitions for the LakeRCM agent.

The same scorers run in two places:
  - Production monitoring (`scorers.py register`) — sampled assessments on live traces
  - Offline evaluation (`run_eval.py`) — 100% coverage against a curated dataset

Keeping the definitions here means a new scorer (or a tightened guideline) lights
up in both places at once.
"""

from __future__ import annotations

from mlflow.entities import Feedback
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    Safety,
    Scorer,
    ToolCallCorrectness,
    scorer,
)

__all__ = [
    "BUILTIN_JUDGES",
    "DOMAIN_GUIDELINES",
    "CODE_SCORERS",
    "production_schedule",
    "offline_scorer_set",
    "no_sql_warehouse_regression",
    "latency_under_slo",
    "tool_call_budget",
    "policy_citations_grounded",
    "no_scaffolding_leak",
]


# ----------------------------- code-based scorers -----------------------------


@scorer
def no_sql_warehouse_regression(trace) -> Feedback:
    """Fail any trace that invoked the SQL Warehouse statement-execution API.

    The chat tools were migrated to Lakebase; re-introducing a Warehouse call
    is a regression.
    """
    spans = trace.data.spans or []
    offenders: list[str] = []
    for span in spans:
        name = (span.name or "").lower()
        if "warehouse" in name or "statementexecution" in name:
            offenders.append(span.name)
            continue
        attrs = span.attributes or {}
        target = str(attrs.get("http.url", "")) + str(attrs.get("db.system", ""))
        if "statements" in target.lower() or "sql/statements" in target.lower():
            offenders.append(span.name)

    if offenders:
        return Feedback(
            value=False,
            rationale=f"SQL Warehouse call detected in spans: {offenders[:3]}",
        )
    return Feedback(value=True)


@scorer
def latency_under_slo(trace) -> Feedback:
    """p95 SLO is 8s end-to-end; fail traces over 10s (hard ceiling)."""
    info = trace.info
    duration_ms = getattr(info, "execution_time_ms", None) or 0
    if duration_ms > 10_000:
        return Feedback(
            value=False,
            rationale=f"Turn exceeded 10s latency ceiling: {duration_ms}ms",
        )
    return Feedback(value=True)


@scorer
def tool_call_budget(trace) -> Feedback:
    """Hard cap: a single turn should not issue more than 8 tool calls."""
    spans = trace.data.spans or []
    tool_calls = [
        s for s in spans if (getattr(s, "span_type", "") or "").upper() == "TOOL"
    ]
    if len(tool_calls) > 8:
        return Feedback(
            value=False,
            rationale=f"Too many tool calls in a single turn: {len(tool_calls)}",
        )
    return Feedback(value=True)


# ----------------------------- domain guidelines -----------------------------

NO_RAW_TEXT_LEAK = Guidelines(
    name="no_raw_text_leak",
    guidelines=(
        "The response must NOT include raw OCR text dumps from documents, "
        "raw SQL queries, database table or column names, internal tool names, "
        "or verbatim stack traces. It should surface only synthesized, "
        "user-readable information."
    ),
)

PROFESSIONAL_TONE = Guidelines(
    name="professional_tone",
    guidelines=(
        "The response must address the user professionally (they are a claims "
        "reviewer) and present data in scannable structures (tables, bullets, "
        "bold for key values). No casual/filler preamble before the content."
    ),
)


# ----------------------------- groupings -----------------------------

# NOTE: This module is the PRODUCTION scorer surface — `production_schedule()`
# is registered as scheduled monitors on agent boot (eval/scorers.py). Keep it
# stable. Eval-only scorers (conditional correctness/groundedness, audit judge,
# offline/gepa/audit sets) live in `eval/eval_scorers.py` so eval changes can't
# churn the live monitor.

# Built-in MLflow GenAI judges. ToolCallCorrectness needs expected tool calls
# attached as expectations on each dataset record; the curation step wires
# these up from feedback Assessments where available.
BUILTIN_JUDGES: list[Scorer] = [
    Correctness(),
    RetrievalGroundedness(),
    Safety(),
    RelevanceToQuery(),
    ToolCallCorrectness(),
]

DOMAIN_GUIDELINES: list[Scorer] = [
    NO_RAW_TEXT_LEAK,
    PROFESSIONAL_TONE,
]

CODE_SCORERS: list[Scorer] = [
    no_sql_warehouse_regression,
    latency_under_slo,
    tool_call_budget,
]


def offline_scorer_set() -> list[Scorer]:
    """Flat scorer list for `mlflow.genai.evaluate(...)` — no sampling, all on.

    NOTE: offline eval / GEPA should prefer `eval.eval_scorers.offline_scorer_set`
    (conditional correctness/groundedness + audit). This raw set is retained for
    back-compat / any direct caller.
    """
    return [*BUILTIN_JUDGES, *DOMAIN_GUIDELINES, *CODE_SCORERS]


def production_schedule() -> list[tuple[Scorer, float]]:
    """(scorer, sample_rate) pairs for production scheduled monitoring.

    Sampling rates balance LLM-judge cost against signal:
      - Safety: 100% (any drop is a hard gate)
      - Code-based: 100% (cheap, deterministic)
      - Domain guidelines: 100% / 10% (one is a hard rule, one is style)
      - Other built-in judges: 10–20% (cost control)
    """
    return [
        (Correctness(), 0.1),
        (RetrievalGroundedness(), 0.2),
        (Safety(), 1.0),
        (RelevanceToQuery(), 0.2),
        (ToolCallCorrectness(), 0.2),
        (NO_RAW_TEXT_LEAK, 1.0),
        (PROFESSIONAL_TONE, 0.1),
        (no_sql_warehouse_regression, 1.0),
        (latency_under_slo, 1.0),
        (tool_call_budget, 1.0),
        # Deterministic, live-traffic-only controls (see the block below).
        (policy_citations_grounded, 1.0),
        (no_scaffolding_leak, 1.0),
    ]


# ============================================================================
# Retrieval-attribution + guard-scaffolding scorers (PRODUCTION)
# ============================================================================
# These live HERE rather than in eval_scorers because they are LIVE-TRAFFIC
# controls. Offline eval swaps every tool for a replay stub
# (run_eval -> trace_replay.replay_tools_from_fixtures), so the real
# retriever never emits a RETRIEVER span and the real guard never emits a
# fence — both scorers would be inert in eval. Only production exercises
# them.
#
# SELF-CONTAINED BY NECESSITY. Scheduled monitoring never imports this module.
# Registration serializes only each @scorer function's BODY, and the monitor
# re-executes that body (mlflow.genai.scorers.scorer_utils.recreate_function)
# in a namespace holding just `mlflow` and a few entity classes (Feedback,
# Trace, ...). Every import, regex and helper a production scorer uses must
# therefore live INSIDE its body: a module-level helper is a NameError on every
# sampled trace, which also silences the alert that reads the assessment. (The
# first version of both scorers called module-level helpers — fourth review.)
# tests/test_production_scorers.py enforces this for every @scorer here.
#
# DEPLOY COST: adding to this module re-registers the scheduled monitors, so
# in-flight assessments for the existing scorers can null out once. That is
# the price of these controls working at all; revert this block to undo.
#
# Both are deterministic and cheap, hence sampled at 1.0.


@scorer(name="policy_citations_grounded")
def policy_citations_grounded(inputs=None, outputs=None, expectations=None, trace=None):
    """A turn that retrieved payer policy must cite it, and cite nothing else.

    Deterministic companion to RetrievalGroundedness. The LLM judge asks "is this
    supported?"; this asks the cheaper, sharper question: "did the agent actually
    ATTRIBUTE it, and did it invent the attribution?"

    Scoped by construction: returns None unless the turn retrieved policy, so it
    never penalizes the status/statistics turns that make up most traffic. When
    policy WAS retrieved it FAILS
      1. no citation at all → an unattributed policy claim
      2. a cited id that was never retrieved → a FABRICATED citation, strictly
         worse than no citation because it looks authoritative
    and PASSES a valid citation, or an honest "the retrieved policies do not
    cover this", which the system prompt requires when the corpus lacks an
    answer (Vector Search always returns top-k, so retrieval happens anyway).

    Policy ids are the citation anchor (scripts/payer_policy_content.py); the
    citation_label embeds the id, so matching ids covers both spellings.

    Known limit: a turn that retrieved policy but answered without USING it
    (a denial explained from the extracted reason) also fails, because text
    alone cannot tell that from an uncited rule. That is why its alert is a
    sustained fail RATE rather than zero tolerance (obs_alerts.yml).
    """
    import re

    from mlflow.entities import Feedback

    if trace is None:
        return None

    policy_id = r"\bPOL-[A-Z]{2}-[A-Z0-9]+-\d+\b"

    # Which policy ids did this turn retrieve? Span outputs survive
    # serialization in several shapes (Document objects, dicts, a JSON string),
    # so scan the raw repr rather than assume one schema.
    retrieved = set()
    for span in getattr(getattr(trace, "data", None), "spans", None) or []:
        kind = str(getattr(span, "span_type", "") or "").upper()
        if kind != "RETRIEVER":
            attrs = getattr(span, "attributes", None) or {}
            kind = str(attrs.get("mlflow.spanType", "")).upper()
        span_outputs = getattr(span, "outputs", None)
        if kind == "RETRIEVER" and span_outputs is not None:
            retrieved.update(re.findall(policy_id, str(span_outputs)))
    if not retrieved:
        return None  # no policy retrieval on this turn: not applicable

    # The response as plain text.
    text = ""
    if isinstance(outputs, str):
        text = outputs
    elif outputs is not None:
        messages = outputs.get("messages") if isinstance(outputs, dict) else None
        parts = [
            m["content"]
            for m in (messages if isinstance(messages, list) else [])
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        ]
        text = "\n".join(parts) if parts else str(outputs)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]

    # An explicit "the retrieved policy does not answer this" declaration. The
    # subject must be the RETRIEVED policy text, not policy in general: "the
    # retrieved policies do not cover lumbar fusion" says the retrieved text is
    # silent (honest, passes), while "Veridane policy does not cover lumbar
    # fusion" is a COVERAGE DETERMINATION, a rule that must be cited. So every
    # branch requires a retrieved/returned/available qualifier, a failure to
    # FIND policy, or an inability to CITE. (An earlier version accepted any
    # negation, so an uncited rule that merely contained "does not include"
    # passed — second review.) Ambiguous coverage language fails, which asks for
    # a citation: the safer error for a claims tool.
    retrieved_policy = (
        r"(?:(?:retrieved|returned|available|provided|these|those|the\s+above)\s+"
        r"(?:\w+\s+){0,2}?(?:polic(?:y|ies)|guidance|criteria|excerpts?|results?)"
        r"|(?:polic(?:y|ies)|excerpts?|results?)\s+(?:i\s+|that\s+i\s+|we\s+)?"
        r"(?:retrieved|found|returned))"
    )
    not_covered = (
        # "the retrieved policies do not cover / don't mention / does not apply"
        retrieved_policy + r"\s+(?:\w+\s+){0,3}?(?:do(?:es)?\s+not|don.?t"
        r"|doesn.?t|did\s+not|didn.?t)\s+(?:\w+\s+){0,2}?(?:cover|address|apply"
        r"|mention|specify|include|contain|answer|discuss)"
        # "not covered by the policies I retrieved" (qualifier required)
        r"|not\s+(?:covered|addressed|mentioned)\s+(?:by|in)\s+(?:the\s+)?"
        + retrieved_policy
        # "I could not find a policy addressing ..."
        + r"|(?:could\s+not|couldn.?t|cannot|can.?t|unable\s+to|did\s+not"
        r"|didn.?t)\s+(?:find|locate|identify)\s+(?:\w+\s+){0,3}?"
        r"(?:polic(?:y|ies)|guidance|criteria)"
        # "No payer policy in the retrieved set addresses this"
        r"|\bno\s+(?:\w+\s+){0,2}?(?:polic(?:y|ies)|guidance|criteria)\s+"
        r"(?:\w+\s+){0,4}?(?:covers?|address(?:es)?|appl(?:y|ies)|mentions?"
        r"|was\s+found|were\s+found|matche?s?)"
        # "I cannot cite a policy for that"
        r"|\bi\s+(?:can(?:not|.?t)|am\s+unable\s+to)\s+cite\b"
    )

    def mentioned_only_as_not_found(pid):
        """Every sentence naming `pid` says THAT id was not found, and none
        uses it as authority.

        "I could not find POL-ZZ-FAKE-999" repeats an id without citing it;
        scoring it as fabricated would fail the honest answer to a question
        about a nonexistent policy (third review). The not-found wording must
        be bound to the id: a sentence-level match let "Per POL-VD-FUSION-777,
        prior auth is required, though I could not find the full policy text"
        pass with an invented citation (fourth review).
        """
        p = re.escape(pid)
        not_found = (
            r"(?:could\s+not|couldn.?t|cannot|can.?t|unable\s+to|did\s+not"
            r"|didn.?t)\s+(?:find|locate|identify|retrieve)\s+(?:\S+\s+){0,4}?"
            + p
            + r"|\bno\s+(?:record|match|mention)\s+(?:of|for)\s+(?:\S+\s+){0,2}?"
            + p
            + r"|\bno\s+such\s+polic(?:y|ies)\W+(?:\S+\s+)?"
            + p
            + r"|"
            + p
            + r"\W+(?:\S+\s+){0,2}?(?:is|was|are|were)\s+not\s+(?:found|among"
            r"|in\b|part\s+of|(?:a\s+)?(?:recognized|known|valid|real))"
            + r"|"
            + p
            + r"\W+(?:\S+\s+){0,2}?(?:does\s+not|doesn.?t)\s+exist"
        )
        authority = (
            r"(?:\bper|\bunder|according\s+to|as\s+stated\s+in|pursuant\s+to"
            r"|based\s+on|\bcit(?:es?|ing))\W+(?:\S+\s+){0,2}?" + p
            # a parenthetical citation: "(Veridane POL-VD-MRI-001)"
            + r"|\([^()]*"
            + p
            + r"[^()]*\)"
            + r"|"
            + p
            + r"\W+(?:requires?|states?|says|mandates?|specifies|covers?"
            r"|excludes?|allows?|permits?|limits?|denies|approves?)\b"
        )
        holding = [s for s in sentences if re.search(r"\b" + p + r"\b", s)]
        return bool(holding) and all(
            re.search(not_found, s, re.IGNORECASE)
            and not re.search(authority, s, re.IGNORECASE)
            for s in holding
        )

    mentioned = set(re.findall(policy_id, text))
    unretrieved = mentioned - retrieved
    denied = {pid for pid in unretrieved if mentioned_only_as_not_found(pid)}
    fabricated = unretrieved - denied
    if fabricated:
        return Feedback(
            value=False,
            rationale=(
                f"Response cites policy id(s) {sorted(fabricated)} that were NOT "
                f"retrieved on this turn (retrieved: {sorted(retrieved)}). A "
                "fabricated citation is worse than an uncited claim."
            ),
        )

    cited = mentioned & retrieved
    if cited:
        return Feedback(
            value=True,
            rationale=f"All cited policies {sorted(cited)} were retrieved on this turn.",
        )

    if denied or re.search(not_covered, text, re.IGNORECASE):
        return Feedback(
            value=True,
            rationale=(
                "No citation, but the response explicitly said the retrieved "
                "policies do not cover the question (or that the requested policy "
                "could not be found) — the required behavior when the corpus lacks "
                "an answer."
            ),
        )
    return Feedback(
        value=False,
        rationale=(
            f"Retrieved {len(retrieved)} payer policy document(s) but the response "
            "cites none of them and did not say the policies do not cover the "
            "question. Policy rules must be attributed."
        ),
    )


@scorer(name="no_scaffolding_leak")
def no_scaffolding_leak(outputs=None) -> Feedback:
    """The untrusted-content fence must never surface in the reviewer's answer.

    `guards.wrap_untrusted` injects a preamble + delimiters into tool payloads
    when injection is detected. If the model echoes that scaffolding, the
    reviewer sees internal machinery — a defect introduced by the guard, so it is
    checked explicitly rather than assumed away.
    """
    from mlflow.entities import Feedback

    # Flattened exactly as in policy_citations_grounded. Duplicated on purpose:
    # a production scorer body must be self-contained (see the block comment).
    text = ""
    if isinstance(outputs, str):
        text = outputs
    elif outputs is not None:
        messages = outputs.get("messages") if isinstance(outputs, dict) else None
        parts = [
            m["content"]
            for m in (messages if isinstance(messages, list) else [])
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        ]
        text = "\n".join(parts) if parts else str(outputs)

    markers = [
        "UNTRUSTED_DOCUMENT_CONTENT",
        "UNTRUSTED CONTENT extracted from a document",
        "[redacted-delimiter]",
    ]
    hit = [m for m in markers if m in text]
    if hit:
        return Feedback(
            value=False,
            rationale=f"Response leaked guard scaffolding to the user: {hit}",
        )
    return Feedback(value=True)
