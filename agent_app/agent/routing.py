"""
Complexity-tiered model routing for the LakeRCM agent.

Route each *conversation* to a model+effort tier chosen once at conversation
start and pinned for the conversation's life — the pattern Databricks' own Smart
Routing uses (classify once per session, pick the cheapest Pareto-frontier point
that clears the task, preserve prompt-cache reuse). Feature-flagged by
`settings.routing_enabled`; OFF → callers never touch this module and behavior is
today's single-endpoint blend.

Two dimensions:
  * MODEL tier  — Phase 2, needs the per-tier gateway model services
      (bundles/ai_gateway): settings.llm_endpoint_low / _med / _high. Empty
      (Phase 1) → the existing blend endpoint is used and only effort varies.
  * EFFORT      — always applied: low → settings.effort_low (low), med →
      settings.effort_med (medium), high → settings.effort_high (high). The blend
      NEVER receives none/low (Opus can emit tool calls as text); that is
      enforced in tier_to_endpoint_effort(), so LOW behaves exactly like MED
      until a dedicated low-tier service exists.

Safety (medical domain, human-in-the-loop review):
  * Classifier failure/timeout/store-miss → routing_default_tier, emitted with a
    `default_on_error` source so it is OBSERVABLE (never a silent downgrade).
  * Classifier answers but is UNPARSEABLE → resolves UP to high under its own
    `unparseable` source, so garbage never masquerades as a successful 'med'.
  * A high-risk signal forces HIGH regardless (upward-only re-pin) — the signal
    is a NARROW set of complexity/stakes cues, deliberately excluding
    domain-common nouns ("denial"/"appeal") that would route everything HIGH.
  * The pin only ever moves UP within a conversation (escalation), never down.

The classifier is GPT-5.6 Luna ("the fast, cost-efficient model in OpenAI's GPT-5.6
family") at MEDIUM reasoning effort, queried on /serving-endpoints — NOT the
gateway blend — so classifier_base_path is required
(the deployed app's llm_base_path is the gateway path). Heavy deps
(agent.llm.build_chat_openai, services.store.get_store) are imported lazily so
this module is cheap to import and easy to unit-test with stubs.
"""

from __future__ import annotations

import contextvars
import logging
import re
import time
from typing import Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

# Ordered so max()/comparison expresses "upward-only" re-pinning.
_TIER_RANK = {"low": 0, "med": 1, "high": 2}
_VALID_TIERS = ("low", "med", "high")

# LangGraph Store namespace for the thread→tier pin. Kept OUT of the per-user
# memory namespace on purpose — routing decisions are not user memories.
ROUTING_NAMESPACE = ("agent_routing",)

# Narrow complexity/stakes cues that force the HIGH tier on ANY turn (upward
# re-pin). Deliberately excludes domain-common terms ("denial", "appeal",
# "claim") — this is a *denial-management* product, so those would route
# essentially every conversation HIGH and erase the savings. These are
# cross-document / adjudication-reasoning signals instead. Tunable; a heuristic,
# not a security control.
#
# WORD-BOUNDARY matched (see high_risk_kind), not substring. Two cues were dropped
# for firing on ordinary claims vocabulary — a false positive is expensive here
# because the cue re-pins the whole thread upward:
#   * "audit"      — "audit trail" is everyday review language; same argument
#                    that (correctly) excluded "denial"/"appeal".
#   * "across the" — matched "across the queue"/"across the board"; narrowed to
#                    these|all|every, which actually imply multi-document work.
# Two DIFFERENT axes, tracked separately. Both force HIGH today, but they are
# not the same signal and should not be measured as one: COMPLEXITY says the
# reasoning is hard (a bigger model genuinely helps), STAKES says the answer
# matters (a bigger model helps less than a human would). Keeping them apart
# lets the eval attribute cost to each, and leaves room for stakes to earn a
# different response later (e.g. a review hand-off) instead of just more tokens.
_COMPLEXITY_PATTERNS = (
    r"reconcil\w*",  # reconcile / reconciling / reconciliation
    r"cross[- ]referenc\w*",
    r"across (?:these|all|every)",
    r"compare (?:these|all)",
    r"conflicting",
    r"discrepan\w*",  # discrepancy / discrepancies
    r"inconsisten\w*",  # inconsistent / inconsistency
)
_STAKES_PATTERNS = (
    r"fraud\w*",
    r"overpay\w*",
)


def _compile_cues(patterns: Tuple[str, ...]) -> "re.Pattern[str]":
    # Trailing \b matters: without it "compare all" fires on "compare allowances".
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)


_COMPLEXITY_RE = _compile_cues(_COMPLEXITY_PATTERNS)
_STAKES_RE = _compile_cues(_STAKES_PATTERNS)

# Tier used when the classifier answers but the answer cannot be parsed. The
# rubric tells the model "when uncertain, choose the HIGHER tier"; an
# unparseable answer is maximal uncertainty, so it resolves UP (and is reported
# under its own `unparseable` source — never as a successful classification).
_UNPARSEABLE_TIER = "high"

# Versioned classifier rubric. Bump _RUBRIC_VERSION when the wording changes so
# routing shifts are traceable. Kept terse — the classifier must answer in one
# word, and a short rubric keeps its reasoning (and first-token latency) small.
_RUBRIC_VERSION = "v1"
_CLASSIFIER_SYSTEM = (
    "You are a routing classifier for a medical-claims document-review "
    "assistant. Read the reviewer's message and rate the reasoning complexity "
    "it demands:\n"
    "- low: a single-fact lookup, greeting, or trivial question.\n"
    "- med: one step of reasoning, or a single document tool call.\n"
    "- high: multi-step reasoning, multiple tool calls, ambiguity, "
    "cross-document reconciliation, or a high-stakes judgment.\n"
    "When uncertain, choose the HIGHER tier. "
    "Reply with EXACTLY one word: low, med, or high."
)


# Reasoning efforts the classifier endpoint accepts. GPT-5.6 Luna on
# /serving-endpoints rejects anything else with 400 unsupported_value (verified
# 2026-09-23: "Supported values are: 'none', 'low', 'medium', 'high', and
# 'xhigh'" — note: NO 'minimal'). A bad configured value would therefore fail
# EVERY classification into default_on_error, so it is coerced, not passed on.
_CLASSIFIER_EFFORTS = ("none", "low", "medium", "high", "xhigh")

# Output budget for the classifier call. Reasoning tokens are billed as output
# AND count against this limit; if reasoning exhausts it the reply comes back
# with NO visible text (OpenAI reasoning guide). The old budget of 16 was sized
# for a non-reasoning model — at medium effort any message that triggers real
# reasoning would come back empty -> unparseable -> HIGH, i.e. silently route
# every hard-looking conversation to the dearest tier. Measured on Luna @ medium
# (2026-09-23): 0 reasoning tokens on typical routing messages, ~75 on a
# reasoning-heavy prompt. 1024 is >10x that, and the timeout still bounds cost.
_CLASSIFIER_MAX_TOKENS = 1024

# Efforts that must never reach the multi-model blend: at none/low Opus can
# emit tool calls as plain text. Dedicated tier services (Phase 2) are exempt —
# their rosters are chosen for the effort they run at.
_BLEND_UNSAFE_EFFORTS = frozenset({"none", "minimal", "low"})


def _classifier_effort() -> Optional[str]:
    """Configured classifier reasoning effort; None = omit the parameter.

    Empty disables it (required for a non-OpenAI classifier: Claude on
    /chat/completions 400s on reasoning_effort). An unsupported value falls back
    to 'medium' with a warning instead of failing every classification.
    """
    raw = getattr(settings, "classifier_reasoning_effort", "medium") or ""
    raw = raw.strip().lower()
    if not raw:
        return None
    if raw in _CLASSIFIER_EFFORTS:
        return raw
    logger.warning(
        "unsupported classifier reasoning effort %r; using 'medium' (supported: %s)",
        raw,
        ", ".join(_CLASSIFIER_EFFORTS),
    )
    return "medium"


def _classifier_timeout() -> float:
    """Classifier request timeout in seconds (settings, with a safe floor)."""
    raw = getattr(settings, "classifier_timeout_seconds", 3.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 3.0
    # A sub-second timeout would fail every call on a cold endpoint.
    return value if value >= 1.0 else 3.0


def _valid_tier(tier: Optional[str], fallback: str = "med") -> str:
    """Coerce arbitrary input to a valid tier, else fallback (also validated)."""
    if isinstance(tier, str) and tier.strip().lower() in _TIER_RANK:
        return tier.strip().lower()
    fb = fallback.strip().lower() if isinstance(fallback, str) else "med"
    return fb if fb in _TIER_RANK else "med"


def _higher(a: str, b: str) -> str:
    """Return the higher-complexity of two tiers (upward-only re-pin)."""
    return a if _TIER_RANK.get(a, 0) >= _TIER_RANK.get(b, 0) else b


def high_risk_kind(text: str) -> Optional[str]:
    """Which axis (if any) forces HIGH on this turn: 'complexity' | 'stakes'.

    Word-boundary matched, NOT substring: a plain ``in`` test fired inside
    longer words and phrases ("compare allowances" → "compare all"), and every
    false positive re-pins the conversation to the most expensive tier.

    Complexity is checked first: when a turn trips both, the reasoning load is
    the reason a bigger model helps, so that is the honest attribution.
    """
    if not text:
        return None
    if _COMPLEXITY_RE.search(text):
        return "complexity"
    if _STAKES_RE.search(text):
        return "stakes"
    return None


def is_high_risk(text: str) -> bool:
    """True when either axis fires. Thin wrapper over high_risk_kind()."""
    return high_risk_kind(text) is not None


# Content-block types that carry the model's reasoning rather than its answer.
# With a reasoning classifier these must never reach _parse_tier: a summary like
# "not high-stakes, so low" contains "high", and high is checked first.
_REASONING_BLOCK_TYPES = frozenset({"reasoning", "thinking", "redacted_thinking"})


def _content_to_text(content) -> str:
    """Coerce an LLM message .content (str, or list of content blocks) to text.

    Reasoning/thinking blocks are skipped: only the answer is parsed.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in _REASONING_BLOCK_TYPES:
                    continue
                parts.append(str(block.get("text") or block.get("content") or ""))
        return " ".join(parts)
    return str(content or "")


def _parse_tier(raw: str) -> Optional[str]:
    """Map a classifier response to a tier, or None when it is UNPARSEABLE.

    Order matters: high, then low, then med/medium. Anything else — empty, a
    truncated token, an error string, or prose — returns None so the caller can
    surface it. Folding that into 'med' and reporting it as a successful
    classification hid classifier breakage and contradicted the rubric's
    "when uncertain, choose the HIGHER tier".
    """
    t = (raw or "").strip().lower()
    if "high" in t:
        return "high"
    if "low" in t:
        return "low"
    if "med" in t:  # 'med' / 'medium'
        return "med"
    return None


async def classify_complexity(text: str) -> Optional[str]:
    """Classify one message into low/med/high via the classifier FM (GPT-5.6 Luna).

    Raises on any transport/model failure — callers (resolve_tier) convert that
    into the observable default tier. Returns None when the model answers but
    the answer is unparseable; resolve_tier reports that as its own
    `unparseable` source rather than as a successful classification.
    """
    from agent.llm import build_chat_openai  # lazy: heavy import, stubbed in tests

    effort = _classifier_effort()
    kwargs: dict = {}
    if effort:
        # Honored on this path: the endpoint validates the value, and a
        # reasoning-heavy prompt spends 75 reasoning tokens at medium vs 107 at
        # high (measured 2026-09-23). On ordinary routing messages Luna spends
        # none, so medium costs nothing extra on easy turns.
        kwargs["reasoning_effort"] = effort
    llm = build_chat_openai(
        base_path=settings.classifier_base_path,
        model=settings.classifier_endpoint,
        # Plain /chat/completions on a serving endpoint — NOT the gateway
        # /responses path. Do NOT add `temperature`: GPT-5.6 Luna rejects any
        # value but the default ("Only the default (1) value is supported").
        use_responses_api=False,
        max_tokens=_CLASSIFIER_MAX_TOKENS,
        # This call sits on the PRE-FIRST-TOKEN path of a new conversation, the
        # most latency-sensitive moment in a review UI. The old hardcoded 15s
        # with one retry meant a stalled classifier could add ~30s before the
        # reviewer saw anything. The fallback is cheap AND observable, so fail
        # fast instead: ~3s x 2 attempts. GPT-5.6 Luna @ medium measured
        # 0.75-0.98s on routing messages and 1.46s on a reasoning-heavy prompt
        # (2026-09-23), so 3s is still 2-4x the expected latency.
        timeout=_classifier_timeout(),
        max_retries=1,
        streaming=False,
        **kwargs,
    )
    resp = await llm.ainvoke(
        [
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": text or ""},
        ]
    )
    return _parse_tier(_content_to_text(getattr(resp, "content", "")))


def _blend_safe(endpoint: Optional[str], effort: str) -> str:
    """Never send an unsafe effort to the blend (endpoint None)."""
    if endpoint is None and effort in _BLEND_UNSAFE_EFFORTS:
        logger.warning(
            "effort %r is unsafe on the multi-model blend; using 'medium'", effort
        )
        return "medium"
    return effort


def tier_to_endpoint_effort(tier: str) -> Tuple[Optional[str], str]:
    """Map a tier to (llm_endpoint | None, reasoning_effort).

    endpoint None → use the existing blend (Phase 1, effort-only).

    LOW is a real tier only once a dedicated low-tier service is configured
    (llm_endpoint_low). Until then it resolves exactly like MED — today's
    behavior, byte-for-byte — because its effort ('low') is unsafe on the blend.
    Any effort the blend cannot take is clamped to 'medium' regardless of how the
    per-tier effort vars are set: that guarantee lives here, not in config.
    """
    tier = _valid_tier(tier)
    if tier == "high":
        ep, eff = settings.llm_endpoint_high or None, settings.effort_high or "high"
    elif tier == "low" and getattr(settings, "llm_endpoint_low", ""):
        ep = settings.llm_endpoint_low
        eff = getattr(settings, "effort_low", "") or "low"
    else:  # med, and low with no dedicated service
        ep, eff = settings.llm_endpoint_med or None, settings.effort_med or "medium"
    return ep, _blend_safe(ep, eff)


# --- Store-backed pin (thread → tier) ----------------------------------------
# get_store() RAISES on failure (only its .setup() is guarded internally), so
# every helper wraps it. On any Store failure we classify-and-use without
# persisting — never block the turn.


def _pin_is_expired(value: dict) -> bool:
    """True when a pin is older than routing_pin_ttl_seconds.

    A pin is a *decision with a shelf life*, not a life sentence. Without this,
    one escalation (or one false-positive cue) fixed an entire long-lived thread
    at the dearest tier for every later turn — including "thanks" and "next
    document" — which inverts the whole point of tiering. Upward-only still
    holds WITHIN a window; past the TTL the next turn gets a fresh decision.

    ttl <= 0 disables expiry. A pin with no `pinned_at` (written before this
    field existed) is treated as live, so enabling the TTL never invalidates
    in-flight conversations.
    """
    ttl = getattr(settings, "routing_pin_ttl_seconds", 0) or 0
    try:
        ttl = float(ttl)
    except (TypeError, ValueError):
        return False
    if ttl <= 0:
        return False
    pinned_at = value.get("pinned_at")
    if not isinstance(pinned_at, (int, float)):
        return False
    return (time.time() - float(pinned_at)) > ttl


def get_pinned_tier(thread_id: str) -> Optional[str]:
    if not thread_id:
        return None
    try:
        from services.store import get_store  # lazy: stubbed in tests

        item = get_store().get(ROUTING_NAMESPACE, thread_id)
        if item is None:
            return None
        value = getattr(item, "value", None) or {}
        if not isinstance(value, dict):
            return None
        tier = value.get("tier")
        if tier not in _TIER_RANK:
            return None
        if _pin_is_expired(value):
            logger.debug("routing pin expired (thread=%s); reclassifying", thread_id)
            return None
        return tier
    except Exception as e:
        logger.debug("routing pin read failed (thread=%s): %s", thread_id, e)
        return None


def pin_tier(thread_id: str, tier: str) -> None:
    """Persist thread→tier, upward-only. index=False so PostgresStore does NOT
    embed it (the store's index embeds a `content` field — we store none), and
    we NEVER persist the classifier's free-text reason (it can paraphrase PHI)."""
    if not thread_id:
        return
    tier = _valid_tier(tier)
    try:
        from services.store import get_store  # lazy: stubbed in tests

        store = get_store()
        existing = None
        item = store.get(ROUTING_NAMESPACE, thread_id)
        if item is not None:
            v = getattr(item, "value", None) or {}
            if isinstance(v, dict) and not _pin_is_expired(v):
                # An EXPIRED pin must not drag the new one upward — that would
                # make the TTL useless (the thread could never come back down).
                existing = v.get("tier")
        final = _higher(tier, existing) if existing in _TIER_RANK else tier
        # pinned_at restarts the shelf life on every (re-)decision.
        store.put(
            ROUTING_NAMESPACE,
            thread_id,
            {"tier": final, "pinned_at": time.time()},
            index=False,
        )
    except Exception as e:
        logger.debug("routing pin write failed (thread=%s): %s", thread_id, e)


# The decision for THIS turn, so the post-model hook can tell the client which
# tier actually ran. Set in the request handler before the graph runs, read
# inside it — the same contextvar pattern as tools.set_active_document_id, which
# exists because values set before the graph are visible to code inside it
# (including sync tools LangChain runs in an executor thread, which copies the
# context). It cannot be state: ag_ui_langgraph 0.0.35 filters STATE_SNAPSHOT
# down to {messages, tools}, so a custom state field never reaches the browser.
_routing_decision: contextvars.ContextVar[Optional[Tuple[str, str]]] = (
    contextvars.ContextVar("_routing_decision", default=None)
)


def set_routing_decision(tier: Optional[str], source: Optional[str]) -> None:
    """Record the resolved (tier, source) for this turn, or clear it with None."""
    if tier and source:
        _routing_decision.set((tier, source))
    else:
        _routing_decision.set(None)


def get_routing_decision() -> Optional[Tuple[str, str]]:
    """The (tier, source) resolved for this turn, or None when routing is off."""
    return _routing_decision.get()


def normalize_requested_tier(value) -> Optional[str]:
    """Read a UI tier selection. Returns a tier, or None for "no request".

    ``auto`` is the product's word for "let the classifier decide", so it maps to
    None and the caller behaves exactly as it did before a selector existed.
    Anything unrecognised also maps to None rather than raising: a stray value
    from an older client must not break a turn, and silently classifying is the
    same thing the user would have got anyway.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in ("", "auto"):
        return None
    if text in ("medium",):  # the UI says "Medium"; the tiers are named "med"
        return "med"
    return text if text in _VALID_TIERS else None


async def resolve_tier(
    thread_id: str, text: str, requested: Optional[str] = None
) -> Tuple[str, str]:
    """Resolve the tier for this turn. Returns (tier, source).

    source ∈ {complexity_cue, stakes_cue, user_override_escalated, user_selected,
    pinned, classified, unparseable, default_on_error} — tagged on the trace so a
    router stuck on the default, a classifier emitting garbage, or a reviewer
    whose choice keeps getting overridden is observable.

    Order:
      1. High-risk cue on THIS turn → HIGH (upward re-pin), even if pinned lower
         AND even if the reviewer explicitly asked for something cheaper. Safety
         outranks the selector by deliberate product decision; the source says
         which case it was so the override is countable rather than merely
         visible in a tooltip.
      2. An explicit tier from the UI → use it, and pin it so the rest of the
         thread follows. A selector that is quietly ignored is worse than none.
      3. An unexpired pin → reuse it (no reclassify — preserves cache + cost).
      4. First substantive turn, or an EXPIRED pin → classify, then pin.

    ``requested`` is the already-normalized tier (see normalize_requested_tier),
    or None for "auto", which reproduces the pre-selector behavior exactly.
    """
    kind = high_risk_kind(text)
    if kind:
        pin_tier(thread_id, "high")
        # Distinct source per axis so the eval can attribute cost to "the
        # reasoning was hard" vs "the answer mattered" — they warrant different
        # product responses, and one bucket hid which was driving spend.
        #
        # When this overrode a cheaper explicit choice, say so instead: the UI has
        # to explain why the control it is showing did not take effect, and the
        # rate at which that happens is the measure of whether the cue list is
        # too aggressive. The axis is still recoverable from the trace's
        # routing.cue attribute.
        if requested is not None and requested != "high":
            return "high", "user_override_escalated"
        return "high", f"{kind}_cue"

    if requested is not None:
        # Pin it: a mid-conversation change should carry forward, and without the
        # pin the next turn would silently fall back to the classifier.
        pin_tier(thread_id, requested)
        return requested, "user_selected"

    pinned = get_pinned_tier(thread_id)
    if pinned:
        return pinned, "pinned"

    try:
        classified = await classify_complexity(text)
        if classified is None:
            tier = _UNPARSEABLE_TIER
            source = "unparseable"
            logger.warning(
                "complexity classifier returned an unparseable tier; using %s",
                tier,
            )
        else:
            tier = classified
            source = "classified"
    except Exception as e:
        tier = _valid_tier(settings.routing_default_tier)
        source = "default_on_error"
        logger.warning(
            "complexity classifier failed; defaulting to %s tier: %s", tier, e
        )
    pin_tier(thread_id, tier)
    return tier, source
