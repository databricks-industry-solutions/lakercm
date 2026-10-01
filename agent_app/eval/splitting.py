"""Deterministic train/holdout splitting for the eval dataset.

Pure (hashlib + json only) so it is unit-testable without mlflow or the agent
graph. `eval.optimize` re-exports these names; GEPA (`optimize.optimize`) and
the promotion gate (`compare._holdout_records`) both call `_split_records`. On
the same dataset they agree exactly. The dataset can change between the two
jobs, though, so the gate does not rely on that alone: it also drops every key
optimize() recorded as trained on (eval/split_record.py).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _record_field(record: Any, field: str) -> Any:
    """Read a field from a record whether it's a dict or an attr-object."""
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _split_key(record: Any) -> str:
    """A STABLE identity for splitting — the user question text when available,
    else the JSON of `inputs`.

    Hashing the whole `inputs` is fragile: it includes the large `_tool_fixtures`
    blob, which the UC dataset round-trip (`to_df().to_dict()`) re-serializes
    differently than the authored Python dicts, shifting the hash buckets (a
    70/30 split landed 56/15 in-workspace). The user message is small and
    serialization-stable, so the split is reproducible and predictable.
    """
    inputs = getattr(record, "inputs", None)
    if inputs is None:
        inputs = _record_field(record, "inputs")
    if isinstance(inputs, dict):
        msgs = inputs.get("messages")
        if isinstance(msgs, (list, tuple)):
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                    return str(m["content"])
    return json.dumps(
        inputs if inputs is not None else record, sort_keys=True, default=str
    )


def key_digest(key: str) -> str:
    """SHA-256 of a split key, the form in which a split is RECORDED.

    Split keys are user questions, which can quote claim details, so a record
    of which keys were trained on stores digests, never the text.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def _split_records(
    records: list[Any], train_pct: int = 70, stratify_by: str | None = None
) -> tuple[list[Any], list[Any]]:
    """Deterministic train/holdout split by stable hash of the input.

    Sides are decided per split KEY (`_split_key`, the user question), by its
    content hash, and every record with a key goes to that key's side. Records
    sharing a key therefore always land together (see the leakage assertion
    below).

    With `stratify_by`, a one-sided stratum holding two or more distinct keys
    gets a key flipped to the other side: the key nearest the cut, preferring a
    key no other stratum holds. Earlier, strata were grouped and then ignored,
    so a class such as the adversarial probes could land entirely in train
    (sixth review). Flipping by KEY, not record, matters: moving one of two
    same-question records in different strata put the key on both sides and
    crashed the leakage guard, in optimize() and in the gate (seventh review).

    A key other strata also hold is flipped only if every one of those strata
    that has both sides keeps both sides. Without that check, balancing a later
    stratum could flip a key an earlier, already balanced one depended on, and
    leave THAT one-sided (eighth review). So the guarantee, stated exactly: a
    stratum with two or more keys ends with both sides unless every key it holds
    is shared and none can move without emptying a side elsewhere. (Choosing
    sides so that no set is one-sided is set splitting, which has no general
    greedy solution; question texts are almost always unique to one stratum, so
    the exception needs a stratum made entirely of repeated questions.)

    Stability: a key's side follows its hash, except a key flipped to balance a
    stratum, whose side depends on the rest of the dataset. See the module
    docstring for why the gate therefore also consults the recorded train keys.
    """

    def bucket_of(key: str) -> int:
        return int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "big") % 100

    keyed = [(_split_key(record), record) for record in records]
    is_train = {key: bucket_of(key) < train_pct for key, _record in keyed}

    if stratify_by and 0 < train_pct < 100:
        strata: dict[str, list[str]] = {}
        strata_of_key: dict[str, set[str]] = {}
        for key, record in keyed:
            tags = _record_field(record, "tags") or {}
            stratum = str(
                (tags.get(stratify_by) if isinstance(tags, dict) else None)
                or _record_field(record, stratify_by)
                or "_unstratified"
            )
            keys = strata.setdefault(stratum, [])
            if key not in keys:
                keys.append(key)
            strata_of_key.setdefault(key, set()).add(stratum)

        def keeps_both_sides(stratum: str, flipped: str) -> bool:
            """Whether `stratum` still has both sides once `flipped` moves.

            Also true for a stratum that was one-sided: moving one of its keys
            only ever gives it the side it lacked.
            """
            keys = strata[stratum]
            sides = {is_train[k] != (k == flipped) for k in keys}
            return len(keys) < 2 or len(sides) == 2

        for stratum in sorted(strata):
            keys = strata[stratum]
            sides = {is_train[k] for k in keys}
            if len(keys) < 2 or len(sides) == 2:
                continue
            # Nearest the cut first: the highest bucket when all are train, the
            # lowest when all are holdout. Own keys first: moving one cannot
            # affect another stratum.
            all_train = sides == {True}
            nearest = sorted(keys, key=lambda k: (bucket_of(k), k), reverse=all_train)
            candidates = [k for k in nearest if len(strata_of_key[k]) == 1] + [
                k for k in nearest if len(strata_of_key[k]) > 1
            ]
            for key in candidates:
                if all(keeps_both_sides(s, key) for s in strata_of_key[key]):
                    is_train[key] = not is_train[key]
                    break

    train = [record for key, record in keyed if is_train[key]]
    holdout = [record for key, record in keyed if not is_train[key]]

    # Leakage guard: a record must never appear in both splits.
    train_keys = {_split_key(r) for r in train}
    holdout_keys = {_split_key(r) for r in holdout}
    overlap = train_keys & holdout_keys
    assert not overlap, f"train/holdout leakage on {len(overlap)} record(s)"

    return train, holdout
