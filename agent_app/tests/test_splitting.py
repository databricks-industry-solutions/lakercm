"""Unit tests for the train/holdout splitter (eval/splitting.py).

GEPA trains on one side and the promotion gate scores the other, so the split
decides what the gate can see. Before the sixth review, `stratify_by` grouped
records by stratum and then assigned each by its own hash anyway, so a small
class (the adversarial probes) could land entirely in train and leave the gate
with no holdout row for it. These tests pin the guarantee that replaced it, and
that everything else stayed per-record and stable.

Pure (no mlflow). Run from agent_app/:
  python3 -m pytest tests/test_splitting.py
"""

from __future__ import annotations

import ast
import hashlib
import os
import sys
import unittest

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from eval.splitting import _split_key, _split_records  # noqa: E402

PCT = 60  # config.Settings.eval_train_pct default
KEY = "stratification_key"


def _rec(question: str, stratum: str) -> dict:
    return {
        "inputs": {"messages": [{"role": "user", "content": question}]},
        "tags": {KEY: stratum},
    }


def _bucket(question: str) -> int:
    return int.from_bytes(hashlib.sha1(question.encode()).digest()[:4], "big") % 100


def _questions(n: int, keep) -> list[str]:
    """The first n deterministic questions whose hash bucket satisfies `keep`."""
    out, i = [], 0
    while len(out) < n:
        q = f"question {i}"
        if keep(_bucket(q)):
            out.append(q)
        i += 1
    return out


def _keys(records) -> list[str]:
    return sorted(_split_key(r) for r in records)


class TestSplitRecords(unittest.TestCase):
    def test_all_train_stratum_moves_its_edge_row_to_holdout(self):
        qs = _questions(3, lambda b: b < PCT)
        train, holdout = _split_records([_rec(q, "rare") for q in qs], PCT, KEY)
        self.assertEqual((len(train), len(holdout)), (2, 1))
        nearest_the_cut = max(qs, key=lambda q: (_bucket(q), q))
        self.assertEqual(_split_key(holdout[0]), nearest_the_cut)

    def test_all_holdout_stratum_moves_its_edge_row_to_train(self):
        qs = _questions(3, lambda b: b >= PCT)
        train, holdout = _split_records([_rec(q, "rare") for q in qs], PCT, KEY)
        self.assertEqual((len(train), len(holdout)), (1, 2))
        nearest_the_cut = min(qs, key=lambda q: (_bucket(q), q))
        self.assertEqual(_split_key(train[0]), nearest_the_cut)

    def test_a_mixed_stratum_follows_the_hashes_exactly(self):
        below = _questions(2, lambda b: b < PCT)
        above = _questions(2, lambda b: b >= PCT)
        train, holdout = _split_records(
            [_rec(q, "mixed") for q in below + above], PCT, KEY
        )
        self.assertEqual(_keys(train), sorted(below))
        self.assertEqual(_keys(holdout), sorted(above))

    def test_a_single_row_stratum_is_not_forced(self):
        (q,) = _questions(1, lambda b: b < PCT)
        train, holdout = _split_records([_rec(q, "solo")], PCT, KEY)
        self.assertEqual((len(train), len(holdout)), (1, 0))

    def test_strata_are_balanced_independently(self):
        rare = _questions(3, lambda b: b < PCT)
        mixed_lo = _questions(5, lambda b: b < PCT)[3:]
        mixed_hi = _questions(2, lambda b: b >= PCT)
        records = [_rec(q, "rare") for q in rare] + [
            _rec(q, "mixed") for q in mixed_lo + mixed_hi
        ]
        train, holdout = _split_records(records, PCT, KEY)
        holdout_keys = set(_keys(holdout))
        self.assertEqual(len(holdout_keys & set(rare)), 1)
        self.assertEqual(holdout_keys - set(rare), set(mixed_hi))

    def test_unstratified_split_is_purely_per_record(self):
        qs = _questions(3, lambda b: b < PCT)
        train, holdout = _split_records([_rec(q, "rare") for q in qs], PCT)
        self.assertEqual((len(train), len(holdout)), (3, 0))

    def test_same_question_in_two_strata_lands_together(self):
        # Seventh review: flipping one of two same-question records put the key
        # on both sides and crashed the leakage assertion (optimize + the gate).
        low, high = sorted(_questions(2, lambda b: b < PCT), key=_bucket)
        records = [_rec(high, "smoke"), _rec(low, "smoke"), _rec(high, "generated")]
        train, holdout = _split_records(records, PCT, KEY)  # must not raise
        # `low` is the only key unique to "smoke", so it is the one flipped;
        # both `high` records stay together.
        self.assertEqual(_keys(holdout), [low])
        self.assertEqual(_keys(train), [high, high])

    def test_a_shared_key_flips_with_all_its_records(self):
        # When every key is shared, the flipped key moves in EVERY stratum.
        low, high = sorted(_questions(2, lambda b: b < PCT), key=_bucket)
        records = [
            _rec(high, "smoke"),
            _rec(low, "smoke"),
            _rec(high, "generated"),
            _rec(low, "generated"),
        ]
        train, holdout = _split_records(records, PCT, KEY)  # must not raise
        self.assertEqual(_keys(holdout), [high, high])
        self.assertEqual(_keys(train), [low, low])

    def test_a_key_unique_to_the_stratum_is_flipped_first(self):
        # Flipping a key another stratum also holds could unbalance THAT one.
        shared, own = sorted(_questions(2, lambda b: b < PCT), key=_bucket)[::-1]
        records = [
            _rec(shared, "a"),
            _rec(own, "a"),
            _rec(shared, "b"),
            _rec(_questions(3, lambda b: b >= PCT)[2], "b"),
        ]
        train, holdout = _split_records(records, PCT, KEY)
        self.assertIn(own, _keys(holdout))
        self.assertIn(shared, _keys(train))

    def test_balancing_never_unbalances_an_earlier_stratum(self):
        # Eighth review: "b" is all-holdout and every key it holds is shared, so
        # the old code flipped its lowest bucket, k2, to train. That emptied the
        # holdout side of "a", which had already been balanced.
        (k1,) = _questions(1, lambda b: b < PCT)
        k2, k4 = sorted(_questions(2, lambda b: b >= PCT), key=_bucket)  # k2 nearest
        (k6,) = _questions(2, lambda b: b < PCT)[1:]
        records = [
            _rec(k1, "a"),
            _rec(k2, "a"),
            _rec(k2, "b"),
            _rec(k4, "b"),
            _rec(k4, "c"),
            _rec(k6, "c"),
        ]
        train, holdout = _split_records(records, PCT, KEY)
        side = {_split_key(r): True for r in train} | {
            _split_key(r): False for r in holdout
        }
        # "a" and "c" keep both sides; "b" could only be fixed by breaking one.
        self.assertEqual({side[k1], side[k2]}, {True, False})
        self.assertEqual({side[k4], side[k6]}, {True, False})
        self.assertEqual({side[k2], side[k4]}, {False})

    def test_a_safe_shared_key_is_flipped_when_the_nearest_is_not(self):
        # Same shape, but "c" has a second holdout key, so moving k4 leaves "c"
        # with both sides: k4 is flipped even though k2 is nearer the cut.
        (k1,) = _questions(1, lambda b: b < PCT)
        k2, k4, k5 = sorted(_questions(3, lambda b: b >= PCT), key=_bucket)
        (k6,) = _questions(2, lambda b: b < PCT)[1:]
        records = [
            _rec(k1, "a"),
            _rec(k2, "a"),
            _rec(k2, "b"),
            _rec(k4, "b"),
            _rec(k4, "c"),
            _rec(k5, "c"),
            _rec(k6, "c"),
        ]
        train, holdout = _split_records(records, PCT, KEY)
        self.assertEqual(set(_keys(train)), {k1, k4, k6})
        self.assertEqual(set(_keys(holdout)), {k2, k5})

    def test_deterministic_and_leak_free(self):
        records = [_rec(f"q{i}", f"s{i % 3}") for i in range(30)]
        first = _split_records(records, PCT, KEY)
        second = _split_records(list(reversed(records)), PCT, KEY)
        self.assertEqual(_keys(first[0]), _keys(second[0]))
        self.assertEqual(set(_keys(first[0])) & set(_keys(first[1])), set())
        self.assertEqual(len(first[0]) + len(first[1]), len(records))


def _adversarial_questions() -> list[str]:
    """ADVERSARIAL_SET questions, read from source (eval.dataset imports mlflow)."""
    path = os.path.join(_AGENT_APP_DIR, "eval", "dataset.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        target = (
            getattr(node, "target", None)
            or (getattr(node, "targets", None) or [None])[0]
        )
        if getattr(target, "id", "") == "ADVERSARIAL_SET":
            return [case["q"] for case in ast.literal_eval(node.value)]
    raise AssertionError("ADVERSARIAL_SET not found in eval/dataset.py")


class TestAdversarialCoverage(unittest.TestCase):
    def test_the_probes_reach_both_sides_at_any_reasonable_train_pct(self):
        records = [_rec(q, "adversarial") for q in _adversarial_questions()]
        for pct in (50, 60, 70, 80, 90):
            train, holdout = _split_records(records, pct, KEY)
            self.assertTrue(train and holdout, f"one-sided at train_pct={pct}")


class TestTheGateExcludesWhatWasTrainedOn(unittest.TestCase):
    """Eighth review: optimize and promotion run as separate jobs, and curation
    can add rows in between. The split is re-derived at promotion, and a
    balancing flip depends on the rest of the dataset, so a key GEPA trained on
    could move into the gate's holdout. The gate now drops the recorded keys.
    """

    def _scenario(self):
        # optimize time: "rare" is all-holdout, so its nearest key is flipped
        # into train, and GEPA trains on it.
        near, far = sorted(_questions(2, lambda b: b >= PCT), key=_bucket)
        at_optimize = [_rec(near, "rare"), _rec(far, "rare")]
        # promotion time: curation added a train-side key to "rare".
        (added,) = _questions(1, lambda b: b < PCT)
        at_promotion = at_optimize + [_rec(added, "rare")]
        return near, at_optimize, at_promotion

    def test_the_re_derived_holdout_alone_can_hold_a_trained_key(self):
        near, at_optimize, at_promotion = self._scenario()
        trained, _ = _split_records(at_optimize, PCT, KEY)
        self.assertIn(near, _keys(trained))
        _, holdout_now = _split_records(at_promotion, PCT, KEY)
        self.assertIn(near, _keys(holdout_now), "scenario no longer leaks")

    def test_dropping_the_recorded_keys_closes_it(self):
        from eval.split_record import drop_trained, train_key_digests

        near, at_optimize, at_promotion = self._scenario()
        trained, _ = _split_records(at_optimize, PCT, KEY)
        record = set(train_key_digests(trained))
        _, holdout_now = _split_records(at_promotion, PCT, KEY)
        gate = drop_trained(holdout_now, record)
        self.assertNotIn(near, _keys(gate))
        self.assertEqual(set(_keys(gate)), set(_keys(holdout_now)) - {near})

    def test_the_record_holds_digests_not_question_text(self):
        from eval.split_record import train_key_digests

        digests = train_key_digests([_rec("Claim CLM-99812 for Jane Doe?", "s")])
        self.assertEqual(len(digests), 1)
        self.assertNotIn("Jane", digests[0])
        self.assertRegex(digests[0], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
