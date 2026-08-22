"""
Tests for the ARC Lens instrumentation harness.

    python -m pytest tests/test_harness.py        (if pytest is available)
    python tests/test_harness.py                  (plain, no dependencies)

The GPU-shaped tests fall back to CPU automatically. The integration test needs
torch; it skips cleanly when torch is absent rather than failing.
"""

import itertools
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY_DIR = REPO / "python"
_WORKLOAD_SEQ = itertools.count()
sys.path.insert(0, str(PY_DIR))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# Pure logic — no torch required
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskHeuristic(unittest.TestCase):
    def setUp(self):
        import _arc_bootstrap
        self.risk = _arc_bootstrap._risk_score

    def test_nan_is_always_critical(self):
        score, label = self.risk([1.0, 1.0], 0.1, True)
        self.assertEqual(score, 1.0)
        self.assertEqual(label, "CRITICAL")

    def test_calm_run_is_low(self):
        score, label = self.risk([1.0, 0.9, 0.8, 0.7, 0.6], 0.5, False)
        self.assertEqual(score, 0.0)
        self.assertEqual(label, "LOW")

    def test_doubling_loss_raises_risk(self):
        score, _ = self.risk([1.0, 1.2, 1.5, 1.8, 2.5], 0.5, False)
        self.assertGreater(score, 0.0)

    def test_label_and_score_never_disagree(self):
        # The dashboard colours the badge from the label; a label that does not
        # match its own score is how a MEDIUM risk rendered green.
        for grad in (0.0, 5.0, 11.0, 60.0):
            for bad in (False, True):
                score, label = self.risk([1.0, 5.0, 9.0, 12.0, 30.0], grad, bad)
                expected = (
                    "CRITICAL" if score > 0.8 else
                    "HIGH" if score > 0.5 else
                    "MEDIUM" if score > 0.25 else "LOW"
                )
                self.assertEqual(label, expected, f"grad={grad} bad={bad} score={score}")


class TestShippedDataFiles(unittest.TestCase):
    """The committed JSON is evidence, so it has to actually parse.

    These files exist so a reader can check the numbers in the docs against the
    raw output. That only works if `json.load(open(path))` — what a reader or a
    script will actually type — succeeds. A UTF-8 BOM makes it raise
    JSONDecodeError even though the content is perfectly valid, which is how
    experiment_ab_sweep5_rank_rule.json shipped: unreadable by the one method
    anyone would use to verify it.

    This repo has form on encoding (a double-encoded UTF-8 regression in
    dashboard.html), so it is worth a standing check rather than a one-off fix.
    """

    def test_every_committed_json_loads_with_a_plain_open(self):
        docs = REPO / "docs"
        files = sorted(docs.glob("*.json"))
        self.assertTrue(files, "no JSON found under docs/ — has the layout moved?")
        for path in files:
            with self.subTest(path=path.name):
                with open(path) as fh:          # deliberately no encoding= override
                    try:
                        json.load(fh)
                    except json.JSONDecodeError as exc:
                        self.fail(f"{path.name} does not parse with a plain open(): {exc}")

    def test_no_committed_json_starts_with_a_byte_order_mark(self):
        for path in sorted((REPO / "docs").glob("*.json")):
            with self.subTest(path=path.name):
                self.assertFalse(
                    path.read_bytes().startswith(b"\xef\xbb\xbf"),
                    f"{path.name} begins with a UTF-8 BOM",
                )


class TestFiniteGuard(unittest.TestCase):
    """JSON cannot represent NaN or Infinity, and absent beats invented."""

    def setUp(self):
        import _arc_bootstrap
        self.finite = _arc_bootstrap._finite

    def test_passes_through_real_numbers(self):
        self.assertEqual(self.finite(1.5), 1.5)
        self.assertEqual(self.finite(0), 0.0)

    def test_drops_nan_and_inf(self):
        self.assertIsNone(self.finite(float("nan")))
        self.assertIsNone(self.finite(float("inf")))
        self.assertIsNone(self.finite(float("-inf")))

    def test_drops_unparseable(self):
        self.assertIsNone(self.finite(None))
        self.assertIsNone(self.finite("abc"))

    def test_output_is_json_serialisable(self):
        # json.dumps would happily emit bare NaN, which is invalid JSON and
        # crashes the extension host's JSON.parse.
        payload = {"v": self.finite(float("nan"))}
        self.assertEqual(json.loads(json.dumps(payload))["v"], None)


class TestStructuralDetector(unittest.TestCase):
    """Silent failures must be caught, and noise must not fire."""

    def _monitor(self):
        import _arc_bootstrap

        m = _arc_bootstrap.OptimizerMonitor.__new__(_arc_bootstrap.OptimizerMonitor)
        m.baseline = {}
        m.baseline_samples = 0
        m.structural_strikes = 0
        m.structural_kind = None
        # Structural checks are inert until the run is past its warmup, so the
        # step counter has to be there for any of this to be exercised.
        _arc_bootstrap.STATE.step = _arc_bootstrap.STRUCTURAL_WARMUP_STEPS
        return m, _arc_bootstrap

    def _settle_baseline(self, m, mod, advanced):
        for _ in range(mod.STRUCTURAL_BASELINE_SAMPLES):
            m.check_structural(advanced)

    def test_healthy_run_never_triggers(self):
        m, _ = self._monitor()
        for _ in range(20):
            self.assertIsNone(m.check_structural(
                {"gradient_entropy": 0.30, "effective_rank": 70.0, "weight_update_ratio": 0.001}
            ))

    def test_nothing_is_judged_before_the_warmup(self):
        """No verdict may be reached during the opening transient.

        Every structural signal moves by orders of magnitude in a run's first
        few dozen steps simply because the model goes from random to structured.
        Judging there makes normal early learning look like collapse — which is
        how the entropy rule took a healthy CIFAR run from 87.4% to chance
        accuracy at step 125.

        Capturing the baseline early is a separate question and the answer is
        the opposite one — see the test below. The two were conflated, and the
        conflation was a bug.
        """
        m, mod = self._monitor()
        mod.STATE.step = 1
        collapsed = {"gradient_entropy": 1e-9, "effective_rank": 0.1, "weight_update_ratio": 0.9}
        for _ in range(mod.STRUCTURAL_SUSTAIN + 5):
            self.assertIsNone(m.check_structural(collapsed),
                              "no structural verdict may be reached during warmup")

    def test_baseline_is_captured_before_the_warmup_ends(self):
        """A collapse that happens early must not become its own baseline.

        This is the bug that made the rank rule structurally blind. The baseline
        used to be taken from the first samples *after* step 200, so a run that
        died before then had its "healthy" reference measured on the corpse.
        Every subsequent ratio compared dead against dead and sat near 1.0.

        Measured on two real runs, the effect inverted the verdict — the dead
        arm scored better than the healthy one:

            arm               baseline   floor   floor / baseline
            lr=0.03  73.6%     28.312   27.940       98.69%
            lr=0.50  10.0%     25.207   25.136       99.72%   <- dead

        Only `effective_rank` is captured this early, and that is deliberate: it
        moves ~3% across the warmup, while gradient entropy moves four orders of
        magnitude. An early baseline is safe for the former and was catastrophic
        for the latter.
        """
        m, mod = self._monitor()
        mod.STATE.step = 1
        healthy_opening = {"effective_rank": 28.75}
        for _ in range(mod.STRUCTURAL_BASELINE_SAMPLES):
            m.check_structural(healthy_opening)
        self.assertAlmostEqual(m.baseline.get("effective_rank"), 28.75, places=4,
                               msg="the opening rank must be recorded as the reference")

        # Now collapse it, still inside the warmup, then let the warmup expire.
        mod.STATE.step = 1
        for _ in range(mod.STRUCTURAL_SUSTAIN + 5):
            m.check_structural({"effective_rank": 1.0})
        self.assertAlmostEqual(m.baseline.get("effective_rank"), 28.75, places=4,
                               msg="a post-collapse value must not overwrite the baseline")

        mod.STATE.step = mod.STRUCTURAL_WARMUP_STEPS
        result = None
        for _ in range(mod.STRUCTURAL_SUSTAIN):
            result = m.check_structural({"effective_rank": 1.0})
        self.assertIsNotNone(result, "a collapse that began before the warmup must still be caught")
        self.assertEqual(result[0], "representation_collapse")

    def test_a_collapsed_entropy_alone_never_triggers_anything(self):
        """The second removed rule, locked out.

        `gradient_entropy_collapse` fired below 1% of an opening baseline. It
        took a healthy run to chance accuracy, and measuring the trajectory
        afterwards showed why no threshold could have rescued it: a healthy run
        (lr=0.25, 87.4%) and a dead one (lr=0.5, 10%) both settle to the *same*
        value, ~1.45e-05, from around step 70 onward. The upstream estimator bins
        a heavy-tailed distribution linearly and saturates for any run.
        """
        m, mod = self._monitor()
        healthy = {"gradient_entropy": 0.30, "effective_rank": 70.0, "weight_update_ratio": 0.001}
        self._settle_baseline(m, mod, healthy)

        for entropy in (1e-3, 1e-5, 1.45e-05, 1e-9):
            collapsed = {"gradient_entropy": entropy, "effective_rank": 70.0,
                         "weight_update_ratio": 0.001}
            for _ in range(mod.STRUCTURAL_SUSTAIN + 3):
                self.assertIsNone(
                    m.check_structural(collapsed),
                    f"gradient entropy {entropy} must not trigger an intervention",
                )

    def test_single_bad_sample_is_treated_as_noise(self):
        m, mod = self._monitor()
        healthy = {"gradient_entropy": 0.30, "effective_rank": 70.0, "weight_update_ratio": 0.001}
        self._settle_baseline(m, mod, healthy)
        bad = {"effective_rank": 20.0}
        self.assertIsNone(m.check_structural(bad))
        self.assertIsNone(m.check_structural(healthy))  # recovers, strike resets
        self.assertIsNone(m.check_structural(bad))

    def test_a_high_update_ratio_alone_never_triggers_anything(self):
        """The removed rule, locked out.

        `weight_update_ratio` was measured across four learning rates on a real
        CIFAR-10 workload and does not separate healthy runs from failing ones:
        the healthiest run peaked at 0.285 and sustained 31 consecutive samples
        above the old 0.05 ceiling, while the *damaged* run peaked at 0.322 with
        only 26. Acting on it cost 1.74 and 0.78 points of validation accuracy in
        two A/B runs. It is reported as telemetry and must never intervene.
        """
        m, mod = self._monitor()
        healthy = {"gradient_entropy": 0.30, "effective_rank": 70.0, "weight_update_ratio": 0.001}
        self._settle_baseline(m, mod, healthy)

        for ratio in (0.06, 0.1, 0.3, 0.9):
            big_steps = {"gradient_entropy": 0.30, "effective_rank": 70.0,
                         "weight_update_ratio": ratio}
            for _ in range(mod.STRUCTURAL_SUSTAIN + 3):
                self.assertIsNone(
                    m.check_structural(big_steps),
                    f"update ratio {ratio} must not trigger an intervention",
                )

    def test_rank_collapse_fires_on_a_sustained_drop(self):
        m, mod = self._monitor()
        healthy = {"gradient_entropy": 0.30, "effective_rank": 70.0, "weight_update_ratio": 0.001}
        self._settle_baseline(m, mod, healthy)

        collapsed = {"gradient_entropy": 0.30, "effective_rank": 20.0, "weight_update_ratio": 0.001}
        result = None
        for _ in range(mod.STRUCTURAL_SUSTAIN):
            result = m.check_structural(collapsed)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "representation_collapse")

    def test_a_modest_rank_decline_is_left_alone(self):
        # Measured: a healthy run bottoms at 96% of its baseline rank and a
        # damaged one at 83%, against a trigger at 50%. Neither may fire.
        m, mod = self._monitor()
        healthy = {"gradient_entropy": 0.30, "effective_rank": 70.0, "weight_update_ratio": 0.001}
        self._settle_baseline(m, mod, healthy)

        for fraction in (0.96, 0.83, 0.6):
            specialising = {"gradient_entropy": 0.30, "effective_rank": 70.0 * fraction,
                            "weight_update_ratio": 0.001}
            for _ in range(mod.STRUCTURAL_SUSTAIN + 3):
                self.assertIsNone(
                    m.check_structural(specialising),
                    f"rank at {fraction:.0%} of baseline must not trigger",
                )

    def test_rank_collapse_is_the_only_structural_trigger(self):
        # Two of the three original rules were removed after each was measured
        # firing on healthy runs. With every signal pinned at its worst value,
        # rank collapse is the only verdict that may come back.
        m, mod = self._monitor()
        healthy = {"gradient_entropy": 0.30, "effective_rank": 70.0, "weight_update_ratio": 0.001}
        self._settle_baseline(m, mod, healthy)

        both_bad = {"gradient_entropy": 1e-9, "effective_rank": 1.0, "weight_update_ratio": 0.9}
        result = None
        for _ in range(mod.STRUCTURAL_SUSTAIN):
            result = m.check_structural(both_bad)
        self.assertEqual(result[0], "representation_collapse")


class TestLossPlateau(unittest.TestCase):
    """A run can die while its loss stays perfectly finite.

    This is the failure the structural tier was built for and did not catch. A
    real CIFAR-10 run at lr=0.5 (seed 1234, 2 epochs, 780 steps) finished at
    10.00% validation accuracy — chance — with its loss pinned at ln(10) from
    roughly step 25 onward. ARC recorded zero failures and zero interventions,
    because nothing went non-finite, the gradient norm stayed near 0.07, and the
    effective-rank rule needs a 50% fall that a dead network never produces.

    The numbers below are measured, not invented. Replaying the per-batch losses
    of that run and of a healthy lr=0.03 control (73.61% accuracy, same script,
    same seed, same 780 steps) through the best-loss-with-patience counter:

        arm                 max consecutive steps without improvement
        healthy lr=0.03                    82
        dead    lr=0.50                   764

    A 9.3x gap, against the ~10-point separation the rank signal offered and the
    zero the entropy signal offered. The default patience of 300 sits 3.7x above
    the healthy maximum and fires on the dead run at step 316.
    """

    def _monitor(self):
        import _arc_bootstrap

        m = _arc_bootstrap.OptimizerMonitor.__new__(_arc_bootstrap.OptimizerMonitor)
        m.best_loss = float("inf")
        m.opening_losses = []
        m.steps_without_improvement = 0
        return m, _arc_bootstrap

    def test_a_converged_run_is_not_a_plateau(self):
        """Convergence *is* a plateau, and the first version of this rule fired on it.

        The counter keys off the best-ever batch loss, so as a run succeeds its
        own record gets harder to beat and the stalls grow without bound. The
        780-step fixture above never revealed that. A 10-epoch A/B did,
        immediately: the `lr=0.03` arm reached **87.5% validation accuracy** and
        tripped this rule twice.

        No patience value fixes it — the stall length is unbounded on a healthy
        run. What separates the cases is whether the run ever got anywhere:

            arm              first loss   best loss   best/first
            lr=0.03 healthy     2.3018      0.6233       0.271
            lr=0.50 dead        2.3221      2.0632       0.888

        This is the regression test for that, and it runs 5x longer than the
        original fixture precisely because the original length is what hid the
        bug.
        """
        import random

        m, mod = self._monitor()
        rng = random.Random(7)
        # A converged run: big early gains, then a long noisy flat tail that
        # never beats its own record again.
        losses = [2.30 * (0.25 ** (i / 600)) for i in range(600)]
        losses += [0.58 + rng.uniform(0.0, 0.35) for _ in range(3300)]
        for i, loss in enumerate(losses, start=1):
            self.assertIsNone(
                m.check_plateau(loss),
                f"plateau fired at step {i} on a run that improved from its opening loss",
            )
        # And the stall really did exceed the patience — otherwise this test
        # would pass for the wrong reason and prove nothing.
        self.assertGreater(len(losses), mod.LOSS_PLATEAU_PATIENCE * 3)

    def test_one_unlucky_opening_batch_does_not_suppress_a_real_death(self):
        """The progress reference is a median, not a single sample.

        The guard asks "did this run ever get anywhere", measured against its
        opening loss. Taken from one batch that is a sample of size one: an
        unlucky high first batch makes a dead run look like it improved
        enormously, and the guard then suppresses a true positive — the rule
        goes quiet on exactly the failure it exists to catch.

        Here the run opens with a 9.5 spike and then dies at ln(10) forever.
        Against the single first loss the ratio is 2.06/9.5 = 0.22, comfortably
        under the 0.60 threshold, and the rule would stay silent. Against the
        median of the opening samples it is ~0.89, and the death is caught.
        """
        m, mod = self._monitor()
        losses = [9.5] + self._dead_losses(3900)[1:]
        fired = None
        for i, loss in enumerate(losses, start=1):
            if m.check_plateau(loss) is not None:
                fired = i
                break
        self.assertIsNotNone(
            fired,
            "a single high opening batch suppressed detection of a dead run",
        )
        self.assertLess(fired, 400)

    def test_a_zero_opening_loss_does_not_raise(self):
        """A degenerate opening reference must not throw inside failure handling.

        The progress ratio divides by the opening loss. If that opening is
        exactly 0.0 the guard's own `opening > 0` check skips the division, and
        the message below it then divided by zero anyway — raising
        ZeroDivisionError on the path that runs while a failure is being
        handled, which is the worst place in the harness to throw.

        Found by reading the diff, not by a failing run.
        """
        m, _ = self._monitor()
        result = None
        for _ in range(1200):
            result = m.check_plateau(0.0)
            if result is not None:
                break
        self.assertIsNotNone(result, "a run pinned at exactly zero still stalls")
        self.assertEqual(result[0], "loss_plateau")

    def test_a_converged_negative_objective_is_not_a_silent_death(self):
        """A negative loss must not defeat the progress guard.

        The guard is `best / opening < RATIO`. That test inverts under a sign
        flip, so for a negative-valued objective — a WGAN critic, a continuous-
        distribution NLL, Dice-minus-one — it stops separating a converged run
        from a dead one. The non-positive branch reported the plateau anyway,
        which fails *open*: this run improves from -0.5 to -5.0, a genuine 10x,
        then sits at its best for the patience window exactly as any converged
        run does, and was reported as a silent death three times before being
        silenced.

        That is the same false positive LOSS_PLATEAU_PROGRESS_RATIO was added to
        prevent, and the same failure mode both deleted structural rules were
        deleted for.
        """
        m, mod = self._monitor()
        for _ in range(mod.LOSS_PLATEAU_OPENING_SAMPLES):
            m.check_plateau(-0.5)
        for _ in range(500):
            self.assertIsNone(
                m.check_plateau(-5.0),
                "a negative-objective run that improved 10x was called a silent death",
            )

    def test_a_negative_objective_that_never_improves_is_not_claimed_either(self):
        """The cost of the fix above, asserted rather than left implicit.

        With a negative opening the ratio carries no information in either
        direction, so the rule cannot tell this genuinely dead run from the
        converged one above and reports neither. That is a real miss, and it is
        the deliberate choice: every other guard in the harness fails closed, and
        this project does not claim a failure it cannot substantiate.
        """
        m, mod = self._monitor()
        for _ in range(mod.LOSS_PLATEAU_OPENING_SAMPLES):
            m.check_plateau(-0.5)
        for _ in range(500):
            self.assertIsNone(m.check_plateau(-0.5))

    def test_a_dead_run_still_fires_with_the_progress_guard(self):
        """The guard must not disarm the rule it is guarding."""
        m, _ = self._monitor()
        fired = None
        for i, loss in enumerate(self._dead_losses(3900), start=1):
            if m.check_plateau(loss) is not None:
                fired = i
                break
        self.assertIsNotNone(fired, "a run pinned at ln(10) must still be caught")
        self.assertLess(fired, 400)

    def _healthy_losses(self, n=780):
        """Falling loss with heavy per-batch noise, as measured on lr=0.03.

        Noise amplitude is deliberately large — a real per-batch loss on this
        task swings ~40% of its mean — because a plateau rule that only survives
        on a smoothed curve is a rule that fires on real training.
        """
        import random

        rng = random.Random(1234)
        out = []
        for i in range(n):
            floor = 2.30 * (0.27 ** (i / n))  # 2.30 -> ~0.62, as measured
            out.append(floor * (1.0 + rng.uniform(-0.35, 0.35)))
        return out

    def _dead_losses(self, n=780):
        """Loss pinned at ln(10), as measured on lr=0.50."""
        import math
        import random

        rng = random.Random(1234)
        out = [2.32, 2.28, 2.25, 2.22, 2.24, 2.13, 2.14, 2.10, 2.23, 2.06]
        while len(out) < n:
            out.append(math.log(10) + rng.uniform(-0.04, 0.06))
        return out[:n]

    def test_healthy_run_never_plateaus(self):
        m, mod = self._monitor()
        worst = 0
        for loss in self._healthy_losses():
            self.assertIsNone(
                m.check_plateau(loss),
                f"plateau fired on a healthy falling loss after "
                f"{m.steps_without_improvement} idle steps",
            )
            worst = max(worst, m.steps_without_improvement)
        # Guard the margin itself, not just the verdict: if a future change
        # erodes the gap between the healthy maximum and the patience, this
        # fails while the run is still healthy rather than in production.
        self.assertLess(worst, mod.LOSS_PLATEAU_PATIENCE / 2)

    def test_dead_run_is_caught(self):
        m, mod = self._monitor()
        fired_at = None
        for i, loss in enumerate(self._dead_losses(), start=1):
            if m.check_plateau(loss) is not None:
                fired_at = i
                break
        self.assertIsNotNone(fired_at, "a run pinned at ln(10) for 780 steps was not caught")
        self.assertLess(fired_at, 780, "detection must land inside the run, not after it")

    def test_kind_and_reason_are_reported(self):
        m, mod = self._monitor()
        result = None
        for loss in self._dead_losses():
            result = m.check_plateau(loss)
            if result is not None:
                break
        self.assertIsNotNone(result)
        kind, reason = result
        self.assertEqual(kind, "loss_plateau")
        self.assertIn("improve", reason.lower())

    def test_counter_resets_after_firing(self):
        """Without a reset the rule re-fires every step and the LR is cut to zero."""
        m, mod = self._monitor()
        for loss in self._dead_losses():
            if m.check_plateau(loss) is not None:
                break
        self.assertEqual(m.steps_without_improvement, 0)

    def test_a_nonfinite_signal_is_reported_not_silently_dropped(self):
        """A signal going inf is information, and it was being discarded.

        `grad_flow_ratio` is late-layer gradient norm over early-layer, and it
        returns inf exactly when the early layers stop receiving gradient at
        all. Measured on a real pair of runs, the healthy arm stayed inside
        1.36-3.10 for all 16 samples while the dead arm hit 50.11 at step 50 and
        went non-finite from step 75 on. `_finite()` was correctly refusing to
        put inf into the JSON, which left a gap on the chart precisely where the
        signal was loudest.

        The flag carries the fact without inventing a number — a sentinel value
        would be indistinguishable from a real measurement once plotted.
        """
        import _arc_bootstrap

        m = _arc_bootstrap.OptimizerMonitor.__new__(_arc_bootstrap.OptimizerMonitor)

        class FakeSnapshot:
            signals = {
                "gradient.global": {
                    "total_grad_norm_l2": 0.07,
                    "grad_flow_ratio": float("inf"),
                },
                "weight.global": {"mean_effective_rank": 25.1},
            }

        class FakeCollector:
            def step(self): pass
            def collect(self): return FakeSnapshot()

        m.collector = FakeCollector()
        adv = m.collect_advanced()
        self.assertIsNotNone(adv)
        self.assertNotIn("grad_flow_ratio", adv, "inf must never reach the JSON")
        self.assertIn("grad_flow_ratio", adv.get("nonfinite", []),
                      "the divergence must still be reported")
        self.assertEqual(adv["effective_rank"], 25.1, "finite signals are unaffected")

    def test_the_agent_snapshot_survives_a_nonfinite_flag(self):
        """`nonfinite` is a list, and the snapshot rounds every value.

        This runs while a failure is already being handled, which is the worst
        possible place to raise a TypeError.
        """
        import arc_agent

        snap = arc_agent._snapshot(
            step=100, epoch=1, loss=2.30, grad_norm=0.07, lr=0.5,
            loss_history=[2.30, 2.31, 2.30],
            advanced={"effective_rank": 25.1, "nonfinite": ["grad_flow_ratio"]},
        )
        tel = snap["advanced_telemetry"]
        self.assertEqual(tel["nonfinite"], ["grad_flow_ratio"])
        self.assertEqual(tel["effective_rank"], 25.1)
        json.dumps(snap)  # must stay serialisable

    def test_the_progress_threshold_separates_both_sides(self):
        """Pins the 0.60 threshold itself, from both directions.

        Every other test here uses values far from the boundary, so the constant
        could drift a long way — or be read with the comparison inverted — while
        the suite stayed green. This fixes it in place: a run that improved just
        past the threshold is convergence and stays silent, one that improved
        just short of it is a stall and fires.

        Deliberately expressed relative to LOSS_PLATEAU_PROGRESS_RATIO rather
        than hard-coding 0.59/0.61, so retuning the threshold on new evidence
        moves this test with it instead of forcing someone to edit a magic
        number they may not understand.
        """
        import _arc_bootstrap as mod

        ratio = mod.LOSS_PLATEAU_PROGRESS_RATIO
        opening = 1.0

        for best, should_fire in ((opening * (ratio - 0.01), False),
                                  (opening * (ratio + 0.01), True)):
            m, _ = self._monitor()
            for _ in range(mod.LOSS_PLATEAU_OPENING_SAMPLES):
                m.check_plateau(opening)
            m.check_plateau(best)

            verdict = None
            for _ in range(mod.LOSS_PLATEAU_PATIENCE + 5):
                verdict = m.check_plateau(best) or verdict

            self.assertEqual(
                verdict is not None, should_fire,
                f"best/opening = {best / opening:.3f} against a {ratio} threshold: "
                f"expected {'a plateau' if should_fire else 'silence'}",
            )

    def test_an_absent_loss_is_not_a_plateau(self):
        """NaN reaches here when the loss could not be observed at all.

        Counting that as "no improvement" would let a run with an unobservable
        loss — torch.autograd.backward(), LBFGS — accumulate idle steps and trip
        the rule while training perfectly well.
        """
        m, _ = self._monitor()
        for _ in range(2000):
            self.assertIsNone(m.check_plateau(float("nan")))
        self.assertEqual(m.steps_without_improvement, 0)


class TestRiskGaugeTracksTheStall(unittest.TestCase):
    """The gauge and the failure banner are read together, so they must agree.

    A silent death moves none of the risk inputs: the loss does not double, it
    sits still, and the gradient norm on a dead CIFAR run is about 0.07. The
    dashboard therefore showed "FAILURE DETECTED — stalled" beside a risk gauge
    reading LOW / 0.0, and kept showing it for the rest of the run. That is the
    single most visible self-contradiction the tool could produce.
    """

    def _monitor(self):
        import _arc_bootstrap

        m = _arc_bootstrap.OptimizerMonitor.__new__(_arc_bootstrap.OptimizerMonitor)
        m.best_loss = float("inf")
        m.opening_losses = []
        m.steps_without_improvement = 0
        m.plateau_confirmed = False
        return m, _arc_bootstrap

    def test_risk_rises_before_the_failure_marker_lands(self):
        """The gauge should lead the verdict, not trail it."""
        m, mod = self._monitor()
        for _ in range(mod.LOSS_PLATEAU_OPENING_SAMPLES):
            m.check_plateau(2.3026)

        seen = {}
        for i in range(1, mod.LOSS_PLATEAU_PATIENCE):
            m.check_plateau(2.3026)
            seen[i] = mod._risk_score([2.3026] * 5, 0.07, False, m.stall_ratio())

        self.assertEqual(seen[1][1], "LOW", "a brief stall is not yet a concern")
        self.assertGreater(seen[mod.LOSS_PLATEAU_PATIENCE - 1][0], seen[1][0],
                           "risk must climb as the stall lengthens")
        self.assertIn(seen[mod.LOSS_PLATEAU_PATIENCE - 1][1], ("HIGH", "CRITICAL"),
                      "by the time the verdict is imminent the gauge must not read LOW")

    def test_risk_stays_up_after_the_verdict(self):
        """Sticky, because the counter resets and a dead run does not recover.

        `check_plateau` zeroes `steps_without_improvement` every time it reaches
        patience. Without a latch the gauge sawtooths — HIGH, fire, LOW, climb
        again — which on screen reads as the run having recovered.
        """
        m, mod = self._monitor()
        for _ in range(mod.LOSS_PLATEAU_OPENING_SAMPLES):
            m.check_plateau(2.3026)
        for _ in range(mod.LOSS_PLATEAU_PATIENCE):
            m.check_plateau(2.3026)
        self.assertTrue(m.plateau_confirmed)

        for _ in range(mod.LOSS_PLATEAU_PATIENCE * 2):
            m.check_plateau(2.3026)
            _, label = mod._risk_score([2.3026] * 5, 0.07, False, m.stall_ratio())
            self.assertIn(label, ("HIGH", "CRITICAL"),
                          "the gauge must not fall back to LOW while the run is still dead")

    def test_a_converged_run_never_raises_the_gauge(self):
        """The false positive this would otherwise reintroduce, on the demo's own healthy arm.

        A converged run stalls indefinitely too — that is why the plateau rule
        needed a progress guard. The gauge reads the same guard, so an 87% run
        sitting at its best loss forever stays LOW.
        """
        m, mod = self._monitor()
        for _ in range(mod.LOSS_PLATEAU_OPENING_SAMPLES):
            m.check_plateau(2.30)
        m.check_plateau(0.62)

        worst = 0.0
        for _ in range(mod.LOSS_PLATEAU_PATIENCE * 3):
            m.check_plateau(0.62)
            worst = max(worst, mod._risk_score([0.62] * 5, 0.5, False, m.stall_ratio())[0])
        self.assertEqual(worst, 0.0, "a converged run must never raise the risk gauge")
        self.assertFalse(m.plateau_confirmed)

    def test_a_non_finite_loss_still_outranks_everything(self):
        import _arc_bootstrap
        self.assertEqual(_arc_bootstrap._risk_score([1.0], 0.1, True, 0.0), (1.0, "CRITICAL"))


class TestPlateauReportsWithoutActing(unittest.TestCase):
    """Detecting the plateau was right. Acting on it destroyed the run.

    The four-LR A/B measured the lr=0.5 pair (seed 1234, 10 epochs, identical
    data order). Both arms sat at chance — 10.00% — for four epochs. Then:

        epoch    baseline (no action)     active (3x reduce_lr from step 316)
            5    26.73%  lr=2.56e-01      10.00%  lr=3.20e-02
           10    73.19%                   10.00%  loss 2.3026 = ln(10)

    The control arm escaped once cosine decay walked the LR down on its own.
    The arm ARC "helped" never did: cutting the LR at the moment of the plateau
    removed the only steps large enough to carry the weights out of the dead
    region. A -63.19pp delta, in ARC's disfavour, from a correct detection.

    So `loss_plateau` is report-only, and these tests hold it there. The trap is
    that reporting is not automatically passive — `_handle_failure` ends by
    calling `optimizer.zero_grad()`, which discards the user's update and is the
    single most consequential intervention available on a diverging run.
    """

    def _fire(self, mode="active", attempts=0):
        """Drive _handle_failure for a plateau and record what it touched."""
        import _arc_bootstrap

        class Optimizer:
            def __init__(self):
                self.zeroed = 0
                self.param_groups = [{"lr": 0.5}]

            def zero_grad(self, set_to_none=False):
                self.zeroed += 1

        events = []
        opt = Optimizer()
        original_emit = _arc_bootstrap.emit
        original_enabled = _arc_bootstrap.INTERVENTIONS_ENABLED
        original_state = _arc_bootstrap.STATE
        _arc_bootstrap.emit = events.append
        _arc_bootstrap.INTERVENTIONS_ENABLED = (mode == "active")
        _arc_bootstrap.STATE = _arc_bootstrap._RunState()
        _arc_bootstrap.STATE.attempts_by_kind["loss_plateau"] = attempts
        try:
            _arc_bootstrap._handle_failure(
                opt, None, step=316, loss_val=2.3026, grad_norm=0.07, lr=0.5,
                advanced={}, kind="loss_plateau",
                reason="loss has failed to improve for 300 consecutive steps",
            )
            result = types.SimpleNamespace(
                events=events,
                optimizer=opt,
                attempts=_arc_bootstrap.STATE.attempts_by_kind["loss_plateau"],
                interventions=_arc_bootstrap.STATE.intervention_count,
                # Read before STATE is restored, or the assertion checks the
                # wrong object and passes for the wrong reason.
                abandoned=set(_arc_bootstrap.STATE.abandoned_kinds),
                silenced=set(_arc_bootstrap.STATE.silenced_kinds),
            )
        finally:
            _arc_bootstrap.emit = original_emit
            _arc_bootstrap.INTERVENTIONS_ENABLED = original_enabled
            _arc_bootstrap.STATE = original_state
        return result

    def test_the_plateau_is_still_reported(self):
        """Report-only must not mean silent. The detection was the part that worked."""
        detected = [e for e in self._fire().events if e.get("type") == "failure_detected"]
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0]["kind"], "loss_plateau")
        self.assertEqual(detected[0]["step"], 316)

    def test_no_intervention_is_emitted(self):
        r = self._fire()
        self.assertEqual(r.interventions, 0)
        self.assertEqual([e for e in r.events if e.get("type") == "intervention"], [])

    def test_the_users_update_is_not_discarded(self):
        """The regression that would silently reintroduce the -63pp result.

        zero_grad() before the caller's own optimizer.step() makes that step a
        no-op. A rule that claims to change nothing while dropping updates is
        indistinguishable, in the run's outcome, from one that intervenes.
        """
        r = self._fire()
        self.assertEqual(r.optimizer.zeroed, 0)
        self.assertEqual(r.optimizer.param_groups[0]["lr"], 0.5, "LR must be untouched")

    def test_both_ab_arms_behave_identically(self):
        """A report-only kind cannot produce an arm difference — that is the point.

        The gate sits above the INTERVENTIONS_ENABLED check precisely so the
        control arm and the active arm run the same code for this kind.
        """
        active = self._fire(mode="active")
        baseline = self._fire(mode="baseline")
        self.assertEqual(active.interventions, baseline.interventions, 0)
        self.assertEqual(active.optimizer.zeroed, baseline.optimizer.zeroed, 0)
        self.assertEqual(
            [e.get("type") for e in active.events], [e.get("type") for e in baseline.events],
            "the two arms must emit the same event sequence for a report-only kind",
        )

    def test_repeat_reports_are_capped(self):
        """A dead run stays dead, so the rule re-fires forever. Say it, then stop."""
        import _arc_bootstrap

        done = [e for e in self._fire(attempts=_arc_bootstrap.MAX_ATTEMPTS_PER_KIND).events
                if e.get("type") == "detection_silenced"]
        self.assertEqual(len(done), 1)
        self.assertNotIn(
            "rollback", done[0]["message"].lower(),
            "the report-only path never rolled anything back; the message must not claim it did",
        )

    def test_a_reported_plateau_is_never_called_unrecoverable(self):
        """The claim ARC has not earned, and would have been wrong to make.

        `unrecoverable` means "every checkpoint is degenerate, stop paying for
        this GPU". ARC can say that after its recoveries fail. It cannot say it
        having taken no action — and on the measured lr=0.5 pair it would have
        been false: the arm left alone reported this same plateau and finished
        at 73.19%. A user who killed that run on ARC's advice would have thrown
        away a run that was coming back.
        """
        import _arc_bootstrap

        events = self._fire(attempts=_arc_bootstrap.MAX_ATTEMPTS_PER_KIND).events
        self.assertEqual([e for e in events if e.get("type") == "unrecoverable"], [])

    def test_the_run_summary_does_not_list_it_as_unrecoverable(self):
        """`run_summary.unrecoverable` is what the A/B table reads for its verdict.

        Report-only kinds are tracked in a separate set for exactly this reason:
        publishing one there would have labelled the recovering 73.19% control
        arm "unrecoverable" in the results table.
        """
        import _arc_bootstrap

        r = self._fire(attempts=_arc_bootstrap.MAX_ATTEMPTS_PER_KIND)
        self.assertEqual(r.abandoned, set(), "must not reach the run summary's unrecoverable list")
        self.assertEqual(r.silenced, {"loss_plateau"}, "but it must still stop repeating")

    def test_the_agents_response_map_agrees(self):
        """Two files encode this decision. Disagreement is how it comes back."""
        import _arc_bootstrap, arc_agent

        for kind in _arc_bootstrap.REPORT_ONLY_KINDS:
            self.assertEqual(arc_agent.STRUCTURAL_RESPONSES[kind][1], "report")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint store — needs torch, works on CPU
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestCheckpointStore(unittest.TestCase):
    def _build(self, max_checkpoints=3):
        import _arc_bootstrap
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        return _arc_bootstrap.CheckpointStore(model, opt, max_checkpoints), model, opt

    def test_checkpoints_live_on_the_host(self):
        store, _, _ = self._build()
        store.save(0)
        for tensor in store.snapshots[0]["model"].values():
            self.assertEqual(tensor.device.type, "cpu")

    def test_restore_returns_the_original_weights(self):
        store, model, opt = self._build()
        original = model.weight.detach().clone()
        store.save(0)

        model.weight.data.add_(5.0)
        self.assertFalse(torch.allclose(model.weight.detach(), original))

        store.restore()
        self.assertTrue(torch.allclose(model.weight.detach(), original))

    def test_ring_is_bounded(self):
        store, _, _ = self._build(max_checkpoints=2)
        for step in range(6):
            store.save(step)
        self.assertEqual(len(store.snapshots), 2)
        self.assertEqual(store.snapshots[-1]["step"], 5)

    def test_restore_with_no_checkpoint_is_a_noop(self):
        store, _, _ = self._build()
        self.assertEqual(store.restore(), 0)

    def test_rng_state_round_trips(self):
        # Determinism across a rollback is what makes the baseline-vs-active
        # comparison a fair one.
        store, _, _ = self._build()
        store.save(0)
        first = torch.randn(4)
        store.restore()
        second = torch.randn(4)
        self.assertTrue(torch.allclose(first, second))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestGradientNorm(unittest.TestCase):
    """The fused norm must agree with the naive one it replaced."""

    def setUp(self):
        import _arc_bootstrap
        self.mod = _arc_bootstrap

    @staticmethod
    def _naive(params):
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5

    def test_matches_the_naive_per_parameter_computation(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))
        model(torch.randn(8, 6)).sum().backward()
        params = list(model.parameters())

        fused = float(self.mod._grad_norm_tensor(params).item())
        self.assertAlmostEqual(fused, self._naive(params), places=4)

    def test_single_parameter_path(self):
        # len(grads) == 1 takes a separate branch from _foreach_norm.
        torch.manual_seed(0)
        layer = nn.Linear(4, 1, bias=False)
        layer(torch.randn(3, 4)).sum().backward()
        params = list(layer.parameters())
        fused = float(self.mod._grad_norm_tensor(params).item())
        self.assertAlmostEqual(fused, self._naive(params), places=5)

    def test_no_gradients_yields_none(self):
        model = nn.Linear(3, 2)
        self.assertIsNone(self.mod._grad_norm_tensor(list(model.parameters())))

    def test_ignores_parameters_without_gradients(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for p in model[0].parameters():
            p.requires_grad_(False)
        model(torch.randn(5, 4)).sum().backward()
        params = list(model.parameters())
        fused = float(self.mod._grad_norm_tensor(params).item())
        self.assertAlmostEqual(fused, self._naive(params), places=5)
        self.assertGreater(fused, 0.0)


# There was a TestLossTrend suite here, covering a loss-trend guard that has
# since been deleted. The guard tried to read a ~1% improvement out of a signal
# whose noise is ~40% of its own mean; no window recovers that, and the two
# rules that depended on it now stand on measured thresholds instead. Removing
# the code removed the reason for the tests.


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestModelResolution(unittest.TestCase):
    """An optimizer must be matched to the model that owns its parameters.

    The frame walk alone took the *first* nn.Module it found, which is how a
    GAN's discriminator optimizer ended up rolling back the generator.
    """

    def setUp(self):
        import _arc_bootstrap
        self.mod = _arc_bootstrap
        self.mod._EXPLICIT_MODELS.clear()

    def test_gan_optimizers_resolve_to_their_own_models(self):
        generator = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        discriminator = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
        g_opt = torch.optim.Adam(generator.parameters(), lr=1e-3)
        d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

        # Both models are live locals here, exactly as in a real GAN loop.
        self.assertIs(self.mod._find_owning_model(g_opt), generator)
        self.assertIs(self.mod._find_owning_model(d_opt), discriminator)

    def test_prefers_the_specific_submodule_over_a_wrapper(self):
        # A container holding both parts also owns every parameter, so an
        # overlap test alone would tie; the smaller exact owner must win.
        class Pair(nn.Module):
            def __init__(self):
                super().__init__()
                self.gen = nn.Linear(8, 8)
                self.disc = nn.Linear(8, 1)

        pair = Pair()
        d_opt = torch.optim.SGD(pair.disc.parameters(), lr=0.1)
        self.assertIs(self.mod._find_owning_model(d_opt), pair.disc)

    def test_explicit_watch_wins(self):
        model = nn.Linear(5, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        self.mod.watch(model)
        self.assertIs(self.mod._find_owning_model(opt), model)

    def test_unresolvable_optimizer_returns_none(self):
        orphan = torch.nn.Parameter(torch.randn(3))
        opt = torch.optim.SGD([orphan], lr=0.1)
        self.assertIsNone(self.mod._find_owning_model(opt))


@unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), "needs CUDA for GradScaler")
class TestAmpLossCapture(unittest.TestCase):
    """Under AMP the recorded loss must be the unscaled one."""

    def test_unscale_marks_the_optimizer_as_unscaled(self):
        """The standard AMP-with-clipping recipe must not double-unscale.

        `scaler.scale(loss).backward(); scaler.unscale_(opt); clip_grad_norm_(...)`
        is the documented pattern. Only `scaler.step` used to register the
        optimizer as unscaled, so a user who then called `opt.step()` directly
        had the gradient norm divided by the scale a second time — reported
        ~65536x too small, low enough that no gradient rule could ever fire.
        """
        import _arc_bootstrap as mod
        mod.install()

        model = nn.Linear(4, 2).cuda()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda")

        mod.STATE.unscaled_optimizers.clear()
        scaler.scale(model(torch.randn(8, 4, device="cuda")).sum()).backward()
        self.assertNotIn(id(opt), mod.STATE.unscaled_optimizers,
                         "gradients are still scaled before unscale_")
        scaler.unscale_(opt)
        self.assertIn(id(opt), mod.STATE.unscaled_optimizers,
                      "unscale_ must mark the optimizer so the norm is not divided twice")
        mod.STATE.unscaled_optimizers.clear()
        mod.STATE.pending_loss = None
        mod.STATE.pending_is_scaled = False

    def test_scale_records_the_unscaled_loss(self):
        import _arc_bootstrap as mod
        mod.install()  # idempotent

        scaler = torch.amp.GradScaler("cuda")
        loss = torch.tensor(0.25, device="cuda", requires_grad=True)
        scaled = scaler.scale(loss)

        # scale() multiplies by ~65536; the recorded value must not.
        self.assertIsNotNone(mod.STATE.pending_loss)
        self.assertAlmostEqual(float(mod.STATE.pending_loss.item()), 0.25, places=5)
        self.assertGreater(float(scaled.item()), 1.0, "sanity: the scaled tensor really is scaled")
        mod.STATE.pending_loss = None
        mod.STATE.pending_is_scaled = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestLearningRateGuard(unittest.TestCase):
    """An LR intervention must survive the user's scheduler."""

    def _monitor(self):
        import _arc_bootstrap
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        return _arc_bootstrap.OptimizerMonitor(opt, model, "test"), opt

    def test_scale_lr_reduces_immediately(self):
        m, opt = self._monitor()
        old, new = m.scale_lr(0.5)
        self.assertEqual(old, 1.0)
        self.assertEqual(new, 0.5)

    def test_scheduler_cannot_undo_the_reduction(self):
        m, opt = self._monitor()
        m.scale_lr(0.5)
        for _ in range(5):
            for group in opt.param_groups:   # scheduler rewrites lr from base
                group["lr"] = 1.0
            m.enforce_lr()
            self.assertEqual(opt.param_groups[0]["lr"], 0.5)

    def test_no_scheduler_does_not_compound(self):
        m, opt = self._monitor()
        m.scale_lr(0.5)
        for _ in range(5):
            m.enforce_lr()
        self.assertEqual(opt.param_groups[0]["lr"], 0.5)

    def test_a_rollback_cannot_restore_the_diverged_lr(self):
        """The invariant, tested through behaviour rather than through storage.

        A previous version asserted a parallel `snapshot["lr"]` list existed.
        That list was never read by `restore()` — the learning rate comes back
        from the checkpointed optimizer state dict — so the assertion could hold
        while the actual restore was wrong. This restores and reads the result.
        """
        m, opt = self._monitor()
        m.store.save(0)
        m.scale_lr(0.25)
        m.store.restore()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.25,
                               msg="rollback must not resurrect the pre-intervention LR")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: the harness on a real training loop
# ─────────────────────────────────────────────────────────────────────────────

# A configuration that genuinely diverges, verified WITHOUT ARC before being
# used to test that ARC detects divergence.
#
# That verification is not ceremony. The previous fixture — 4 layers of width 64
# at lr=3.0 — does not diverge at all: its loss peaks at 1.93 and no threshold is
# ever crossed. The tests below passed anyway, because the since-removed
# gradient-entropy rule raised a false positive on it, and an assertion of
# "failures were detected" cannot tell a real detection from a spurious one. The
# suite was certifying the bug.
#
# Measured in plain PyTorch, CPU, seed 0: loss exceeds 1e6 at step 10 and reaches
# ~1.2e10. Any fixture asserting detection has to be checked this way first.
SCRIPT_DIVERGE = """
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(16,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(),
    nn.Linear(128,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(),
    nn.Linear(128,128), nn.ReLU(), nn.Linear(128,4),
)
opt = torch.optim.SGD(model.parameters(), lr=8.0, momentum=0.9)
X = torch.randn(512,16); Y = torch.randint(0,4,(512,))
for step in range(60):
    for g in opt.param_groups: g["lr"] = 8.0   # scheduler-style rewrite
    idx = torch.randint(0,512,(64,))
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(X[idx]), Y[idx]).backward()
    opt.step()
print("FINAL_LR", opt.param_groups[0]["lr"])
"""

SCRIPT_SCHEDULER = """
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
# Constructing any of these used to raise AttributeError before step 1, because
# LRScheduler reads optimizer.step.__func__ and ARC had replaced step with a
# plain function.
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
for step in range(20):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(torch.randn(16, 8)), torch.randint(0, 2, (16,)))
    loss.backward()
    opt.step()
    sched.step()
print("SCHEDULER_OK", opt.param_groups[0]["lr"])
"""

SCRIPT_AUTOGRAD_BACKWARD = """
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
before = model[0].weight.detach().clone()
for step in range(30):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(torch.randn(16, 8)), torch.randint(0, 2, (16,)))
    # Function form — not Tensor.backward, so ARC cannot observe the loss.
    torch.autograd.backward(loss)
    opt.step()
moved = not torch.allclose(before, model[0].weight.detach())
print("WEIGHTS_MOVED", moved)
"""

SCRIPT_BASELINE_NAN = """
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(10):
    opt.zero_grad(set_to_none=True)
    out = model(torch.randn(8, 4))
    # A genuinely non-finite loss every step.
    loss = (out * float("inf")).sum()
    loss.backward()
    opt.step()
finite = bool(torch.isfinite(model.weight).all())
print("WEIGHTS_FINITE", finite)
"""

SCRIPT_GRAD_SPIKE = """
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,2))
opt = torch.optim.SGD(model.parameters(), lr=1e-4)
x = torch.randn(16,8); y = torch.randint(0,2,(16,))
for step in range(40):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    # Gradient norm far above the 50 threshold while the loss stays finite.
    for p in model.parameters():
        p.grad = torch.full_like(p, 60.0)
    opt.step()
print("LOSS_FINITE", bool(torch.isfinite(loss)))
"""

SCRIPT_PLATEAU = """
import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
before = model.weight.detach().clone()
x = torch.randn(8, 4)
for step in range(60):
    opt.zero_grad(set_to_none=True)
    # Finite, constant, never improving: the best-ever counter never resets and
    # the progress ratio stays at 1.0. That is the shape of a silent death.
    # It has to run through backward(), because that is where the harness
    # observes the loss at all — multiplying by zero keeps the graph and pins
    # the value.
    loss = model(x).sum() * 0.0 + 2.3026
    loss.backward()
    # Overwrite the (zero) gradient with a real one. A constant loss carries no
    # gradient of its own, and a zero gradient would leave this fixture unable
    # to tell "ARC left my update alone" from "ARC discarded it" — which is the
    # thing being tested.
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.01)
    opt.step()
print("MOVED", bool((model.weight.detach() - before).abs().max() > 1e-6))
"""

SCRIPT_ACCUM = """
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,2))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for update in range(5):
    opt.zero_grad(set_to_none=True)
    for micro in range(4):                      # 4 backwards, 1 update
        x = torch.randn(16,8); y = torch.randint(0,2,(16,))
        (F.cross_entropy(model(x), y) / 4).backward()
    opt.step()
"""


def run_harness(source: str, env_extra=None):
    """Run a script through runner.py and return the parsed event stream."""
    # Unique per call. This used to be a fixed `.test_workload.py`, which two
    # concurrent runs of this suite would overwrite for each other — one test's
    # script executing under another's assertions, producing failures that look
    # like real regressions and vanish on a rerun. It cost two false alarms
    # before it was tracked down. The name has to stay inside REPO so the
    # traceback-line-number tests still see a path they expect.
    script = REPO / f".test_workload_{os.getpid()}_{next(_WORKLOAD_SEQ)}.py"
    script.write_text(source, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("ARC_ADVANCED_EVERY", "5")
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(
            [sys.executable, str(PY_DIR / "runner.py"), str(script)],
            capture_output=True, text=True, env=env, cwd=str(REPO), timeout=600,
        )
    finally:
        script.unlink(missing_ok=True)

    events = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events, proc.stdout


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestIntegration(unittest.TestCase):
    def test_one_metric_per_optimizer_step_not_per_backward(self):
        events, _ = run_harness(SCRIPT_ACCUM)
        metrics = [e for e in events if e["type"] == "metric"]
        summary = next(e for e in events if e["type"] == "run_summary")
        self.assertEqual(len(metrics), 5, "gradient accumulation must not inflate the step count")
        # 20 backwards, 5 updates. `steps` counts updates; the separate
        # `optimizer_steps` field was removed because it could never differ from
        # it, which made it read as a cross-check that did not exist.
        self.assertEqual(summary["backward_calls"], 20)
        self.assertEqual(summary["steps"], 5)

    def test_every_emitted_line_is_valid_json(self):
        _, stdout = run_harness(SCRIPT_ACCUM)
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                json.loads(line)  # raises on NaN/Infinity leaking through

    def test_divergence_is_detected_and_recovered(self):
        events, stdout = run_harness(SCRIPT_DIVERGE)
        failures = [e for e in events if e["type"] == "failure_detected"]
        interventions = [e for e in events if e["type"] == "intervention"]
        self.assertTrue(failures, "a genuinely diverging run must be detected")
        # It must be detected *for the right reason*. Asserting only that some
        # failure was reported is what let a false positive from an unrelated
        # rule satisfy this test while the fixture never diverged at all.
        self.assertEqual(
            failures[0]["kind"], "numerical",
            f"expected a numerical divergence, got {failures[0].get('kind')}",
        )
        self.assertTrue(interventions, "a detected failure must produce an intervention")
        self.assertLessEqual(failures[0]["step"], interventions[0]["step"],
                             "the failure must precede its remedy")

    def test_a_gradient_spike_on_a_finite_loss_triggers_nothing(self):
        """Pins a documented limitation that is easy to break by accident.

        README, ARCHITECTURE and COMPETITIVE_LANDSCAPE all used to list
        "gradient norm above 50" as a trigger of its own. It is not.
        `run_recovery_agent` has one call site and `_handle_failure` only reaches
        it with kind="numerical", so the grad-norm test that latches clipping
        lives *inside* a path only an already-non-finite or exploded loss can
        open. A run whose gradients spike while its loss stays finite is
        measured, charted and risk-scored, and never clipped.

        This test exists because that was asserted in three documents before
        anyone ran it. Measured here rather than reasoned about: the norm reaches
        ~1129, twenty-two times the threshold, and nothing fires.

        If someone later gives gradient explosion its own entry point — an open
        design question in FUTURE_IMPROVEMENTS — this test is the one that should
        fail, and the docs it guards must change with it.
        """
        events, stdout = run_harness(SCRIPT_GRAD_SPIKE)
        self.assertIn("LOSS_FINITE True", stdout, "the fixture must keep the loss finite")

        norms = [e["grad_norm"] for e in events
                 if e["type"] == "metric" and e.get("grad_norm") is not None]
        self.assertGreater(max(norms), 50.0,
                           "fixture must actually exceed the threshold, or it proves nothing")

        self.assertEqual([e for e in events if e["type"] == "failure_detected"], [],
                         "a finite loss must not be reported as a failure")
        self.assertEqual([e for e in events if e["type"] == "intervention"], [],
                         "gradient explosion alone must not produce an intervention")

    def test_a_plateau_is_reported_through_the_real_harness_without_acting(self):
        """End to end: detection reaches the log, nothing reaches the model.

        The unit tests drive `_handle_failure` directly. This one runs a real
        script through `runner.py`, so it also covers the dispatch that decides
        a plateau happened and the wiring between the two — the layer where a
        report-only kind could still fall through to the recovery agent's
        fallback, which rolls back and cuts the LR.

        Patience is lowered by env var rather than by waiting 300 steps; the
        rule's thresholds are configurable precisely so they can be tested.
        """
        events, stdout = run_harness(SCRIPT_PLATEAU, {
            "ARC_PLATEAU_PATIENCE": "10",
            "ARC_PLATEAU_OPENING": "2",
        })
        failures = [e for e in events if e["type"] == "failure_detected"]
        self.assertTrue(failures, "a run that never improves must be detected")
        self.assertEqual(failures[0]["kind"], "loss_plateau",
                         f"expected a plateau, got {failures[0].get('kind')}")

        self.assertEqual([e for e in events if e["type"] == "intervention"], [],
                         "a report-only kind must produce no intervention")
        self.assertEqual([e for e in events if e["type"] == "unrecoverable"], [],
                         "ARC took no action here, so it cannot judge the run unrecoverable")

        # The user's own updates must still land. If ARC zeroed the gradients on
        # the way out of the report path, the weights never move and this prints
        # False — silently turning "we changed nothing" into a false claim.
        self.assertIn("MOVED True", stdout,
                      "the training script's updates must reach the weights untouched")

        summary = next(e for e in events if e["type"] == "run_summary")
        self.assertEqual(summary["unrecoverable"], [])

    def test_intervention_survives_the_scheduler(self):
        events, stdout = run_harness(SCRIPT_DIVERGE)
        interventions = [e for e in events if e["type"] == "intervention"]
        self.assertTrue(interventions)
        final_lr = float(stdout.split("FINAL_LR")[1].split()[0])
        self.assertLess(final_lr, 8.0,
                        "the script rewrites lr=8.0 every step; ARC's reduction must persist")

    def test_baseline_mode_reports_but_never_intervenes(self):
        events, _ = run_harness(SCRIPT_DIVERGE, {"ARC_MODE": "baseline"})
        self.assertTrue([e for e in events if e["type"] == "failure_detected"],
                        "baseline must still detect and report")
        self.assertEqual([e for e in events if e["type"] == "intervention"], [],
                         "baseline is the control arm and must change nothing")

    def test_lr_schedulers_can_be_constructed(self):
        """Regression: ARC used to crash every script that used a scheduler.

        `LRScheduler.__init__` reads `optimizer.step.__func__`. ARC replaced
        `step` with a plain function, which has no `__func__`, so constructing a
        scheduler raised AttributeError before training began — on the majority
        of real training scripts. The repo's own demo sets the LR by hand, so
        nothing here exercised it until this test.
        """
        events, stdout = run_harness(SCRIPT_SCHEDULER)
        self.assertIn("SCHEDULER_OK", stdout,
                      "constructing an LR scheduler must not raise")
        errors = [e for e in events if e["type"] == "error"]
        self.assertEqual(errors, [], f"unexpected error: {errors[:1]}")

    def test_an_unobservable_loss_is_not_treated_as_a_failure(self):
        """Regression: absent loss was diagnosed as NaN, freezing healthy runs.

        `torch.autograd.backward()` is the function form and is not the patched
        `Tensor.backward`, so ARC never sees the loss. Treating that as NaN made
        it roll back, cut the LR and zero the gradients on every step — the model
        never moved, while the log filled with successful interventions.
        """
        events, stdout = run_harness(SCRIPT_AUTOGRAD_BACKWARD)
        self.assertIn("WEIGHTS_MOVED True", stdout,
                      "ARC must not suppress updates it cannot observe")
        self.assertEqual([e for e in events if e["type"] == "intervention"], [],
                         "an unobservable loss must not trigger a recovery")
        degraded = [e for e in events if e["type"] == "degraded"]
        self.assertTrue(any("loss" in (e.get("component") or "") for e in degraded),
                        "the limitation must be reported, not hidden")

    def test_baseline_mode_does_not_protect_the_run(self):
        """The A/B is only valid if the control arm is genuinely unprotected.

        Baseline mode used to call `optimizer.zero_grad()` on a detected
        failure. Dropping the bad update is most of what a rollback achieves, so
        the control arm was being rescued by the mechanism the experiment exists
        to measure. With a non-finite loss every step and no protection, the
        weights must actually go non-finite.
        """
        events, stdout = run_harness(SCRIPT_BASELINE_NAN, {"ARC_MODE": "baseline"})
        self.assertIn("WEIGHTS_FINITE False", stdout,
                      "baseline mode must let the run fail")
        self.assertTrue([e for e in events if e["type"] == "failure_detected"],
                        "baseline must still detect and report the failure")
        self.assertEqual([e for e in events if e["type"] == "intervention"], [])

    def test_traceback_points_at_the_user_script(self):
        events, _ = run_harness("raise RuntimeError('boom')\n")
        errors = [e for e in events if e["type"] == "error"]
        self.assertTrue(errors)
        tb = errors[0]["traceback"]
        self.assertIn("line 1", tb, "line numbers must not be shifted by injected code")
        self.assertNotIn("_arc_bootstrap", tb, "runner frames must be stripped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
