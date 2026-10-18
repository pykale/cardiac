"""LS-EMVAE's robustness curve: one lead fewer at each step, scored on the test split.

The claim the lead-specific design rests on is that a fine-tuned model keeps working as
leads go missing. ``robustness.csv`` is where a run reports it, so what is checked here
is that the curve withholds the leads it says it does, in the order configured, and
never the last one.
"""

from __future__ import annotations

import csv

import pytest

from examples.lsemvae_ecg import runner as lsemvae_runner
from kalecardiac.evaluate import multiclass_roc_auc, r2_score, roc_auc
from tests.examples.test_example_workflows import lsemvae_cfg


class TestWithholdingOrder:
    def test_leads_go_from_the_end_of_the_group_and_the_first_stays(self):
        order = lsemvae_runner.withholding_order((), ("I", "II", "III", "aVR", "aVL", "aVF"))
        assert order == ["aVF", "aVL", "aVR", "III", "II"]

    def test_a_configured_priority_goes_first(self):
        assert lsemvae_runner.withholding_order(("I",), ("I", "II", "III")) == ["I", "III"]

    def test_a_priority_lead_outside_the_group_is_passed_over(self):
        assert lsemvae_runner.withholding_order(("aVF", "III"), ("I", "II", "III")) == ["III", "II"]

    def test_the_last_lead_is_never_withheld(self):
        assert lsemvae_runner.withholding_order(("II", "I"), ("I", "II")) == ["II"]
        assert lsemvae_runner.withholding_order((), ("I",)) == []


class TestResolveRobustnessOrder:
    def test_empty_by_default(self, tmp_path):
        assert lsemvae_runner.resolve_robustness_order(lsemvae_cfg(tmp_path)) == ()

    def test_leads_are_matched_canonically(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ROBUSTNESS.ORDER": ["lead_ii", "i"]})
        assert lsemvae_runner.resolve_robustness_order(cfg) == ("II", "I")

    def test_a_lead_the_labelled_cohort_lacks_is_named(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ROBUSTNESS.ORDER": ["V6"]})
        with pytest.raises(ValueError, match=r"ROBUSTNESS.ORDER names \['V6'\]"):
            lsemvae_runner.resolve_robustness_order(cfg)

    def test_a_lead_named_twice_is_refused(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ROBUSTNESS.ORDER": ["I", "LEAD_I"]})
        with pytest.raises(ValueError, match="more than once"):
            lsemvae_runner.resolve_robustness_order(cfg)

    def test_a_bad_order_fails_the_run_before_anything_is_written(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ROBUSTNESS.ORDER": ["V6"]})
        with pytest.raises(ValueError, match="ROBUSTNESS.ORDER"):
            lsemvae_runner.run(cfg)
        assert not any(tmp_path.iterdir())


class TestWithholdingMetric:
    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, ("roc_auc", roc_auc, "label")),
            (
                {"ENDPOINT.LABEL_COLUMN": "subtype", "ENDPOINT.NUM_CLASSES": 3},
                ("roc_auc_ovr", multiclass_roc_auc, "label"),
            ),
            ({"ENDPOINT.REGRESSION": True}, ("r2", r2_score, "value")),
        ],
        ids=["binary", "three-class", "regression"],
    )
    def test_each_endpoint_is_rescored_with_its_threshold_free_metric(self, tmp_path, overrides, expected):
        assert lsemvae_runner.withholding_metric(lsemvae_cfg(tmp_path, **overrides)) == expected


def _curve(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.slow
class TestRobustnessCurve:
    def test_each_fold_is_pushed_down_to_one_lead(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ECG.FINETUNE_LEADS": ["I", "II", "III"]})
        lsemvae_runner.run(cfg)
        rows = _curve(tmp_path / "lsemvae" / "I+II+III" / "robustness.csv")

        for fold in ("0", "1"):
            steps = [row for row in rows if row["fold"] == fold]
            assert [row["leads"] for row in steps] == ["I+II+III", "I+II", "I"]
            assert [row["withheld"] for row in steps] == ["", "III", "III+II"]
            assert all(0.0 <= float(row["roc_auc"]) <= 1.0 for row in steps)

    def test_the_intact_step_is_the_folds_reported_score(self, tmp_path):
        # Step 0 withholds nothing, so it must be the very score metrics.json reports.
        import json

        cfg = lsemvae_cfg(tmp_path)
        lsemvae_runner.run(cfg)
        out = tmp_path / "lsemvae" / "I+II"
        intact = [float(row["roc_auc"]) for row in _curve(out / "robustness.csv") if row["step"] == "0"]
        folds = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["folds"]
        assert intact == pytest.approx([fold["roc_auc"] for fold in folds])

    def test_a_configured_order_is_followed(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ROBUSTNESS.ORDER": ["I"]})
        lsemvae_runner.run(cfg)
        rows = _curve(tmp_path / "lsemvae" / "I+II" / "robustness.csv")
        assert {row["leads"] for row in rows if row["step"] == "1"} == {"II"}

    def test_a_regression_curve_is_measured_in_r_squared(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ENDPOINT.REGRESSION": True})
        lsemvae_runner.run(cfg)
        rows = _curve(tmp_path / "lsemvae" / "I+II" / "robustness.csv")
        assert "r2" in rows[0] and "roc_auc" not in rows[0]

    def test_disabled_writes_no_curve(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ROBUSTNESS.ENABLED": False})
        lsemvae_runner.run(cfg)
        assert not (tmp_path / "lsemvae" / "I+II" / "robustness.csv").exists()

    def test_a_single_lead_group_has_no_curve_to_draw(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path)
        lsemvae_runner.run(cfg, lead_groups=[("I",)])
        assert not (tmp_path / "lsemvae" / "I" / "robustness.csv").exists()


@pytest.mark.slow
class TestReportsDoNotChangeTraining:
    """Turning a report on or off must not change the models a run trains.

    Every report iterates a loader, and every iteration draws from torch's generator;
    without isolation, a robustness pass after fold 0 reshuffles fold 1's training.
    """

    def test_lsemvae_folds_score_the_same_with_and_without_reports(self, tmp_path):
        reported = lsemvae_runner.run(
            lsemvae_cfg(tmp_path / "on", **{"ROBUSTNESS.ENABLED": True, "OUTPUT.INTERPRET": True}),
            lead_groups=[("I", "II"), ("I",)],
        )
        silent = lsemvae_runner.run(
            lsemvae_cfg(tmp_path / "off", **{"ROBUSTNESS.ENABLED": False, "OUTPUT.INTERPRET": False}),
            lead_groups=[("I", "II"), ("I",)],
        )
        for group in ("I+II", "I"):
            assert reported[group]["roc_auc"]["values"] == silent[group]["roc_auc"]["values"]

    def test_cardiovae_arms_score_the_same_with_and_without_reports(self, tmp_path):
        from examples.cardiovae_multimodal import runner as cardiovae_runner
        from tests.examples.test_example_workflows import cardiovae_cfg

        arms = [("ecg", "cxr"), ("ecg",)]
        reported = cardiovae_runner.run(cardiovae_cfg(tmp_path / "on", **{"OUTPUT.INTERPRET": True}), arms=arms)
        silent = cardiovae_runner.run(cardiovae_cfg(tmp_path / "off", **{"OUTPUT.INTERPRET": False}), arms=arms)
        for arm in ("ecg+cxr", "ecg"):
            assert reported[arm]["roc_auc"]["values"] == silent[arm]["roc_auc"]["values"]
