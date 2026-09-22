"""Splitting: the three sets stay disjoint, and a subject never spans two of them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalecardiac.loaddata import (
    SPLIT_NAMES,
    CrossValidation,
    HoldOut,
    Predefined,
    SplitError,
    composite_labels,
    subject_frame,
    train_test_split,
)


@pytest.fixture
def table():
    """Sixty rows over thirty subjects: two recordings each, which is the leakage case."""
    subjects = [f"S{index:02d}" for index in range(30)]
    return pd.DataFrame(
        {
            "subject_id": [subject for subject in subjects for _ in range(2)],
            "label": [index % 2 for index in range(30) for _ in range(2)],
            "site": [("a" if index < 20 else "b") for index in range(30) for _ in range(2)],
        }
    )


class TestCompositeLabels:
    def test_no_keys_gives_a_constant_label(self, table):
        assert np.unique(composite_labels(table)).tolist() == [0]

    def test_several_keys_combine_into_one(self, table):
        labels = composite_labels(table, ["label", "site"])
        assert len(np.unique(labels)) == 4

    def test_a_rare_combination_is_folded_rather_than_failing(self):
        frame = pd.DataFrame({"label": [0] * 10 + [1]})
        labels = composite_labels(frame, ["label"], min_count=2)
        # The single positive joins the majority class rather than blocking the split.
        assert len(np.unique(labels)) == 1

    def test_an_unknown_column_is_named(self, table):
        with pytest.raises(SplitError, match="not columns"):
            composite_labels(table, ["nonexistent"])


class TestHoldOut:
    def test_produces_three_disjoint_named_sets(self, table):
        split = next(HoldOut(group_by="subject_id", stratify_by=["label"], random_state=0).split(table))
        assert sorted(split) == sorted(SPLIT_NAMES)
        indices = [set(split[name].tolist()) for name in SPLIT_NAMES]
        assert not (indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])
        assert sum(len(part) for part in indices) == len(table)

    def test_a_subject_never_spans_two_sets(self, table):
        split = next(HoldOut(group_by="subject_id", random_state=0).split(table))
        seen: dict[str, str] = {}
        for name in SPLIT_NAMES:
            for subject in table.iloc[split[name]]["subject_id"]:
                assert seen.setdefault(subject, name) == name

    def test_validation_comes_out_of_the_training_half(self, table):
        with_validation = next(HoldOut(test_size=0.3, val_size=0.2, group_by="subject_id", random_state=0).split(table))
        without = next(HoldOut(test_size=0.3, val_size=0.0, group_by="subject_id", random_state=0).split(table))
        # The test set is identical either way; only the training half is carved up.
        assert set(with_validation["test"].tolist()) == set(without["test"].tolist())

    def test_zero_validation_gives_an_empty_set(self, table):
        split = next(HoldOut(val_size=0.0, group_by="subject_id", random_state=0).split(table))
        assert split["val"].size == 0

    def test_the_same_seed_gives_the_same_split(self, table):
        first = next(HoldOut(group_by="subject_id", random_state=7).split(table))
        second = next(HoldOut(group_by="subject_id", random_state=7).split(table))
        assert all(np.array_equal(first[name], second[name]) for name in SPLIT_NAMES)

    @pytest.mark.parametrize("kwargs", [{"test_size": 0.0}, {"test_size": 1.0}, {"val_size": 0.6, "test_size": 0.5}])
    def test_impossible_shares_are_rejected(self, kwargs):
        with pytest.raises(SplitError):
            HoldOut(**kwargs)

    def test_a_cohort_too_small_to_divide_is_rejected(self):
        with pytest.raises(SplitError, match="at least 3"):
            next(HoldOut().split(pd.DataFrame({"label": [0, 1]})))

    def test_an_unknown_group_column_is_named(self, table):
        with pytest.raises(SplitError, match="group_by"):
            next(HoldOut(group_by="patient").split(table))


class TestCrossValidation:
    def test_every_row_is_tested_exactly_once(self, table):
        folds = list(CrossValidation(n_splits=5, group_by="subject_id", random_state=0).split(table))
        assert len(folds) == 5
        tested = np.concatenate([fold["test"] for fold in folds])
        assert sorted(tested.tolist()) == list(range(len(table)))

    def test_no_subject_spans_train_and_test_within_a_fold(self, table):
        for fold in CrossValidation(n_splits=5, group_by="subject_id", random_state=0).split(table):
            train = set(table.iloc[fold["train"]]["subject_id"])
            val = set(table.iloc[fold["val"]]["subject_id"])
            test = set(table.iloc[fold["test"]]["subject_id"])
            assert not (train & test or val & test or train & val)

    def test_every_fold_keeps_both_classes(self, table):
        # Grouping constrains stratification -- whole subjects move together, so with
        # six subjects per fold the rates cannot match exactly. What must hold is that
        # each fold is still scorable, and that the folds average to the cohort rate.
        rates = [
            table.iloc[fold["test"]]["label"].mean()
            for fold in CrossValidation(n_splits=5, group_by="subject_id", stratify_by=["label"], random_state=0).split(
                table
            )
        ]
        assert all(0.0 < rate < 1.0 for rate in rates)
        assert np.mean(rates) == pytest.approx(table["label"].mean(), abs=0.1)

    def test_one_fold_is_not_a_cross_validation(self):
        with pytest.raises(SplitError, match="at least 2"):
            CrossValidation(n_splits=1)


class TestPredefined:
    def test_reproduces_the_given_test_set_exactly(self, table):
        subjects = sorted(set(table["subject_id"]))
        assignment = {"train": subjects[:20], "test": subjects[20:]}
        split = next(Predefined(assignment, group_by="subject_id", random_state=0).split(table))
        assert set(table.iloc[split["test"]]["subject_id"]) == set(assignment["test"])

    def test_validation_is_drawn_only_from_the_training_half(self, table):
        subjects = sorted(set(table["subject_id"]))
        assignment = {"train": subjects[:20], "test": subjects[20:]}
        split = next(Predefined(assignment, group_by="subject_id", val_size=0.25, random_state=0).split(table))
        assert set(table.iloc[split["val"]]["subject_id"]) <= set(assignment["train"])

    def test_identifiers_not_in_the_table_are_ignored(self, table):
        subjects = sorted(set(table["subject_id"]))
        assignment = {"train": subjects[:20] + ["ghost"], "test": subjects[20:]}
        split = next(Predefined(assignment, group_by="subject_id", random_state=0).split(table))
        assert len(split["train"]) + len(split["val"]) == 40

    def test_an_assignment_matching_nothing_suggests_the_usual_cause(self, table):
        with pytest.raises(SplitError, match="zero padding"):
            next(Predefined({"train": ["x"], "test": ["y"]}).split(table))

    def test_an_unknown_id_column_is_named(self, table):
        with pytest.raises(SplitError, match="id_column"):
            next(Predefined({"train": ["a"], "test": ["b"]}, id_column="pid").split(table))


class TestTrainTestSplit:
    def test_returns_two_disjoint_index_arrays(self, table):
        kept, held = train_test_split(table, test_size=0.25, group_by="subject_id", random_state=0)
        assert not set(kept.tolist()) & set(held.tolist())
        assert len(kept) + len(held) == len(table)

    def test_an_impossible_share_is_rejected(self, table):
        with pytest.raises(SplitError):
            train_test_split(table, test_size=1.5)


class TestSubjectFrame:
    def test_builds_a_table_from_identifiers_and_labels(self):
        frame = subject_frame(["a", "b"], label=[0, 1])
        assert list(frame.columns) == ["subject_id", "label"]
        assert frame["subject_id"].tolist() == ["a", "b"]

    def test_a_mismatched_column_is_rejected(self):
        with pytest.raises(SplitError, match="values for"):
            subject_frame(["a", "b"], label=[0])


def test_splitting_a_synthetic_cohort_keeps_both_classes(cohort, label_target):
    """The end-to-end case: fold a real cohort table and check it stays scorable."""
    frame = label_target.frame(cohort.subject_id)
    for fold in CrossValidation(n_splits=5, group_by="subject_id", stratify_by=["label"], random_state=0).split(frame):
        assert frame.iloc[fold["test"]]["label"].nunique() == 2
