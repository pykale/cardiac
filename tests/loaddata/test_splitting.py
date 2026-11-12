"""Splitting: the three sets stay disjoint, and a subject never spans two of them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalecardiac.loaddata import (
    SPLIT_NAMES,
    CrossValidationSplit,
    HoldOutSplit,
    PredefinedSplit,
    SplitError,
    get_stratification_labels,
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


class TestGetStratificationLabels:
    def test_no_keys_gives_a_constant_label(self, table):
        assert np.unique(get_stratification_labels(table)).tolist() == [0]

    def test_several_keys_combine_into_one(self, table):
        labels = get_stratification_labels(table, ["label", "site"])
        assert len(np.unique(labels)) == 4

    def test_a_rare_combination_is_folded_rather_than_failing(self):
        frame = pd.DataFrame({"label": [0] * 10 + [1]})
        labels = get_stratification_labels(frame, ["label"], min_count=2)
        # The single positive joins the majority class rather than blocking the split.
        assert len(np.unique(labels)) == 1

    def test_an_unknown_column_is_named(self, table):
        with pytest.raises(SplitError, match="not columns"):
            get_stratification_labels(table, ["nonexistent"])


class TestHoldOutSplit:
    def test_produces_three_disjoint_named_sets(self, table):
        split = next(HoldOutSplit(group_by="subject_id", stratify_by=["label"], random_state=0).split(table))
        assert sorted(split) == sorted(SPLIT_NAMES)
        indices = [set(split[name].tolist()) for name in SPLIT_NAMES]
        assert not (indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])
        assert sum(len(part) for part in indices) == len(table)

    def test_a_subject_never_spans_two_sets(self, table):
        split = next(HoldOutSplit(group_by="subject_id", random_state=0).split(table))
        seen: dict[str, str] = {}
        for name in SPLIT_NAMES:
            for subject in table.iloc[split[name]]["subject_id"]:
                assert seen.setdefault(subject, name) == name

    def test_validation_comes_out_of_the_training_half(self, table):
        with_validation = next(
            HoldOutSplit(test_size=0.3, val_size=0.2, group_by="subject_id", random_state=0).split(table)
        )
        without = next(HoldOutSplit(test_size=0.3, val_size=0.0, group_by="subject_id", random_state=0).split(table))
        # The test set is identical either way; only the training half is carved up.
        assert set(with_validation["test"].tolist()) == set(without["test"].tolist())

    def test_zero_validation_gives_an_empty_set(self, table):
        split = next(HoldOutSplit(val_size=0.0, group_by="subject_id", random_state=0).split(table))
        assert split["val"].size == 0

    def test_the_same_seed_gives_the_same_split(self, table):
        first = next(HoldOutSplit(group_by="subject_id", random_state=7).split(table))
        second = next(HoldOutSplit(group_by="subject_id", random_state=7).split(table))
        assert all(np.array_equal(first[name], second[name]) for name in SPLIT_NAMES)

    @pytest.mark.parametrize("kwargs", [{"test_size": 0.0}, {"test_size": 1.0}, {"val_size": 0.6, "test_size": 0.5}])
    def test_impossible_shares_are_rejected(self, kwargs):
        with pytest.raises(SplitError):
            HoldOutSplit(**kwargs)

    def test_a_cohort_too_small_to_divide_is_rejected(self):
        with pytest.raises(SplitError, match="at least 3"):
            next(HoldOutSplit().split(pd.DataFrame({"label": [0, 1]})))

    def test_an_unknown_group_column_is_named(self, table):
        with pytest.raises(SplitError, match="group_by"):
            next(HoldOutSplit(group_by="patient").split(table))


class TestCrossValidationSplit:
    def test_every_row_is_tested_exactly_once(self, table):
        folds = list(CrossValidationSplit(n_splits=5, group_by="subject_id", random_state=0).split(table))
        assert len(folds) == 5
        tested = np.concatenate([fold["test"] for fold in folds])
        assert sorted(tested.tolist()) == list(range(len(table)))

    def test_no_subject_spans_train_and_test_within_a_fold(self, table):
        for fold in CrossValidationSplit(n_splits=5, group_by="subject_id", random_state=0).split(table):
            train = set(table.iloc[fold["train"]]["subject_id"])
            val = set(table.iloc[fold["val"]]["subject_id"])
            test = set(table.iloc[fold["test"]]["subject_id"])
            assert not (train & test or val & test or train & val)

    @pytest.mark.parametrize("seed", range(10))
    def test_every_fold_holds_the_cohorts_share_of_positives_whatever_the_seed(self, table, seed):
        # Fifteen positive subjects of thirty, five folds of six: three in each. Built on
        # StratifiedGroupKFold(shuffle=True), this failed for some seeds, because it
        # balances one permutation of the groups and returns another.
        splitter = CrossValidationSplit(n_splits=5, group_by="subject_id", stratify_by=["label"], random_state=seed)
        for fold in splitter.split(table):
            tested = table.iloc[fold["test"]].drop_duplicates("subject_id")
            assert (int(tested["label"].sum()), len(tested)) == (3, 6)

    def test_one_row_per_subject_is_stratified_too(self):
        # The examples' tables: seed 0 once put one positive in one fold and eleven in the other.
        frame = pd.DataFrame({"subject_id": [f"s{index:02d}" for index in range(24)], "label": [0, 1] * 12})
        for seed in range(20):
            splitter = CrossValidationSplit(n_splits=2, group_by="subject_id", stratify_by=["label"], random_state=seed)
            assert [int(frame.iloc[fold["test"]]["label"].sum()) for fold in splitter.split(frame)] == [6, 6]

    def test_rows_without_a_group_are_stratified_one_by_one(self):
        frame = pd.DataFrame({"label": [0, 1] * 12})
        folds = list(CrossValidationSplit(n_splits=3, stratify_by=["label"], random_state=0).split(frame))
        assert sorted(np.concatenate([fold["test"] for fold in folds]).tolist()) == list(range(24))
        assert [int(frame.iloc[fold["test"]]["label"].sum()) for fold in folds] == [4, 4, 4]

    def test_the_seed_decides_the_folds(self, table):
        def tested(seed):
            splitter = CrossValidationSplit(n_splits=5, group_by="subject_id", stratify_by=["label"], random_state=seed)
            return [sorted(fold["test"].tolist()) for fold in splitter.split(table)]

        assert tested(3) == tested(3)
        assert tested(3) != tested(4)

    def test_one_fold_is_not_a_cross_validation(self):
        with pytest.raises(SplitError, match="at least 2"):
            CrossValidationSplit(n_splits=1)


class TestPredefinedSplit:
    def test_reproduces_the_given_test_set_exactly(self, table):
        subjects = sorted(set(table["subject_id"]))
        assignment = {"train": subjects[:20], "test": subjects[20:]}
        split = next(PredefinedSplit(assignment, group_by="subject_id", random_state=0).split(table))
        assert set(table.iloc[split["test"]]["subject_id"]) == set(assignment["test"])

    def test_validation_is_drawn_only_from_the_training_half(self, table):
        subjects = sorted(set(table["subject_id"]))
        assignment = {"train": subjects[:20], "test": subjects[20:]}
        split = next(PredefinedSplit(assignment, group_by="subject_id", val_size=0.25, random_state=0).split(table))
        assert set(table.iloc[split["val"]]["subject_id"]) <= set(assignment["train"])

    def test_identifiers_not_in_the_table_are_ignored(self, table):
        subjects = sorted(set(table["subject_id"]))
        assignment = {"train": subjects[:20] + ["ghost"], "test": subjects[20:]}
        split = next(PredefinedSplit(assignment, group_by="subject_id", random_state=0).split(table))
        assert len(split["train"]) + len(split["val"]) == 40

    def test_an_assignment_matching_nothing_suggests_the_usual_cause(self, table):
        with pytest.raises(SplitError, match="zero padding"):
            next(PredefinedSplit({"train": ["x"], "test": ["y"]}).split(table))

    def test_an_unknown_id_column_is_named(self, table):
        with pytest.raises(SplitError, match="id_column"):
            next(PredefinedSplit({"train": ["a"], "test": ["b"]}, id_column="pid").split(table))


class TestSmallCohortsAndDescriptions:
    def test_an_impossible_validation_share_is_refused(self):
        with pytest.raises(SplitError, match="val_size must lie in"):
            CrossValidationSplit(n_splits=3, val_size=1.0)

    def test_a_splitter_describes_its_settings(self):
        assert repr(HoldOutSplit(test_size=0.25)).startswith("HoldOutSplit(val_size=0.15, group_by=None")

    def test_a_training_half_too_small_to_carve_is_named(self):
        frame = pd.DataFrame({"subject_id": ["a", "b", "c"]})
        with pytest.raises(SplitError, match="need at least 2 unique subject_id to hold part out, got 1"):
            next(HoldOutSplit(test_size=0.6, val_size=0.2, group_by="subject_id").split(frame))

    def test_rows_without_groups_are_split_one_by_one(self):
        frame = pd.DataFrame({"label": [0, 1] * 10})
        split = next(HoldOutSplit(test_size=0.25, val_size=0.2, random_state=0).split(frame))
        assert sorted(np.concatenate([split[name] for name in SPLIT_NAMES]).tolist()) == list(range(20))
        assert len(split["test"]) == 5

    def test_a_hold_out_too_small_to_stratify_shuffles_rather_than_fails(self):
        # One held-out row cannot carry both classes, so stratification is dropped.
        frame = pd.DataFrame({"label": [0, 1] * 5})
        split = next(HoldOutSplit(test_size=0.1, val_size=0.0, stratify_by=["label"], random_state=0).split(frame))
        assert len(split["test"]) == 1 and len(split["train"]) == 9
