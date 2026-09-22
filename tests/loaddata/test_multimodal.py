"""The multimodal cohort contract: assembly, missing modalities, collation, targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from kalecardiac.loaddata import (
    LABEL_KEY,
    ArraySource,
    ColumnTarget,
    ECGArraySource,
    ImageArraySource,
    MultimodalDataset,
    SubjectSample,
    check_target,
    collate_subjects,
    lead_sources,
)


class TestArraySource:
    def test_infers_its_shape_from_the_values(self):
        source = ArraySource({"a": np.zeros(5), "b": np.ones(5)})
        assert source.shape == (5,)
        assert source.placeholder().shape == (5,)

    def test_values_of_differing_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            ArraySource({"a": np.zeros(5), "b": np.zeros(6)})

    def test_empty_mapping_needs_a_declared_shape(self):
        with pytest.raises(ValueError, match="shape must be given"):
            ArraySource({})
        assert ArraySource({}, shape=(4,)).placeholder().shape == (4,)

    def test_absent_subject_is_missing(self):
        assert ArraySource({"a": np.zeros(3)}).get("b", 0) is None


class TestMultimodalDataset:
    def test_assembles_every_modality_for_a_subject(self, multimodal_dataset, cohort):
        sample = multimodal_dataset[0]
        assert isinstance(sample, SubjectSample)
        assert sample.subject_id == cohort.subject_id[0]
        assert sorted(sample.modalities) == ["ecg", "image"]
        assert sample.target[LABEL_KEY].item() == cohort.labels[sample.subject_id]

    def test_present_is_true_where_the_source_had_something(self, multimodal_dataset):
        sample = multimodal_dataset[0]
        assert all(bool(flag) for flag in sample.present.values())

    def test_a_missing_modality_is_a_zero_placeholder_marked_absent(self, cohort, ecg_source, label_target):
        # One subject has no image; the batch must stay rectangular and say so.
        partial = {name: cohort.image[name] for name in cohort.subject_id[1:]}
        images = ImageArraySource(partial, channels=1, size=cohort.image[cohort.subject_id[0]].shape[1:])
        dataset = MultimodalDataset(cohort.subject_id, {"ecg": ecg_source, "image": images}, target=label_target)
        sample = dataset[0]
        assert bool(sample.present["ecg"]) is True
        assert bool(sample.present["image"]) is False
        assert torch.equal(sample.modalities["image"], torch.zeros_like(sample.modalities["image"]))

    def test_the_same_keys_appear_for_every_subject(self, lead_dataset):
        keys = {tuple(sorted(lead_dataset[index].modalities)) for index in range(len(lead_dataset))}
        assert len(keys) == 1

    def test_no_source_is_rejected(self, cohort):
        with pytest.raises(ValueError, match="at least one modality source"):
            MultimodalDataset(cohort.subject_id, {})

    def test_repeated_identifiers_are_rejected(self, cohort, ecg_source):
        # A subject appearing twice would be scored twice and could land in two splits.
        with pytest.raises(ValueError, match="repeat"):
            MultimodalDataset([cohort.subject_id[0], cohort.subject_id[0]], {"ecg": ecg_source})

    def test_metadata_from_names_a_known_source(self, cohort, ecg_source):
        with pytest.raises(ValueError, match="not sources"):
            MultimodalDataset(cohort.subject_id, {"ecg": ecg_source}, metadata_from=["cxr"])

    def test_an_unlabelled_cohort_carries_no_target(self, cohort, ecg_source):
        dataset = MultimodalDataset(cohort.subject_id, {"ecg": ecg_source})
        assert dataset[0].target == {}

    def test_lead_cohort_has_one_modality_per_lead(self, lead_dataset, cohort):
        assert sorted(lead_dataset[0].modalities) == sorted(cohort.leads)


class TestCollation:
    def test_stacks_a_batch_with_a_leading_dimension(self, multimodal_dataset, cohort):
        batch = collate_subjects([multimodal_dataset[index] for index in range(4)])
        assert len(batch) == 4
        assert batch.modalities["ecg"].shape[0] == 4
        assert batch.present["ecg"].shape == (4,)
        assert batch.target[LABEL_KEY].shape == (4,)
        assert batch.subject_id == cohort.subject_id[:4]

    def test_an_empty_batch_is_rejected(self):
        with pytest.raises(ValueError, match="empty list"):
            collate_subjects([])

    def test_samples_disagreeing_about_modalities_are_rejected(self, multimodal_dataset):
        first, second = multimodal_dataset[0], multimodal_dataset[1]
        second.modalities.pop("image")
        second.present.pop("image")
        with pytest.raises(ValueError, match="disagree about modalities"):
            collate_subjects([first, second])

    def test_a_shape_mismatch_names_the_modality_and_the_fix(self, multimodal_dataset):
        first, second = multimodal_dataset[0], multimodal_dataset[1]
        second.modalities["ecg"] = second.modalities["ecg"][:, :-1]
        with pytest.raises(ValueError, match="kalecardiac.prepdata"):
            collate_subjects([first, second])

    def test_metadata_is_carried_per_subject_in_batch_order(self, cohort, ecg_source, label_target):
        dataset = MultimodalDataset(cohort.subject_id, {"ecg": ecg_source}, target=label_target)
        batch = collate_subjects([dataset[0], dataset[1]])
        assert batch.metadata["sampling_rate"] == [cohort.sampling_rate] * 2

    def test_to_device_moves_tensors_and_keeps_metadata(self, multimodal_dataset):
        batch = collate_subjects([multimodal_dataset[0], multimodal_dataset[1]])
        moved = batch.to("cpu")
        assert moved.subject_id == batch.subject_id
        assert moved.modalities["ecg"].device.type == "cpu"


class TestColumnTarget:
    def test_maps_batch_keys_to_named_columns(self):
        frame = pd.DataFrame({"subject_id": ["a", "b"], "mpap_mmhg": [18.0, 31.0]})
        target = ColumnTarget(frame, columns={"value": "mpap_mmhg"})
        assert target.required_columns == ("mpap_mmhg",)
        assert target.for_("b")["value"].item() == pytest.approx(31.0)

    def test_batched_lookup_agrees_with_single_lookup(self, label_target, cohort):
        identifiers = cohort.subject_id[:5]
        batched = label_target.values_for(identifiers)[LABEL_KEY]
        singly = torch.stack([label_target.for_(name)[LABEL_KEY] for name in identifiers])
        assert torch.equal(batched, singly)

    def test_a_missing_column_is_named(self):
        frame = pd.DataFrame({"subject_id": ["a"], "x": [1]})
        with pytest.raises(KeyError, match="no column"):
            ColumnTarget(frame, columns={"label": "y"})

    def test_duplicate_identifiers_are_rejected(self):
        frame = pd.DataFrame({"subject_id": ["a", "a"], "label": [0, 1]})
        with pytest.raises(KeyError, match="must be unique"):
            ColumnTarget(frame, columns={"label": "label"})

    def test_a_missing_label_is_not_silently_a_negative(self):
        # An unknown outcome is not a negative one; deciding that is the cohort's job.
        frame = pd.DataFrame({"subject_id": ["a", "b"], "label": [1, None]})
        with pytest.raises(ValueError, match="not a negative one"):
            ColumnTarget(frame, columns={"label": "label"})

    def test_from_mapping_needs_agreeing_subjects(self):
        with pytest.raises(ValueError, match="same subjects"):
            ColumnTarget.from_mapping({"label": {"a": 1}, "value": {"b": 2.0}})

    def test_from_mapping_needs_at_least_one_key(self):
        with pytest.raises(ValueError, match="at least one target key"):
            ColumnTarget.from_mapping({})

    def test_frame_returns_a_table_a_splitter_can_stratify_on(self, label_target, cohort):
        frame = label_target.frame(cohort.subject_id)
        assert list(frame.columns) == ["subject_id", LABEL_KEY]
        assert len(frame) == len(cohort.subject_id)

    def test_stratify_labels_prefers_the_label_key(self, cohort):
        target = ColumnTarget.from_mapping({"value": cohort.values, "label": cohort.labels})
        labels = target.stratify_labels(cohort.subject_id)
        assert set(np.unique(labels)) <= {0.0, 1.0}


class TestTargetContract:
    def test_a_conforming_target_passes(self, label_target):
        check_target(label_target)

    def test_a_non_conforming_object_is_rejected_by_name(self):
        class Incomplete:
            def for_(self, identifier):
                return {}

        with pytest.raises(TypeError, match="required_columns"):
            check_target(Incomplete())


def test_ecg_and_image_cohorts_are_the_same_class(cohort, ecg_source, image_source, label_target):
    """Adding a modality is a dictionary entry, never a new dataset class."""
    unimodal = MultimodalDataset(cohort.subject_id, {"ecg": ecg_source}, target=label_target)
    bimodal = MultimodalDataset(cohort.subject_id, {"ecg": ecg_source, "image": image_source}, target=label_target)
    per_lead = MultimodalDataset(cohort.subject_id, lead_sources(ecg_source), target=label_target)
    assert type(unimodal) is type(bimodal) is type(per_lead)
    assert len(unimodal[0].modalities) == 1
    assert len(bimodal[0].modalities) == 2
    assert len(per_lead[0].modalities) == len(cohort.leads)


def test_array_source_serves_a_clinical_vector(cohort, ecg_source, label_target):
    """A tabular modality needs no new source type."""
    vectors = {name: np.array([index % 3, 1.0]) for index, name in enumerate(cohort.subject_id)}
    dataset = MultimodalDataset(
        cohort.subject_id,
        {"ecg": ecg_source, "clinical": ArraySource(vectors)},
        target=label_target,
    )
    batch = collate_subjects([dataset[index] for index in range(3)])
    assert batch.modalities["clinical"].shape == (3, 2)


def test_ecg_source_accepts_torch_and_numpy_alike(cohort):
    """Tensors and arrays are both ordinary inputs."""
    as_tensors = {name: torch.from_numpy(value) for name, value in cohort.ecg.items()}
    source = ECGArraySource(as_tensors, leads=cohort.leads)
    assert source.get(cohort.subject_id[0], 0).shape == (len(cohort.leads), source.length)
