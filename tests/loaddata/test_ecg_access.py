"""ECG access: lead naming, shape validation, and splitting a recording into leads."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kalecardiac.loaddata import (
    LIMB_LEADS,
    PRECORDIAL_LEADS,
    STANDARD_12_LEAD,
    ECGArraySource,
    ECGFileSource,
    ECGFormatError,
    LeadSource,
    as_lead_array,
    canonical_lead,
    canonical_leads,
    lead_indices,
    lead_sources,
)


class TestLeadNaming:
    @pytest.mark.parametrize("spelling", ["aVR", "AVR", "avr", "LEAD_aVR", "lead_AVR", " aVR "])
    def test_spellings_canonicalise_to_one_name(self, spelling):
        assert canonical_lead(spelling) == "aVR"

    def test_unknown_name_keeps_its_own_spelling(self):
        # A cohort may record a lead this package has never heard of; only the export
        # prefix is stripped.
        assert canonical_lead("LEAD_X7") == "X7"

    def test_standard_order_is_limb_then_precordial(self):
        assert STANDARD_12_LEAD == LIMB_LEADS + PRECORDIAL_LEADS
        assert len(STANDARD_12_LEAD) == 12

    def test_empty_lead_set_is_rejected(self):
        with pytest.raises(ECGFormatError, match="must declare its leads"):
            canonical_leads([])

    def test_two_spellings_of_one_lead_are_rejected(self):
        # Choosing which copy is real is the cohort's decision, not this function's.
        with pytest.raises(ECGFormatError, match="more than once"):
            canonical_leads(["I", "LEAD_I"])

    def test_indices_follow_the_requested_order_not_storage_order(self):
        assert lead_indices(STANDARD_12_LEAD, ["II", "I", "V6"]) == [1, 0, 11]

    def test_missing_lead_is_named_in_the_error(self):
        with pytest.raises(ECGFormatError, match=r"\['V1'\]"):
            lead_indices(LIMB_LEADS, ["I", "V1"])


class TestLeadArrayCoercion:
    def test_transposes_a_samples_by_leads_recording(self):
        array = as_lead_array(np.zeros((500, 12)), num_leads=12)
        assert array.shape == (12, 500)

    def test_leaves_a_leads_by_samples_recording_alone(self):
        assert as_lead_array(np.zeros((12, 500)), num_leads=12).shape == (12, 500)

    def test_accepts_one_dimensional_input_for_a_single_lead(self):
        assert as_lead_array(np.zeros(500), num_leads=1).shape == (1, 500)

    def test_rejects_one_dimensional_input_for_several_leads(self):
        with pytest.raises(ECGFormatError, match="1-D recording holds one lead"):
            as_lead_array(np.zeros(500), num_leads=12)

    def test_square_recording_is_not_guessed_at(self):
        # 12x12 has an axis of length 12 either way; taking the first is the documented
        # behaviour, and it must not silently transpose.
        assert as_lead_array(np.arange(144).reshape(12, 12), num_leads=12)[0, 1] == 1

    def test_shape_with_no_matching_axis_is_rejected(self):
        with pytest.raises(ECGFormatError, match="no axis of length 12"):
            as_lead_array(np.zeros((6, 500)), num_leads=12)


class TestECGArraySource:
    def test_reports_shape_and_rate(self, cohort, ecg_source):
        assert ecg_source.num_leads == len(cohort.leads)
        assert ecg_source.length == cohort.ecg[cohort.subject_id[0]].shape[1]
        assert ecg_source.sampling_rate == cohort.sampling_rate

    def test_returns_a_tensor_of_the_declared_shape(self, cohort, ecg_source):
        recording = ecg_source.get(cohort.subject_id[0], 0)
        assert isinstance(recording, torch.Tensor)
        assert recording.shape == (len(cohort.leads), ecg_source.length)
        assert recording.dtype == torch.float32

    def test_absent_subject_is_missing_rather_than_an_error(self, ecg_source):
        assert ecg_source.get("nobody", 0) is None

    def test_placeholder_matches_the_real_shape(self, ecg_source, cohort):
        placeholder = ecg_source.placeholder()
        assert placeholder.shape == ecg_source.get(cohort.subject_id[0], 0).shape
        assert torch.equal(placeholder, torch.zeros_like(placeholder))

    def test_selection_reorders_as_asked(self, cohort):
        source = ECGArraySource(cohort.ecg, leads=cohort.leads, select=["II", "I"])
        assert source.leads == ("II", "I")
        recording = source.get(cohort.subject_id[0], 0)
        full = ECGArraySource(cohort.ecg, leads=cohort.leads).get(cohort.subject_id[0], 0)
        assert torch.equal(recording[0], full[1])
        assert torch.equal(recording[1], full[0])

    def test_selecting_an_absent_lead_is_rejected(self, cohort):
        with pytest.raises(ECGFormatError):
            ECGArraySource(cohort.ecg, leads=cohort.leads, select=["V1"])

    def test_transform_is_applied(self, cohort):
        source = ECGArraySource(cohort.ecg, leads=cohort.leads, transform=lambda array: array[:, :64])
        assert source.length == 64
        assert source.get(cohort.subject_id[0], 0).shape[1] == 64

    def test_a_transform_may_reshape_the_channels(self, cohort):
        # Flattening every lead into one long signal is what the CardioVAE encoder
        # consumes, and deriving the limb leads from two recorded ones changes the
        # count the other way. Both are ordinary preprocessing.
        source = ECGArraySource(cohort.ecg, leads=cohort.leads, transform=lambda array: array.reshape(1, -1))
        assert source.channels == 1
        assert source.length == len(cohort.leads) * cohort.ecg[cohort.subject_id[0]].shape[1]
        assert source.get(cohort.subject_id[0], 0).shape == (1, source.length)

    def test_the_measured_shape_is_what_later_subjects_are_held_to(self, cohort):
        # Measured from the first recording, then enforced -- so a transform that is
        # inconsistent between subjects fails at the subject that caused it, rather
        # than at a collation that cannot say which subject was wrong.
        calls = {"n": 0}

        def drifting(array):
            calls["n"] += 1
            return array[:, :32] if calls["n"] <= 2 else array[:, :16]

        source = ECGArraySource(cohort.ecg, leads=cohort.leads, transform=drifting)
        assert source.length == 32
        with pytest.raises(ECGFormatError, match="must reach a batch at one shape"):
            for index, identifier in enumerate(cohort.subject_id):
                source.get(identifier, index)

    def test_a_non_two_dimensional_transform_output_is_rejected(self, cohort):
        with pytest.raises(ECGFormatError, match="2-D"):
            ECGArraySource(cohort.ecg, leads=cohort.leads, transform=lambda array: array[0, :32])

    def test_placeholder_follows_the_transformed_shape(self, cohort):
        source = ECGArraySource(cohort.ecg, leads=cohort.leads, transform=lambda array: array.reshape(1, -1))
        assert source.placeholder().shape == (1, source.length)

    def test_a_recording_of_the_wrong_length_is_rejected_by_name(self, cohort):
        recordings = dict(cohort.ecg)
        odd = cohort.subject_id[1]
        recordings[odd] = recordings[odd][:, :-3]
        source = ECGArraySource(recordings, leads=cohort.leads, length=cohort.ecg[cohort.subject_id[0]].shape[1])
        with pytest.raises(ECGFormatError, match=odd):
            source.get(odd, 1)

    def test_empty_cohort_needs_a_declared_length(self, cohort):
        with pytest.raises(ECGFormatError, match="length must be given"):
            ECGArraySource({}, leads=cohort.leads)

    def test_provenance_carries_the_rate_and_lead_order(self, cohort, ecg_source):
        provenance = ecg_source.provenance(cohort.subject_id[0], 0)
        assert provenance["sampling_rate"] == cohort.sampling_rate
        assert provenance["leads"] == tuple(cohort.leads)


class TestECGFileSource:
    @pytest.fixture
    def written(self, cohort, tmp_path):
        paths = {}
        for identifier in cohort.subject_id[:6]:
            path = tmp_path / f"{identifier}.npy"
            np.save(path, cohort.ecg[identifier])
            paths[identifier] = path
        return paths

    def test_reads_the_same_values_as_the_array_source(self, cohort, written):
        source = ECGFileSource(written, leads=cohort.leads)
        identifier = cohort.subject_id[0]
        assert torch.allclose(source.get(identifier, 0), torch.from_numpy(cohort.ecg[identifier]))

    def test_construction_does_not_read_every_file(self, cohort, written, monkeypatch):
        reads = []

        def counting_loader(path):
            reads.append(path)
            return np.load(path)

        ECGFileSource(written, leads=cohort.leads, loader=counting_loader)
        # Exactly one file, to infer the length -- not the whole cohort.
        assert len(reads) == 1

    def test_lead_splitting_reads_each_subject_once(self, cohort, written):
        source = ECGFileSource(written, leads=cohort.leads)
        source._cached = None
        reads = []
        original = source.loader
        source.loader = lambda path: (reads.append(path), original(path))[1]

        identifier = cohort.subject_id[0]
        for lead in lead_sources(source).values():
            lead.get(identifier, 0)
        assert len(reads) == 1

    def test_unknown_extension_asks_for_a_loader(self, cohort, tmp_path):
        path = tmp_path / "x.dcm"
        path.write_bytes(b"")
        with pytest.raises(ECGFormatError, match="pass loader="):
            ECGFileSource({"a": path}, leads=["I"])


class TestLeadSources:
    def test_each_lead_becomes_a_single_channel_modality(self, cohort, ecg_source):
        sources = lead_sources(ecg_source)
        assert sorted(sources) == sorted(cohort.leads)
        for lead, source in sources.items():
            value = source.get(cohort.subject_id[0], 0)
            assert value.shape == (1, ecg_source.length), lead

    def test_a_lead_holds_the_row_the_recording_holds(self, cohort, ecg_source):
        full = ecg_source.get(cohort.subject_id[0], 0)
        for index, lead in enumerate(cohort.leads):
            taken = LeadSource(ecg_source, lead).get(cohort.subject_id[0], 0)
            assert torch.equal(taken[0], full[index])

    def test_a_subset_can_be_requested_in_its_own_order(self, cohort, ecg_source):
        sources = lead_sources(ecg_source, leads=["aVF", "I"])
        assert list(sources) == ["aVF", "I"]

    def test_prefix_names_a_second_recording(self, cohort, ecg_source):
        sources = lead_sources(ecg_source, leads=["I"], prefix="rest_")
        assert list(sources) == ["rest_I"]

    def test_absent_subject_stays_absent_per_lead(self, ecg_source):
        assert LeadSource(ecg_source, "I").get("nobody", 0) is None

    def test_requesting_a_lead_the_source_lacks_is_rejected(self, ecg_source):
        with pytest.raises(ECGFormatError):
            LeadSource(ecg_source, "V3")

    def test_splitting_a_flattened_source_by_lead_is_refused(self, cohort):
        # Once the leads have been concatenated into one signal, "lead aVR of the
        # output" is asking for something that is no longer there.
        flattened = ECGArraySource(cohort.ecg, leads=cohort.leads, transform=lambda array: array.reshape(1, -1))
        assert flattened.splits_by_lead is False
        with pytest.raises(ECGFormatError, match="no longer leads"):
            LeadSource(flattened, "I")

    def test_an_untransformed_source_splits_by_lead(self, ecg_source):
        assert ecg_source.splits_by_lead is True
