"""ECG transforms: lead selection, lead derivation, and the assembled pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from kalecardiac.loaddata import LIMB_LEADS, STANDARD_12_LEAD, ECGFormatError
from kalecardiac.prepdata import (
    derive_limb_leads,
    get_ecg_transform,
    get_lead_selector,
    interleave_leads,
    select_leads,
)


@pytest.fixture
def twelve_lead():
    generator = np.random.default_rng(0)
    return generator.normal(size=(12, 200)).astype(np.float32)


class TestSelectLeads:
    def test_takes_the_named_rows(self, twelve_lead):
        selected = select_leads(twelve_lead, STANDARD_12_LEAD, ["I", "V1"])
        assert selected.shape == (2, 200)
        assert np.array_equal(selected[0], twelve_lead[0])
        assert np.array_equal(selected[1], twelve_lead[6])

    def test_returns_them_in_the_requested_order(self, twelve_lead):
        selected = select_leads(twelve_lead, STANDARD_12_LEAD, ["V1", "I"])
        assert np.array_equal(selected[0], twelve_lead[6])
        assert np.array_equal(selected[1], twelve_lead[0])

    def test_matches_names_canonically(self, twelve_lead):
        # The same lead is exported as aVR, AVR and LEAD_aVR by three different tools.
        by_prefix = select_leads(twelve_lead, STANDARD_12_LEAD, ["LEAD_aVR"])
        by_case = select_leads(twelve_lead, STANDARD_12_LEAD, ["avr"])
        assert np.array_equal(by_prefix, by_case)

    def test_a_row_count_mismatch_is_rejected(self, twelve_lead):
        with pytest.raises(ECGFormatError, match="does not match"):
            select_leads(twelve_lead, LIMB_LEADS, ["I"])

    def test_an_absent_lead_is_rejected(self, twelve_lead):
        with pytest.raises(ECGFormatError):
            select_leads(twelve_lead, LIMB_LEADS, ["V1"])


class TestDeriveLimbLeads:
    @pytest.fixture
    def two_lead(self):
        generator = np.random.default_rng(1)
        return generator.normal(size=(2, 100)).astype(np.float32)

    def test_einthoven_relation_holds_for_lead_three(self, two_lead):
        derived = derive_limb_leads(two_lead, ["I", "II"], ["III"])
        assert np.allclose(derived[0], two_lead[1] - two_lead[0], atol=1e-5)

    def test_goldberger_relations_hold_for_the_augmented_leads(self, two_lead):
        lead_i, lead_ii = two_lead
        derived = derive_limb_leads(two_lead, ["I", "II"], ["aVR", "aVL", "aVF"])
        assert np.allclose(derived[0], -(lead_i + lead_ii) / 2.0, atol=1e-5)
        assert np.allclose(derived[1], lead_i - lead_ii / 2.0, atol=1e-5)
        assert np.allclose(derived[2], lead_ii - lead_i / 2.0, atol=1e-5)

    def test_the_six_limb_leads_sum_as_theory_requires(self, two_lead):
        # aVR + aVL + aVF = 0 for any consistent frontal-plane reconstruction.
        derived = derive_limb_leads(two_lead, ["I", "II"], LIMB_LEADS)
        assert np.allclose(derived[3] + derived[4] + derived[5], 0.0, atol=1e-5)

    def test_present_leads_are_passed_through_unchanged(self, twelve_lead):
        derived = derive_limb_leads(twelve_lead, STANDARD_12_LEAD, ["I", "aVR"])
        assert np.array_equal(derived[0], twelve_lead[0])
        assert np.array_equal(derived[1], twelve_lead[3])

    def test_a_precordial_lead_cannot_be_derived(self, two_lead):
        with pytest.raises(ECGFormatError, match="plane the limb leads do not"):
            derive_limb_leads(two_lead, ["I", "II"], ["V1"])

    def test_derivation_needs_both_source_leads(self):
        signal = np.zeros((1, 10), dtype=np.float32)
        with pytest.raises(ECGFormatError, match="needs both I and II"):
            derive_limb_leads(signal, ["I"], ["III"])


class TestLeadSelector:
    def test_returns_a_one_argument_callable(self, twelve_lead):
        select = get_lead_selector(STANDARD_12_LEAD, ["II", "V6"])
        assert select(twelve_lead).shape == (2, 200)

    def test_names_itself_for_a_pipeline_repr(self):
        assert "II" in get_lead_selector(STANDARD_12_LEAD, ["II"]).__name__

    def test_derive_mode_reconstructs_a_missing_lead(self):
        signal = np.random.default_rng(2).normal(size=(2, 50)).astype(np.float32)
        select = get_lead_selector(["I", "II"], LIMB_LEADS, derive=True)
        assert select(signal).shape == (6, 50)


class TestInterleaveLeads:
    def test_the_strip_holds_every_leads_sample_before_the_next_sample(self):
        # The reference implementation flattens a (samples, leads) recording row by row.
        recording = np.arange(12, dtype=np.float32).reshape(3, 4)
        strip = interleave_leads(recording)
        assert strip.shape == (1, 12)
        assert np.array_equal(strip[0], recording.T.reshape(-1))
        assert np.array_equal(strip[0, :3], recording[:, 0])

    def test_a_lead_reads_back_out_of_the_strip(self, twelve_lead):
        assert np.array_equal(interleave_leads(twelve_lead).reshape(-1, 12)[:, 4], twelve_lead[4])

    def test_interleaving_comes_after_every_per_lead_step(self):
        # A lead recorded at ten times another's amplitude comes out on the same scale,
        # because each lead is normalized on its own before the leads are interleaved.
        generator = np.random.default_rng(0)
        recording = np.vstack([generator.normal(size=150), 10 * generator.normal(size=150)])
        transform = get_ecg_transform(length=100, interleave=True)
        strip = transform(recording)
        assert transform.transforms[-1] is interleave_leads
        assert strip.shape == (1, 200)
        assert np.allclose(strip[0].reshape(100, 2).std(axis=0), 1.0, atol=1e-4)


class TestBuildEcgPipeline:
    def test_produces_the_requested_length_and_normalisation(self, twelve_lead):
        pipeline = get_ecg_transform(length=128, sampling_rate=500, source_sampling_rate=250)
        result = pipeline(twelve_lead)
        assert result.shape == (12, 128)
        assert np.allclose(result.mean(axis=1), 0.0, atol=1e-5)
        assert np.allclose(result.std(axis=1), 1.0, atol=1e-5)

    def test_resampling_is_skipped_when_the_rates_agree(self, twelve_lead):
        pipeline = get_ecg_transform(length=200, sampling_rate=500, source_sampling_rate=500)
        assert not any("resample" in step for step in repr(pipeline).split(", "))

    def test_amplitude_scaling_preserves_channel_ratios(self, twelve_lead):
        pipeline = get_ecg_transform(length=200, normalize_leads=False, amplitude_scale=4.0)
        result = pipeline(twelve_lead)
        assert np.allclose(result, twelve_lead / 4.0, atol=1e-5)

    def test_missing_samples_are_filled_before_anything_else(self):
        signal = np.zeros((2, 40), dtype=np.float32)
        signal[0] = np.linspace(0, 1, 40)
        signal[1] = np.linspace(0, 1, 40)
        signal[0, 10] = np.nan
        result = get_ecg_transform(length=40)(signal)
        assert np.isfinite(result).all()

    def test_asking_for_both_normalisations_is_rejected(self):
        with pytest.raises(ValueError, match="choose one"):
            get_ecg_transform(length=100, normalize_leads=True, amplitude_scale=2.0)

    def test_asking_for_neither_is_rejected(self):
        with pytest.raises(ValueError, match="no normalisation requested"):
            get_ecg_transform(length=100, normalize_leads=False)


def test_pipeline_plugs_into_an_ecg_source(cohort):
    """The integration the pipeline exists for: a source applies it per subject."""
    from kalecardiac.loaddata import ECGArraySource

    pipeline = get_ecg_transform(length=64, sampling_rate=cohort.sampling_rate)
    source = ECGArraySource(cohort.ecg, leads=cohort.leads, transform=pipeline)
    recording = source.get(cohort.subject_id[0], 0)
    assert recording.shape == (len(cohort.leads), 64)
    assert float(recording.mean(dim=1).abs().max()) < 1e-4


class TestDeriveShapeCheck:
    def test_a_recording_that_disagrees_with_its_leads_is_refused(self):
        with pytest.raises(ECGFormatError, match="does not match the 2 declared leads"):
            derive_limb_leads(np.zeros((3, 10)), ["I", "II"], ["III"])
