"""Signal transforms: shapes, invariants, and the flat-channel edge case."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from kalecardiac.prepdata import (
    MIN_SCALE,
    SignalPipeline,
    as_signal,
    clip_amplitude,
    crop_or_pad,
    interpolate_missing,
    resample,
    scale_amplitude,
    standardise,
    step_name,
)


@pytest.fixture
def signal():
    generator = np.random.default_rng(0)
    return generator.normal(size=(4, 200)).astype(np.float32)


class TestAsSignal:
    def test_adds_a_channel_axis_to_a_flat_signal(self):
        assert as_signal(np.zeros(100)).shape == (1, 100)

    def test_never_transposes(self):
        # Unlike the loader, a transform has not been told how many channels to expect.
        assert as_signal(np.zeros((100, 4))).shape == (100, 4)

    def test_rejects_a_three_dimensional_input(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            as_signal(np.zeros((2, 3, 4)))

    def test_rejects_an_empty_signal(self):
        with pytest.raises(ValueError, match="empty signal"):
            as_signal(np.zeros((2, 0)))


class TestInterpolateMissing:
    def test_fills_an_interior_gap_by_interpolation(self):
        signal = np.array([[0.0, np.nan, 2.0]])
        assert interpolate_missing(signal)[0, 1] == pytest.approx(1.0)

    def test_leaves_a_clean_signal_unchanged(self, signal):
        assert np.allclose(interpolate_missing(signal), signal)

    def test_a_wholly_missing_channel_becomes_zeros(self):
        # Nothing to interpolate between; zeros are the honest answer, and the channel
        # is flat afterwards so standardise will not amplify it.
        signal = np.array([[np.nan, np.nan], [1.0, 2.0]])
        assert np.array_equal(interpolate_missing(signal)[0], np.zeros(2))

    def test_output_is_always_finite(self):
        signal = np.array([[np.nan, 1.0, np.inf, -np.inf]])
        assert np.isfinite(interpolate_missing(signal)).all()


class TestCropOrPad:
    def test_crops_from_the_start_by_default(self, signal):
        cropped = crop_or_pad(signal, length=50)
        assert cropped.shape == (4, 50)
        assert np.array_equal(cropped, signal[:, :50])

    def test_centre_crop_takes_the_middle(self, signal):
        cropped = crop_or_pad(signal, length=50, centre=True)
        assert np.array_equal(cropped, signal[:, 75:125])

    def test_pads_a_short_signal_to_length(self, signal):
        padded = crop_or_pad(signal, length=300)
        assert padded.shape == (4, 300)
        assert np.array_equal(padded[:, :200], signal)
        assert np.array_equal(padded[:, 200:], np.zeros((4, 100)))

    def test_an_exact_length_is_returned_unchanged(self, signal):
        assert np.array_equal(crop_or_pad(signal, length=200), signal)

    def test_a_non_positive_length_is_rejected(self, signal):
        with pytest.raises(ValueError, match="must be positive"):
            crop_or_pad(signal, length=0)


class TestResample:
    def test_doubling_the_rate_doubles_the_samples(self, signal):
        assert resample(signal, source_rate=250, target_rate=500).shape == (4, 400)

    def test_halving_the_rate_halves_the_samples(self, signal):
        assert resample(signal, source_rate=500, target_rate=250).shape == (4, 100)

    def test_equal_rates_are_a_no_op(self, signal):
        assert np.array_equal(resample(signal, 500, 500), signal)

    def test_endpoints_are_preserved(self, signal):
        resampled = resample(signal, source_rate=500, target_rate=250)
        assert resampled[:, 0] == pytest.approx(signal[:, 0], abs=1e-5)
        assert resampled[:, -1] == pytest.approx(signal[:, -1], abs=1e-5)

    def test_a_linear_ramp_stays_linear(self):
        ramp = np.linspace(0.0, 1.0, 101)[None, :]
        resampled = resample(ramp, source_rate=100, target_rate=50)
        assert np.allclose(resampled[0], np.linspace(0.0, 1.0, resampled.shape[1]), atol=1e-6)

    def test_a_non_positive_rate_is_rejected(self, signal):
        with pytest.raises(ValueError, match="must be positive"):
            resample(signal, source_rate=0, target_rate=500)


class TestStandardise:
    def test_each_channel_has_zero_mean_and_unit_variance(self, signal):
        standardised = standardise(signal)
        assert np.allclose(standardised.mean(axis=1), 0.0, atol=1e-5)
        assert np.allclose(standardised.std(axis=1), 1.0, atol=1e-5)

    def test_whole_recording_mode_keeps_relative_amplitude(self):
        signal = np.vstack([np.ones(10), 3 * np.ones(10)]) + np.linspace(0, 1, 10)
        standardised = standardise(signal, per_channel=False)
        # One mean and scale for both channels, so their separation survives.
        assert standardised[1].mean() > standardised[0].mean()

    def test_a_flat_channel_is_centred_but_not_amplified(self):
        signal = np.vstack([np.full(10, 2.0), np.linspace(0, 1, 10)])
        standardised = standardise(signal)
        assert np.allclose(standardised[0], 0.0)
        assert np.isfinite(standardised).all()

    def test_a_near_flat_channel_below_the_floor_is_not_amplified(self):
        signal = np.vstack([np.full(10, 1.0) + MIN_SCALE / 10, np.linspace(0, 1, 10)])
        assert np.abs(standardise(signal)[0]).max() < 1.0


class TestAmplitude:
    def test_scaling_divides_by_the_constant(self, signal):
        assert np.allclose(scale_amplitude(signal, scale=2.0), signal / 2.0)

    def test_scaling_preserves_the_ratio_between_channels(self, signal):
        scaled = scale_amplitude(signal, scale=5.0)
        assert np.allclose(scaled[0] / scaled[1], signal[0] / signal[1])

    def test_a_zero_scale_is_rejected(self, signal):
        with pytest.raises(ValueError, match="non-zero"):
            scale_amplitude(signal, scale=0.0)

    def test_clipping_bounds_an_excursion(self):
        signal = np.array([[-50.0, 0.0, 50.0]])
        assert np.array_equal(clip_amplitude(signal, limit=5.0), np.array([[-5.0, 0.0, 5.0]]))

    def test_a_non_positive_limit_is_rejected(self, signal):
        with pytest.raises(ValueError, match="must be positive"):
            clip_amplitude(signal, limit=0.0)


class TestSignalPipeline:
    def test_applies_steps_in_order(self, signal):
        pipeline = SignalPipeline([partial(crop_or_pad, length=64), standardise])
        result = pipeline(signal)
        assert result.shape == (4, 64)
        assert np.allclose(result.mean(axis=1), 0.0, atol=1e-5)

    def test_order_matters_and_is_visible(self, signal):
        # Standardising before cropping uses statistics from samples the model never
        # sees, so the cropped result is not standardised.
        cropped_then = SignalPipeline([partial(crop_or_pad, length=64), standardise])(signal)
        standardised_then = SignalPipeline([standardise, partial(crop_or_pad, length=64)])(signal)
        assert not np.allclose(cropped_then, standardised_then)

    def test_an_empty_pipeline_is_the_identity(self, signal):
        assert np.array_equal(SignalPipeline([])(signal), signal)

    def test_a_non_callable_step_is_rejected(self):
        with pytest.raises(TypeError, match="must be callable"):
            SignalPipeline(["standardise"])

    def test_repr_names_the_steps_through_a_partial(self, signal):
        rendered = repr(SignalPipeline([partial(crop_or_pad, length=64), standardise]))
        assert "crop_or_pad(length=64)" in rendered
        assert "standardise" in rendered

    def test_step_name_sees_through_nested_partials(self):
        assert "crop_or_pad" in step_name(partial(partial(crop_or_pad, length=8), centre=True))
