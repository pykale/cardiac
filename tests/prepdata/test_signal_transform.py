"""Signal transforms: shapes, invariants, the flat-channel edge case, and parity with PyKale."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from kale.prepdata.signal_transform import normalize_signal as kale_normalize_signal
from torchvision.transforms import Compose

from kalecardiac.prepdata import (
    MIN_SCALE,
    check_signal,
    clip_signal,
    crop_or_pad_signal,
    interpolate_signal,
    normalize_signal,
    resample_signal,
    scale_signal,
)


@pytest.fixture
def signal():
    generator = np.random.default_rng(0)
    return generator.normal(size=(4, 200)).astype(np.float32)


def tone(frequency: float, rate: float, seconds: float = 2.0) -> np.ndarray:
    """A one-channel sine wave."""
    time = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * frequency * time)[None, :]


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


class TestCheckSignal:
    def test_adds_a_channel_axis_to_a_flat_signal(self):
        assert check_signal(np.zeros(100)).shape == (1, 100)

    def test_never_transposes(self):
        # Unlike the loader, a transform has not been told how many channels to expect.
        assert check_signal(np.zeros((100, 4))).shape == (100, 4)

    def test_rejects_a_three_dimensional_input(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            check_signal(np.zeros((2, 3, 4)))

    def test_rejects_an_empty_signal(self):
        with pytest.raises(ValueError, match="empty signal"):
            check_signal(np.zeros((2, 0)))


class TestInterpolateSignal:
    def test_fills_an_interior_gap_by_interpolation(self):
        signal = np.array([[0.0, np.nan, 2.0]])
        assert interpolate_signal(signal)[0, 1] == pytest.approx(1.0)

    def test_leaves_a_clean_signal_unchanged(self, signal):
        assert np.allclose(interpolate_signal(signal), signal)

    def test_a_wholly_missing_channel_becomes_zeros(self):
        # Nothing to interpolate between; zeros are the honest answer, and the channel
        # is flat afterwards so normalizing will not amplify it.
        signal = np.array([[np.nan, np.nan], [1.0, 2.0]])
        assert np.array_equal(interpolate_signal(signal)[0], np.zeros(2))

    def test_output_is_always_finite(self):
        signal = np.array([[np.nan, 1.0, np.inf, -np.inf]])
        assert np.isfinite(interpolate_signal(signal)).all()


class TestCropOrPadSignal:
    def test_crops_from_the_start_by_default(self, signal):
        cropped = crop_or_pad_signal(signal, length=50)
        assert cropped.shape == (4, 50)
        assert np.array_equal(cropped, signal[:, :50])

    def test_a_center_crop_takes_the_middle(self, signal):
        cropped = crop_or_pad_signal(signal, length=50, center=True)
        assert np.array_equal(cropped, signal[:, 75:125])

    def test_pads_a_short_signal_to_length(self, signal):
        padded = crop_or_pad_signal(signal, length=300)
        assert padded.shape == (4, 300)
        assert np.array_equal(padded[:, :200], signal)
        assert np.array_equal(padded[:, 200:], np.zeros((4, 100)))

    def test_an_exact_length_is_returned_unchanged(self, signal):
        assert np.array_equal(crop_or_pad_signal(signal, length=200), signal)

    def test_a_non_positive_length_is_rejected(self, signal):
        with pytest.raises(ValueError, match="must be positive"):
            crop_or_pad_signal(signal, length=0)


class TestResampleSignal:
    def test_doubling_the_rate_doubles_the_samples(self, signal):
        assert resample_signal(signal, source_rate=250, target_rate=500).shape == (4, 400)

    def test_halving_the_rate_halves_the_samples(self, signal):
        assert resample_signal(signal, source_rate=500, target_rate=250).shape == (4, 100)

    def test_a_ratio_that_is_not_whole_rounds_the_length_up(self):
        # 360 Hz to 500 Hz is up by 25 and down by 18.
        assert resample_signal(np.zeros((1, 5000)), source_rate=360, target_rate=500).shape == (1, 6945)

    def test_equal_rates_are_a_no_op(self, signal):
        assert np.array_equal(resample_signal(signal, 500, 500), signal)

    def test_a_ramp_survives_decimation_exactly(self):
        ramp = np.linspace(0.0, 1.0, 101)[None, :]
        assert np.allclose(resample_signal(ramp, source_rate=100, target_rate=50)[0], ramp[0, ::2], atol=1e-5)

    def test_content_in_band_is_kept(self):
        # 10 Hz is well inside the 125 Hz band a 250 Hz recording can hold.
        assert rms(resample_signal(tone(10, 500), 500, 250)) == pytest.approx(rms(tone(10, 500)), rel=0.02)

    def test_content_above_the_new_nyquist_is_filtered_not_folded(self):
        # 200 Hz cannot be represented at 250 Hz. Decimating without a filter folds it
        # to 50 Hz at full strength; a polyphase resampler removes it.
        assert rms(resample_signal(tone(200, 500), 500, 250)) < 0.05 * rms(tone(200, 500))

    def test_a_non_positive_rate_is_rejected(self, signal):
        with pytest.raises(ValueError, match="must be positive"):
            resample_signal(signal, source_rate=0, target_rate=500)


class TestNormalizeSignal:
    def test_each_channel_has_zero_mean_and_unit_variance(self, signal):
        normalized = normalize_signal(signal)
        assert np.allclose(normalized.mean(axis=1), 0.0, atol=1e-5)
        assert np.allclose(normalized.std(axis=1), 1.0, atol=1e-5)

    def test_agrees_with_pykale_on_an_ordinary_recording(self, signal):
        # The same function as PyKale's under the same name, for (channels, samples)
        # where PyKale's takes (samples, channels).
        assert np.allclose(normalize_signal(signal), kale_normalize_signal(signal.T).T, atol=1e-5)

    def test_whole_recording_mode_keeps_relative_amplitude(self):
        signal = np.vstack([np.ones(10), 3 * np.ones(10)]) + np.linspace(0, 1, 10)
        normalized = normalize_signal(signal, per_channel=False)
        # One mean and scale for both channels, so their separation survives.
        assert normalized[1].mean() > normalized[0].mean()

    def test_a_flat_channel_is_centered_but_not_amplified(self):
        signal = np.vstack([np.full(10, 2.0), np.linspace(0, 1, 10)])
        normalized = normalize_signal(signal)
        assert np.allclose(normalized[0], 0.0)
        assert np.isfinite(normalized).all()

    def test_a_near_flat_channel_below_the_floor_is_not_amplified(self):
        # The reason this does not call PyKale's: there, a channel whose deviation is
        # rounding noise comes out at full amplitude, because the deviation is non-zero.
        noise = np.linspace(0.0, MIN_SCALE / 10, 10)
        signal = np.vstack([noise, np.linspace(0, 1, 10)])
        assert np.abs(normalize_signal(signal)[0]).max() < 1.0
        assert np.abs(kale_normalize_signal(signal.T).T[0]).max() > 1.0


class TestAmplitude:
    def test_scaling_divides_by_the_constant(self, signal):
        assert np.allclose(scale_signal(signal, scale=2.0), signal / 2.0)

    def test_scaling_preserves_the_ratio_between_channels(self, signal):
        scaled = scale_signal(signal, scale=5.0)
        assert np.allclose(scaled[0] / scaled[1], signal[0] / signal[1])

    def test_a_zero_scale_is_rejected(self, signal):
        with pytest.raises(ValueError, match="non-zero"):
            scale_signal(signal, scale=0.0)

    def test_clipping_bounds_an_excursion(self):
        signal = np.array([[-50.0, 0.0, 50.0]])
        assert np.array_equal(clip_signal(signal, limit=5.0), np.array([[-5.0, 0.0, 5.0]]))

    def test_a_non_positive_limit_is_rejected(self, signal):
        with pytest.raises(ValueError, match="must be positive"):
            clip_signal(signal, limit=0.0)


class TestComposition:
    """Steps are chained with torchvision's Compose, as PyKale chains its transforms."""

    def test_steps_apply_in_order(self, signal):
        result = Compose([partial(crop_or_pad_signal, length=64), normalize_signal])(signal)
        assert result.shape == (4, 64)
        assert np.allclose(result.mean(axis=1), 0.0, atol=1e-5)

    def test_order_matters(self, signal):
        # Normalizing before cropping uses statistics from samples the model never
        # sees, so the cropped result is not normalized.
        cropped_first = Compose([partial(crop_or_pad_signal, length=64), normalize_signal])(signal)
        normalized_first = Compose([normalize_signal, partial(crop_or_pad_signal, length=64)])(signal)
        assert not np.allclose(cropped_first, normalized_first)

    def test_the_steps_are_named_in_a_log(self):
        rendered = repr(Compose([partial(crop_or_pad_signal, length=64), normalize_signal]))
        assert rendered.index("crop_or_pad_signal") < rendered.index("normalize_signal")
