"""Image transforms: resizing, intensity normalisation, and the assembled pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from kalecardiac.prepdata import (
    ImagePipeline,
    as_image,
    build_image_pipeline,
    resize_image,
    scale_image,
    standardise_image,
)


@pytest.fixture
def image():
    generator = np.random.default_rng(0)
    return generator.uniform(0, 255, size=(1, 40, 50)).astype(np.float32)


class TestAsImage:
    def test_adds_a_channel_axis(self):
        assert as_image(np.zeros((8, 9))).shape == (1, 8, 9)

    def test_rejects_a_four_dimensional_input(self):
        with pytest.raises(ValueError, match="2-D or 3-D"):
            as_image(np.zeros((2, 1, 8, 9)))

    def test_rejects_an_empty_image(self):
        with pytest.raises(ValueError, match="empty image"):
            as_image(np.zeros((1, 0, 8)))


class TestResizeImage:
    def test_produces_the_requested_size(self, image):
        assert resize_image(image, (16, 16)).shape == (1, 16, 16)

    def test_an_exact_size_is_returned_unchanged(self, image):
        assert np.array_equal(resize_image(image, (40, 50)), image)

    def test_a_constant_image_stays_constant(self):
        constant = np.full((1, 32, 32), 7.0, dtype=np.float32)
        assert np.allclose(resize_image(constant, (8, 8)), 7.0)

    def test_upsampling_works_as_well_as_downsampling(self, image):
        assert resize_image(image, (80, 100)).shape == (1, 80, 100)

    def test_every_channel_is_resized(self):
        colour = np.random.default_rng(1).uniform(size=(3, 20, 20)).astype(np.float32)
        assert resize_image(colour, (10, 10)).shape == (3, 10, 10)

    def test_an_invalid_size_is_rejected(self, image):
        with pytest.raises(ValueError, match="two positive integers"):
            resize_image(image, (0, 16))


class TestScaleImage:
    def test_a_declared_maximum_maps_to_one(self, image):
        scaled = scale_image(image, maximum=255.0)
        assert scaled.min() >= 0.0 and scaled.max() <= 1.0
        assert np.allclose(scaled, np.clip(image / 255.0, 0, 1))

    def test_without_a_maximum_the_image_fills_the_range(self, image):
        scaled = scale_image(image)
        assert scaled.min() == pytest.approx(0.0)
        assert scaled.max() == pytest.approx(1.0)

    def test_a_constant_image_becomes_zeros_rather_than_nan(self):
        assert np.array_equal(scale_image(np.full((1, 4, 4), 3.0)), np.zeros((1, 4, 4)))

    def test_a_non_positive_maximum_is_rejected(self, image):
        with pytest.raises(ValueError, match="must be positive"):
            scale_image(image, maximum=0.0)


class TestStandardiseImage:
    def test_each_channel_has_zero_mean_and_unit_variance(self):
        colour = np.random.default_rng(2).normal(size=(3, 16, 16)).astype(np.float32)
        standardised = standardise_image(colour)
        assert np.allclose(standardised.mean(axis=(1, 2)), 0.0, atol=1e-5)
        assert np.allclose(standardised.std(axis=(1, 2)), 1.0, atol=1e-5)

    def test_supplied_statistics_are_applied(self, image):
        standardised = standardise_image(image, mean=[100.0], std=[50.0])
        assert np.allclose(standardised, (image - 100.0) / 50.0, atol=1e-4)

    def test_a_statistic_of_the_wrong_length_is_rejected(self, image):
        with pytest.raises(ValueError, match="one value per channel"):
            standardise_image(image, mean=[0.0, 1.0])

    def test_a_constant_channel_is_not_amplified(self):
        flat = np.full((1, 8, 8), 5.0, dtype=np.float32)
        assert np.allclose(standardise_image(flat), 0.0)


class TestImagePipeline:
    def test_resizes_then_scales(self, image):
        result = build_image_pipeline(size=(16, 16))(image)
        assert result.shape == (1, 16, 16)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_standardise_mode_leaves_the_unit_interval(self, image):
        result = build_image_pipeline(size=(16, 16), scale=False, standardise=True)(image)
        assert result.min() < 0.0

    def test_asking_for_both_normalisations_is_rejected(self):
        with pytest.raises(ValueError, match="choose one"):
            build_image_pipeline(size=(8, 8), scale=True, standardise=True)

    def test_a_non_callable_step_is_rejected(self):
        with pytest.raises(TypeError, match="must be callable"):
            ImagePipeline(["resize"])

    def test_repr_names_the_steps(self):
        assert "resize_image((8, 8))" in repr(build_image_pipeline(size=(8, 8)))


def test_pipeline_plugs_into_an_image_source(cohort):
    """The integration the pipeline exists for."""
    from kalecardiac.loaddata import ImageArraySource

    pipeline = build_image_pipeline(size=(16, 16))
    source = ImageArraySource(cohort.image, channels=1, transform=pipeline)
    assert source.get(cohort.subject_id[0], 0).shape == (1, 16, 16)
