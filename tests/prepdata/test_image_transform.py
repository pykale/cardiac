"""Image transforms: resizing, intensity normalization, and the assembled transform."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision.transforms import Compose
from torchvision.transforms.functional import normalize as torchvision_normalize

from kalecardiac.prepdata import (
    check_image,
    get_image_transform,
    normalize_image,
    resize_image,
    scale_image_intensity,
)


@pytest.fixture
def image():
    generator = np.random.default_rng(0)
    return generator.uniform(0, 255, size=(1, 40, 50)).astype(np.float32)


class TestCheckImage:
    def test_adds_a_channel_axis(self):
        assert check_image(np.zeros((8, 9))).shape == (1, 8, 9)

    def test_rejects_a_four_dimensional_input(self):
        with pytest.raises(ValueError, match="2-D or 3-D"):
            check_image(np.zeros((2, 1, 8, 9)))

    def test_rejects_an_empty_image(self):
        with pytest.raises(ValueError, match="empty image"):
            check_image(np.zeros((1, 0, 8)))


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
        scaled = scale_image_intensity(image, maximum=255.0)
        assert scaled.min() >= 0.0 and scaled.max() <= 1.0
        assert np.allclose(scaled, np.clip(image / 255.0, 0, 1))

    def test_without_a_maximum_the_image_fills_the_range(self, image):
        scaled = scale_image_intensity(image)
        assert scaled.min() == pytest.approx(0.0)
        assert scaled.max() == pytest.approx(1.0)

    def test_a_constant_image_becomes_zeros_rather_than_nan(self):
        assert np.array_equal(scale_image_intensity(np.full((1, 4, 4), 3.0)), np.zeros((1, 4, 4)))

    def test_a_non_positive_maximum_is_rejected(self, image):
        with pytest.raises(ValueError, match="must be positive"):
            scale_image_intensity(image, maximum=0.0)


class TestNormalizeImage:
    def test_each_channel_has_zero_mean_and_unit_variance(self):
        colour = np.random.default_rng(2).normal(size=(3, 16, 16)).astype(np.float32)
        normalized = normalize_image(colour)
        assert np.allclose(normalized.mean(axis=(1, 2)), 0.0, atol=1e-5)
        assert np.allclose(normalized.std(axis=(1, 2)), 1.0, atol=1e-5)

    def test_supplied_statistics_are_applied(self, image):
        normalized = normalize_image(image, mean=[100.0], std=[50.0])
        assert np.allclose(normalized, (image - 100.0) / 50.0, atol=1e-4)

    def test_supplied_statistics_agree_with_torchvision(self, image):
        expected = torchvision_normalize(torch.from_numpy(image), mean=[100.0], std=[50.0]).numpy()
        assert np.allclose(normalize_image(image, mean=[100.0], std=[50.0]), expected, atol=1e-5)

    def test_a_statistic_of_the_wrong_length_is_rejected(self, image):
        with pytest.raises(ValueError, match="one value per channel"):
            normalize_image(image, mean=[0.0, 1.0])

    def test_a_constant_channel_is_not_amplified(self):
        flat = np.full((1, 8, 8), 5.0, dtype=np.float32)
        assert np.allclose(normalize_image(flat), 0.0)


class TestGetImageTransform:
    def test_resizes_then_scales(self, image):
        result = get_image_transform(size=(16, 16))(image)
        assert result.shape == (1, 16, 16)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_it_is_a_torchvision_compose(self):
        assert isinstance(get_image_transform(size=(8, 8)), Compose)

    def test_normalize_mode_leaves_the_unit_interval(self, image):
        result = get_image_transform(size=(16, 16), scale=False, normalize=True)(image)
        assert result.min() < 0.0

    def test_asking_for_both_is_rejected(self):
        with pytest.raises(ValueError, match="choose one"):
            get_image_transform(size=(8, 8), scale=True, normalize=True)

    def test_the_steps_are_named_in_a_log(self):
        rendered = repr(get_image_transform(size=(8, 8)))
        assert rendered.index("resize_image") < rendered.index("scale_image_intensity")
        assert "size=(8, 8)" in rendered


def test_the_transform_plugs_into_an_image_source(cohort):
    """The integration the transform exists for."""
    from kalecardiac.loaddata import ImageArraySource

    source = ImageArraySource(cohort.image, channels=1, transform=get_image_transform(size=(16, 16)))
    assert source.get(cohort.subject_id[0], 0).shape == (1, 16, 16)


class TestStatistics:
    def test_a_deviation_per_channel_is_required(self):
        with pytest.raises(ValueError, match="std must have one value per channel \\(2\\), got 1"):
            normalize_image(np.ones((2, 4, 4)), mean=[0.0, 0.0], std=[1.0])
