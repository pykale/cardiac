"""Image access: shape coercion, size validation, and lazy file reading."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kalecardiac.loaddata import ImageArraySource, ImageFileSource, ImageFormatError, as_image_array


class TestImageArrayCoercion:
    def test_adds_a_channel_axis_to_a_flat_image(self):
        assert as_image_array(np.zeros((16, 20)), channels=1).shape == (1, 16, 20)

    def test_rejects_a_flat_image_when_several_channels_are_declared(self):
        with pytest.raises(ImageFormatError, match="2-D image has one channel"):
            as_image_array(np.zeros((16, 20)), channels=3)

    def test_moves_a_trailing_channel_axis_to_the_front(self):
        assert as_image_array(np.zeros((16, 20, 3)), channels=3).shape == (3, 16, 20)

    def test_leaves_a_leading_channel_axis_alone(self):
        assert as_image_array(np.zeros((3, 16, 20)), channels=3).shape == (3, 16, 20)

    def test_rejects_a_shape_with_no_matching_axis(self):
        with pytest.raises(ImageFormatError, match="no axis of length"):
            as_image_array(np.zeros((4, 16, 20)), channels=3)

    def test_rejects_a_four_dimensional_volume(self):
        # A cine stack is a different contract; half-supporting it here would make this
        # wrong for both.
        with pytest.raises(ImageFormatError, match="2-D or 3-D"):
            as_image_array(np.zeros((8, 1, 16, 20)), channels=1)


class TestImageArraySource:
    def test_infers_its_size_from_the_images(self, cohort):
        source = ImageArraySource(cohort.image, channels=1)
        assert source.size == cohort.image[cohort.subject_id[0]].shape[1:]

    def test_returns_a_float32_tensor(self, image_source, cohort):
        image = image_source.get(cohort.subject_id[0], 0)
        assert isinstance(image, torch.Tensor)
        assert image.dtype == torch.float32

    def test_absent_subject_is_missing(self, image_source):
        assert image_source.get("nobody", 0) is None

    def test_placeholder_matches_the_real_shape(self, image_source, cohort):
        assert image_source.placeholder().shape == image_source.get(cohort.subject_id[0], 0).shape

    def test_transform_is_applied(self, cohort):
        source = ImageArraySource(cohort.image, channels=1, transform=lambda image: image[:, :8, :8])
        assert source.size == (8, 8)
        assert source.get(cohort.subject_id[0], 0).shape == (1, 8, 8)

    def test_a_wrongly_sized_image_is_rejected_with_the_fix(self, cohort):
        images = dict(cohort.image)
        odd = cohort.subject_id[1]
        images[odd] = images[odd][:, :-2, :]
        source = ImageArraySource(images, channels=1, size=cohort.image[cohort.subject_id[0]].shape[1:])
        with pytest.raises(ImageFormatError, match="resize_image"):
            source.get(odd, 1)

    def test_empty_cohort_needs_a_declared_size(self):
        with pytest.raises(ImageFormatError, match="size must be given"):
            ImageArraySource({})


class TestImageFileSource:
    @pytest.fixture
    def written(self, cohort, tmp_path):
        paths = {}
        for identifier in cohort.subject_id[:5]:
            path = tmp_path / f"{identifier}.npy"
            np.save(path, cohort.image[identifier])
            paths[identifier] = path
        return paths

    def test_reads_the_same_values_as_the_array_source(self, cohort, written):
        source = ImageFileSource(written, channels=1)
        identifier = cohort.subject_id[0]
        assert torch.allclose(source.get(identifier, 0), torch.from_numpy(cohort.image[identifier]))

    def test_construction_reads_one_file_not_the_cohort(self, cohort, written):
        reads = []
        ImageFileSource(written, channels=1, loader=lambda path: (reads.append(path), np.load(path))[1])
        assert len(reads) == 1

    def test_provenance_records_where_the_image_came_from(self, cohort, written):
        source = ImageFileSource(written, channels=1)
        identifier = cohort.subject_id[0]
        assert source.provenance(identifier, 0)["image_path"] == str(written[identifier])

    def test_provenance_is_empty_for_a_subject_with_no_file(self, written):
        assert ImageFileSource(written, channels=1).provenance("nobody", 0) == {}
