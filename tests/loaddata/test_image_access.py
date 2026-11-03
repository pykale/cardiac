"""Image access: shape coercion, size validation, lazy file reading, and DICOM images."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

<<<<<<< HEAD
from kalecardiac.loaddata import (
    STANDARD_12_LEAD,
    ImageArraySource,
    ImageFileSource,
    ImageFormatError,
    check_image_array,
    get_dicom_paths,
    read_dicom,
    read_dicom_image,
)
from kalecardiac.prepdata import get_image_transform
=======
from kalecardiac.loaddata import ImageArraySource, ImageFileSource, ImageFormatError, check_image_array
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a


class TestImageArrayCoercion:
    def test_adds_a_channel_axis_to_a_flat_image(self):
        assert check_image_array(np.zeros((16, 20)), channels=1).shape == (1, 16, 20)

    def test_rejects_a_flat_image_when_several_channels_are_declared(self):
        with pytest.raises(ImageFormatError, match="2-D image has one channel"):
            check_image_array(np.zeros((16, 20)), channels=3)

    def test_moves_a_trailing_channel_axis_to_the_front(self):
        assert check_image_array(np.zeros((16, 20, 3)), channels=3).shape == (3, 16, 20)

    def test_leaves_a_leading_channel_axis_alone(self):
        assert check_image_array(np.zeros((3, 16, 20)), channels=3).shape == (3, 16, 20)

    def test_rejects_a_shape_with_no_matching_axis(self):
        with pytest.raises(ImageFormatError, match="no axis of length"):
            check_image_array(np.zeros((4, 16, 20)), channels=3)

    def test_rejects_a_four_dimensional_volume(self):
        # A cine stack is a different contract; half-supporting it here would make this
        # wrong for both.
        with pytest.raises(ImageFormatError, match="2-D or 3-D"):
            check_image_array(np.zeros((8, 1, 16, 20)), channels=1)


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


class TestImageEdgesAndDescriptions:
    def test_a_transform_that_breaks_the_layout_is_named(self, cohort):
        source = ImageArraySource(cohort.image, channels=1, size=(32, 32))
        source.transform = lambda image: image[0]  # drops the channel axis
        with pytest.raises(ImageFormatError, match="it must keep \\(1, height, width\\)"):
            source.get(cohort.subject_id[0], 0)

    def test_a_picture_file_is_read_through_pillow(self, tmp_path):
        image = pytest.importorskip("PIL.Image")
        pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
        image.fromarray(pixels).save(tmp_path / "s1.png")
        source = ImageFileSource({"s1": tmp_path / "s1.png"})
        assert source.size == (8, 8)
        assert np.array_equal(source.get("s1", 0).numpy()[0], pixels.astype(np.float32))

    def test_sources_describe_their_subjects_and_shape(self, tmp_path):
        assert repr(ImageArraySource({"s1": np.zeros((8, 8))})) == "ImageArraySource(1 subjects | 1x8x8)"
        np.save(tmp_path / "s1.npy", np.zeros((8, 8), dtype=np.float32))
        assert repr(ImageFileSource({"s1": tmp_path / "s1.npy"})) == "ImageFileSource(1 subjects | 1x8x8)"


# --------------------------------------------------------------------------- #
# DICOM
# --------------------------------------------------------------------------- #

#: Every stored value of a 12-bit detector, once each, in a 64 x 64 image.
GRADIENT = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)


class TestGetDicomPaths:
    def test_each_folder_is_a_subject_named_by_the_folder(self, tmp_path):
        for subject in ("S2", "S10"):
            (tmp_path / subject / "1" / "DICOM").mkdir(parents=True)
        assert get_dicom_paths(tmp_path) == {"S10": tmp_path / "S10", "S2": tmp_path / "S2"}

    def test_a_dicom_file_at_the_top_is_a_subject_named_by_its_stem(self, tmp_path):
        (tmp_path / "s1.dcm").write_bytes(b"")
        (tmp_path / "notes.txt").write_text("not a subject")
        (tmp_path / ".cache").mkdir()
        assert get_dicom_paths(tmp_path) == {"s1": tmp_path / "s1.dcm"}

    def test_a_missing_folder_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="one folder"):
            get_dicom_paths(tmp_path / "absent")

    def test_a_folder_and_a_file_for_one_subject_are_not_guessed_between(self, tmp_path):
        (tmp_path / "s1").mkdir()
        (tmp_path / "s1.dcm").write_bytes(b"")
        with pytest.raises(ValueError, match="both s1 and s1.dcm"):
            get_dicom_paths(tmp_path)


class TestReadDicom:
    def test_reads_the_one_file_anywhere_under_a_subject_folder(self, dicom_writers, tmp_path):
        dicom_writers.write_dicom_image(tmp_path / "s1" / "1" / "DICOM" / "00000000.dcm", GRADIENT, patient_id="s1")
        assert read_dicom(tmp_path / "s1").PatientID == "s1"

    def test_a_folder_is_read_without_pykales_bookkeeping_warning(self, dicom_writers, tmp_path):
        dicom_writers.write_dicom_image(tmp_path / "s1" / "x.dcm", GRADIENT)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            read_dicom(tmp_path / "s1")

    def test_reads_a_file_by_its_path(self, dicom_writers, tmp_path):
        path = dicom_writers.write_dicom_image(tmp_path / "x.dcm", GRADIENT, patient_id="x")
        assert read_dicom(path).PatientID == "x"

    def test_a_folder_without_dicom_is_named(self, dicom_writers, tmp_path):
        (tmp_path / "s1").mkdir()
        with pytest.raises(FileNotFoundError, match="no .dcm file under"):
            read_dicom(tmp_path / "s1")

    def test_a_folder_with_two_files_is_not_guessed_between(self, dicom_writers, tmp_path):
        # A frontal and a lateral view, say: which is the subject's is not file order's to decide.
        for view in ("frontal", "lateral"):
            dicom_writers.write_dicom_image(tmp_path / "s1" / f"{view}.dcm", GRADIENT)
        with pytest.raises(ValueError, match=r"holds 2 DICOM files \['frontal.dcm', 'lateral.dcm'\]"):
            read_dicom(tmp_path / "s1")

    def test_a_path_that_does_not_exist_is_named(self, dicom_writers, tmp_path):
        with pytest.raises(FileNotFoundError, match="no DICOM file or folder"):
            read_dicom(tmp_path / "absent")


class TestReadDicomImage:
    def test_an_image_with_no_display_settings_is_read_as_stored(self, dicom_writers, tmp_path):
        image = read_dicom_image(dicom_writers.write_dicom_image(tmp_path / "x.dcm", GRADIENT))
        assert image.dtype == np.float32
        assert np.array_equal(image, GRADIENT)

    def test_the_rescale_maps_stored_values_to_the_modalitys_units(self, dicom_writers, tmp_path):
        path = dicom_writers.write_dicom_image(tmp_path / "x.dcm", GRADIENT, rescale=(2.0, -100.0))
        assert np.allclose(read_dicom_image(path), 2.0 * GRADIENT - 100.0)

    def test_the_window_selects_the_range_the_image_is_read_in(self, dicom_writers, tmp_path):
        # A window over the middle half: below it is black, above it white, and between
        # them the values keep their order.
        path = dicom_writers.write_dicom_image(tmp_path / "x.dcm", GRADIENT, window=(2048.0, 2048.0))
        image = read_dicom_image(path).ravel()
        assert image[:1024].max() == 0
        assert image[-1024:].min() == 4095
        assert np.all(np.diff(image) >= 0)

    def test_monochrome1_is_inverted_so_that_brighter_is_higher(self, dicom_writers, tmp_path):
        path = dicom_writers.write_dicom_image(tmp_path / "x.dcm", GRADIENT, photometric="MONOCHROME1")
        assert np.array_equal(read_dicom_image(path), GRADIENT.max() - GRADIENT)

    def test_a_colour_image_is_returned_as_stored(self, dicom_writers, tmp_path):
        import pydicom

        path = dicom_writers.write_dicom_image(tmp_path / "x.dcm", np.zeros((4, 4)))
        colour = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        dataset = pydicom.dcmread(path)
        dataset.SamplesPerPixel, dataset.PlanarConfiguration = 3, 0
        dataset.PhotometricInterpretation = "RGB"
        dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit = 8, 8, 7
        dataset.PixelData = colour.tobytes()
        dataset.save_as(path)
        assert np.array_equal(read_dicom_image(path), colour)

    def test_a_dicom_without_pixels_is_not_an_image(self, dicom_writers, tmp_path):
        path = dicom_writers.write_dicom_ecg(tmp_path / "ecg.dcm", np.zeros((12, 50)), STANDARD_12_LEAD)
        with pytest.raises(ImageFormatError, match="holds no image: it is 12-lead ECG Waveform Storage"):
            read_dicom_image(path)

    def test_a_multi_frame_image_is_a_volume_rather_than_an_image(self, dicom_writers, tmp_path):
        import pydicom

        path = dicom_writers.write_dicom_image(tmp_path / "x.dcm", np.zeros((4, 4)))
        dataset = pydicom.dcmread(path)
        dataset.NumberOfFrames = 2
        dataset.PixelData = dataset.PixelData * 2
        dataset.save_as(path)
        with pytest.raises(ImageFormatError, match="2 frames, which make a volume"):
            read_dicom_image(path)


class TestImageFileSourceFromDicom:
    @pytest.fixture
    def written(self, dicom_writers, tmp_path):
        return dicom_writers.write_synthetic_dicom_cohort(
            tmp_path, num_subjects=3, num_samples=50, sampling_rate=50, image_size=(16, 16)
        )

    def test_each_subject_folder_is_a_subject(self, written, tmp_path):
        source = ImageFileSource.from_dicom(tmp_path / "CXR")
        assert sorted(source.paths) == sorted(written.subject_id)
        assert source.size == (16, 16)

    def test_the_image_read_is_the_one_written(self, written, tmp_path):
        subject = written.subject_id[0]
        image = ImageFileSource.from_dicom(tmp_path / "CXR").get(subject, 0).numpy()[0]
        # Stored as 12-bit integers over a window covering every value they can take.
        assert np.allclose(image / 4095, written.image[subject][0], atol=1e-3)

    def test_the_transform_prepares_each_image(self, written, tmp_path):
        source = ImageFileSource.from_dicom(tmp_path / "CXR", transform=get_image_transform(size=(8, 8)))
        image = source.get(written.subject_id[1], 1)
        assert source.size == (8, 8)
        assert tuple(image.shape) == (1, 8, 8)
        assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0

    def test_a_folder_holding_no_subject_is_named(self, tmp_path):
        with pytest.raises(ImageFormatError, match="no subject folder or .dcm file"):
            ImageFileSource.from_dicom(tmp_path)

    def test_a_dicom_file_asks_for_the_reader_that_applies_its_display_settings(self, tmp_path):
        path = tmp_path / "x.dcm"
        path.write_bytes(b"")
        with pytest.raises(ImageFormatError, match="loader=read_dicom_image"):
            ImageFileSource({"x": path})
