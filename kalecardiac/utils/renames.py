"""Names the API used before it took PyKale's conventions, and what replaced each.

Importing an old name from the subpackage that exported it fails with its replacement in
the message -- or, for a name that was removed in favour of a dependency's own function,
with that function -- rather than with Python's bare ``cannot import name``::

    >>> from kalecardiac.prepdata import standardise
    Traceback (most recent call last):
    ...
    ImportError: kalecardiac.prepdata.standardise is now normalize_signal: the API follows PyKale's names (see CHANGELOG.md)

Only the subpackages answer, because that is where the documentation imports from; an old
name imported from the module that defined it fails with Python's own message, and
CHANGELOG.md lists every rename. A renamed configuration key is registered with YACS in
:mod:`kalecardiac.config` instead, so an old YAML file fails the same way.
"""

from __future__ import annotations

#: Old name to new name, per subpackage that exported it. A value starting ``"removed:"``
#: says what to use instead.
RENAMED: dict[str, dict[str, str]] = {
    "kalecardiac.loaddata": {
        "ecg_access": "signal_access",
        "canonical_lead": "normalize_lead_name",
        "canonical_leads": "normalize_lead_names",
        "lead_indices": "get_lead_indices",
        "as_lead_array": "check_ecg_array",
        "lead_sources": "get_lead_sources",
        "as_image_array": "check_image_array",
        "composite_labels": "get_stratification_labels",
        "CohortSplitter": "BaseSplitter",
        "HoldOut": "HoldOutSplit",
        "CrossValidation": "CrossValidationSplit",
        "Predefined": "PredefinedSplit",
        "train_test_split": "removed: use next(HoldOutSplit(val_size=0).split(frame))",
        "subject_frame": "removed: build the table with pandas.DataFrame",
    },
    "kalecardiac.prepdata": {
        "as_signal": "check_signal",
        "as_image": "check_image",
        "interpolate_missing": "interpolate_signal",
        "crop_or_pad": "crop_or_pad_signal",
        "resample": "resample_signal",
        "standardise": "normalize_signal",
        "scale_amplitude": "scale_signal",
        "clip_amplitude": "clip_signal",
        "scale_image": "scale_image_intensity",
        "standardise_image": "normalize_image",
        "build_ecg_pipeline": "get_ecg_transform",
        "build_image_pipeline": "get_image_transform",
        "lead_selector": "get_lead_selector",
        "SignalPipeline": "removed: use torchvision.transforms.Compose",
        "ImagePipeline": "removed: use torchvision.transforms.Compose",
        "step_name": "removed: the repr of a torchvision.transforms.Compose names each step",
    },
    "kalecardiac.model.layers": {"reparameterise": "reparameterize", "expert_groups": "get_expert_groups"},
    "kalecardiac.model.embed": {"freeze": "removed: use the module's own requires_grad_(False)"},
    "kalecardiac.model.predict": {
        "binary_cross_entropy": "removed: use torch.nn.functional.binary_cross_entropy_with_logits",
        "cross_entropy": "removed: use torch.nn.functional.cross_entropy",
        "mean_squared_error": "removed: use torch.nn.functional.mse_loss",
        "class_weight_from_labels": "removed: use sklearn.utils.class_weight.compute_class_weight",
        "positive_weight_from_labels": "removed: use sklearn.utils.class_weight.compute_class_weight",
    },
    "kalecardiac.evaluate": {
        "summarise_folds": "summarize_folds",
        "r2_score": "removed: use sklearn.metrics.r2_score",
    },
    "kalecardiac.interpret": {
        "normalise_attribution": "normalize_attribution",
        "ecg_wave_segments": "delineate_ecg_waves",
        "latent_embedding": "project_latents",
    },
}


def renamed(module: str, name: str) -> ImportError | AttributeError:
    """The error for a name ``module`` does not have, saying what replaced it if anything did.

    For a package's module-level ``__getattr__``. An old name gives an
    :class:`ImportError`, which ``from module import name`` passes on with its message
    intact; any other name gives the ordinary :class:`AttributeError`, so ``hasattr``
    and every tool that probes a module behave as they would without this.

    Args:
        module: The package's ``__name__``.
        name: The attribute asked for.

    Returns:
        The exception to raise.
    """
    replacement = RENAMED.get(module, {}).get(name)
    if replacement is None:
        return AttributeError(f"module {module!r} has no attribute {name!r}")
    if replacement.startswith("removed:"):
        return ImportError(f"{module}.{name} was {replacement} (see CHANGELOG.md)")
    return ImportError(f"{module}.{name} is now {replacement}: the API follows PyKale's names (see CHANGELOG.md)")
