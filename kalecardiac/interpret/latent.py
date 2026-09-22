"""Reading the latent space a representation-learning model built.

Pretraining produces a latent space and no predictions, so the only way to see whether
it learned anything before a labelled cohort arrives is to look at the space itself:
collect the posterior means over a cohort and project them.

Coordinates are returned rather than plots, matching
:mod:`kalecardiac.interpret.attribution`, which returns attributions rather than
heatmaps. A publication figure is an experiment's concern.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from kalecardiac.loaddata.multimodal_access import SubjectBatch


def collect_latents(
    model: nn.Module,
    loader: Iterable[SubjectBatch],
    per_modality: bool = False,
    device: torch.device | str | None = None,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Encode a cohort and return its latent representations.

    The posterior **mean**, not a sample: a description of the cohort should not change
    between two passes over the same subjects.

    Args:
        model: A :class:`~kalecardiac.model.embed.MultimodalVAE`, or anything with a
            compatible ``encode``.
        loader: Yields :class:`~kalecardiac.loaddata.SubjectBatch` batches.
        per_modality: Also return each modality's own posterior mean, before fusion.
            That is what shows whether the modality encoders agree -- whether a latent
            alignment term did what it was added for.
        device: Where to run. ``None`` uses whatever the model is already on.

    Returns:
        ``(subject_ids, joint, per_modality_means)`` -- identifiers in loader order, an
        ``(N, latent_dim)`` array of joint means, and one such array per modality when
        asked for, else an empty dict.

    Raises:
        ValueError: If the loader yields nothing.
        AttributeError: If the model has no ``encode``.
    """
    if not hasattr(model, "encode"):
        raise AttributeError(
            f"{type(model).__name__} has no encode(); collect_latents reads a representation-learning model such "
            f"as MultimodalVAE"
        )

    was_training = model.training
    model.eval()
    if device is not None:
        model.to(device)

    identifiers: list[str] = []
    joint: list[np.ndarray] = []
    modality: dict[str, list[np.ndarray]] = {}

    try:
        with torch.no_grad():
            for batch in loader:
                if device is not None:
                    batch = batch.to(device)
                posterior, posteriors = model.encode(batch.modalities, batch.present)
                identifiers.extend(batch.subject_id)
                joint.append(posterior.mean.detach().cpu().numpy())
                if per_modality:
                    for name, value in posteriors.items():
                        modality.setdefault(name, []).append(value.mean.detach().cpu().numpy())
    finally:
        model.train(was_training)

    if not identifiers:
        raise ValueError("the loader yielded no batches")

    return (
        identifiers,
        np.concatenate(joint),
        {name: np.concatenate(values) for name, values in modality.items()},
    )


def latent_embedding(
    features: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    random_state: int | None = 2026,
    fit_on: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Project latent vectors into a low-dimensional space.

    Two methods, and the choice is not cosmetic. PCA is linear, deterministic and has
    no free parameters, so a structure visible in it is a structure in the data; it is
    the default for that reason and needs no extra dependency. UMAP finds non-linear
    structure and is what the cardiac studies plot, but its layout depends on its
    neighbourhood parameters, and clusters in a UMAP plot are not evidence of clusters
    in the data.

    Args:
        features: ``(N, D)`` vectors to project.
        method: ``"pca"`` or ``"umap"``.
        n_components: Dimensions to project into.
        random_state: Seed, for a reproducible layout.
        fit_on: Optional ``(M, D)`` subset to fit the projection on before transforming
            ``features``. Pass the training rows when the projection is meant to
            illustrate a model's view of held-out subjects; leave it unset when the
            projection is a description of the whole cohort.
        **kwargs: Method-specific options, e.g. ``n_neighbors`` and ``min_dist`` for
            UMAP.

    Returns:
        ``(N, n_components)`` coordinates.

    Raises:
        ImportError: If ``"umap"`` is asked for and umap-learn is not installed.
        ValueError: If ``features`` is not two-dimensional, or ``method`` is unknown.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {features.shape}")

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=min(n_components, *features.shape), random_state=random_state, **kwargs)
    elif method == "umap":
        try:
            import umap
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                'latent_embedding(method="umap") needs umap-learn. Install it with: '
                'pip install "kalecardiac[interpret]"'
            ) from error
        # Bounded by the cohort: UMAP's default of 15 neighbours fails outright on a
        # split smaller than that, which a small cardiac cohort routinely is.
        kwargs.setdefault("n_neighbors", min(15, max(2, len(features) - 1)))
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"unknown method {method!r}; available: 'pca', 'umap'")

    if fit_on is None:
        return np.asarray(reducer.fit_transform(features))
    return np.asarray(reducer.fit(np.asarray(fit_on, dtype=np.float64)).transform(features))
