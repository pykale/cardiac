"""Training objectives, generative and discriminative.

Two families live here because a cardiac workflow uses both, in that order: a
generative objective pretrains a representation on unlabelled recordings, and a
discriminative one fine-tunes it onto a clinical endpoint.

**The generative side.** An ELBO is a reconstruction term plus a KL term, and the whole
difficulty in a multimodal setting is that those two are not commensurable. A
twelve-lead recording at 500 Hz has 60,000 samples; a sum-of-squares over them is
larger than the KL over a 256-dimensional latent by four orders of magnitude, so an
unscaled ELBO is a reconstruction loss with a rounding error attached, and the latent
never regularises. Three controls address it, and all three appear in the cardiac
reference implementations:

``weights``
    One per modality, so a modality that is small or easy does not disappear beside a
    large one. CardioVAE weights its ECG ten times its radiograph.
``scale_factor``
    Applied to the summed reconstruction, bringing it into range of the KL term.
``annealing_factor``
    Raised from 0 to 1 over the first epochs, so the model learns to reconstruct before
    it is asked to regularise -- without which a strong decoder collapses the posterior
    to the prior and the encoder learns nothing.

**Per-sample, not per-batch.** Every loss here reduces to a mean over the batch, so a
value is comparable between a batch of 16 and a batch of 512, and the learning rate
does not have to be retuned when the batch size changes. The cardiac reference
implementations sum over the batch and divide by its size at the call site, which gives
the same number by a longer route.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from kalecardiac.model.layers.gaussian import GaussianPosterior

#: Bounds on a log-variance before it is exponentiated. An encoder that has not yet
#: learned anything can emit a large log-variance early in training, and ``exp`` of it
#: overflows to ``inf``, which reaches every parameter as a NaN gradient in one step.
LOG_VAR_LIMIT = 10.0


def gaussian_kl_divergence(posterior: GaussianPosterior, free_bits: float = 0.0) -> Tensor:
    """KL divergence from a diagonal Gaussian posterior to the standard normal.

    The closed form ``-0.5 * sum(1 + log_var - mean^2 - exp(log_var))``, summed over
    the latent dimensions and averaged over the batch.

    Args:
        posterior: The ``(B, D)`` posterior.
        free_bits: Nats per dimension that are not penalised. Above zero this is the
            free-bits trick: a dimension carrying less than this much information costs
            nothing, which stops a strong decoder switching dimensions off one by one.
            0 is the ordinary KL.

    Returns:
        A scalar: the mean KL per subject.

    Raises:
        ValueError: If ``free_bits`` is negative.
    """
    if free_bits < 0:
        raise ValueError(f"free_bits must be non-negative, got {free_bits}")

    log_var = posterior.log_var.clamp(-LOG_VAR_LIMIT, LOG_VAR_LIMIT)
    per_dimension = -0.5 * (1 + log_var - posterior.mean.pow(2) - log_var.exp())
    if free_bits > 0:
        per_dimension = per_dimension.clamp_min(free_bits)
    return per_dimension.sum(dim=-1).mean()


def reconstruction_loss(
    reconstruction: Tensor,
    target: Tensor,
    present: Tensor | None = None,
    kind: str = "mse",
) -> Tensor:
    """Reconstruction error for one modality, summed over its elements.

    Summed over the modality's own axes and averaged over the batch, so the value is
    the error per subject. Summing rather than averaging over the elements is what
    makes the term commensurable with a KL that is also summed over dimensions -- and
    what makes ``weights`` and ``scale_factor`` necessary, since the sum then scales
    with the modality's size.

    Args:
        reconstruction: ``(B, ...)`` model output.
        target: ``(B, ...)`` ground truth, the same shape.
        present: ``(B,)`` boolean, false where the modality was absent for that
            subject. Absent subjects contribute nothing, since their target is a
            placeholder rather than a recording; if no subject has the modality the
            loss is zero.
        kind: ``"mse"`` for a real-valued modality, ``"bce"`` for one in the unit
            interval whose decoder ends in a sigmoid.

    Returns:
        A scalar: the mean summed error per subject that has the modality.

    Raises:
        ValueError: If the shapes disagree, or ``kind`` is unknown.
    """
    if reconstruction.shape != target.shape:
        raise ValueError(f"reconstruction {tuple(reconstruction.shape)} and target {tuple(target.shape)} must match")

    if kind == "mse":
        elementwise = nn.functional.mse_loss(reconstruction, target, reduction="none")
    elif kind == "bce":
        elementwise = nn.functional.binary_cross_entropy(
            reconstruction.clamp(0.0, 1.0), target.clamp(0.0, 1.0), reduction="none"
        )
    else:
        raise ValueError(f"unknown reconstruction kind {kind!r}; available: 'mse', 'bce'")

    per_subject = elementwise.flatten(start_dim=1).sum(dim=1)
    if present is None:
        return per_subject.mean()

    weight = present.to(device=per_subject.device, dtype=per_subject.dtype)
    count = weight.sum()
    if float(count) == 0.0:
        # Zero rather than NaN, and still attached to the graph so that a batch in
        # which one modality happens to be wholly absent does not break the backward
        # pass for the modalities that are present.
        return (per_subject * weight).sum()
    return (per_subject * weight).sum() / count


def elbo_loss(
    reconstructions: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    posterior: GaussianPosterior,
    weights: Mapping[str, float] | None = None,
    present: Mapping[str, Tensor] | None = None,
    kinds: Mapping[str, str] | None = None,
    annealing_factor: float = 1.0,
    scale_factor: float = 1.0,
    free_bits: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """The multimodal evidence lower bound, as a loss to minimise.

    ``scale_factor * sum_m weight_m * recon_m + annealing_factor * KL``.

    Args:
        reconstructions: One reconstruction per modality.
        targets: The corresponding inputs, keyed the same way. A modality present in
            ``reconstructions`` but not here is skipped.
        posterior: The joint posterior the latent was sampled from. It must be *that*
            posterior: regularising a distribution the model did not sample from leaves
            the sampled one unconstrained.
        weights: Reconstruction weight per modality; 1.0 where unnamed.
        present: ``(B,)`` boolean per modality, so an absent modality's placeholder is
            not reconstructed against.
        kinds: ``"mse"`` or ``"bce"`` per modality; ``"mse"`` where unnamed.
        annealing_factor: Weight on the KL term, ramped from 0 to 1 over the first
            epochs of training.
        scale_factor: Applied to the summed reconstruction term.
        free_bits: Nats per latent dimension exempt from the KL penalty.

    Returns:
        ``(loss, parts)`` -- the scalar objective, and its terms as floats for logging.

    Raises:
        ValueError: If no modality can be reconstructed.
    """
    weights = dict(weights or {})
    kinds = dict(kinds or {})
    shared = [name for name in reconstructions if name in targets]
    if not shared:
        raise ValueError(
            f"no modality to reconstruct: the model decodes {sorted(reconstructions)} and the batch carries "
            f"{sorted(targets)}"
        )

    parts: dict[str, float] = {}
    total = torch.zeros((), device=posterior.mean.device, dtype=posterior.mean.dtype)
    for name in shared:
        term = reconstruction_loss(
            reconstructions[name],
            targets[name],
            present=None if present is None else present.get(name),
            kind=kinds.get(name, "mse"),
        )
        weighted = float(weights.get(name, 1.0)) * term
        total = total + weighted
        parts[f"recon_{name}"] = float(term.detach())

    kl = gaussian_kl_divergence(posterior, free_bits=free_bits)
    parts["recon"] = float(total.detach())
    parts["kl"] = float(kl.detach())
    return scale_factor * total + annealing_factor * kl, parts


def latent_alignment_loss(
    modality_posteriors: Mapping[str, GaussianPosterior],
    joint: GaussianPosterior,
    present: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Pull each modality's posterior mean towards the joint one.

    The regulariser LS-EMVAE adds to its ELBO. Its purpose is transfer rather than
    reconstruction: nothing in an ELBO requires the per-modality posteriors to agree
    with each other, so a lead's encoder can settle in its own corner of the latent
    space and be useless on its own. Penalising the distance to the joint mean makes
    every modality's representation readable in the same coordinates -- which is what a
    downstream model relies on when it is fine-tuned on a *subset* of the modalities
    the pretraining used.

    Mean squared error between means, averaged over modalities so the scale does not
    depend on how many there are. The joint is detached: this is a term that moves the
    modality encoders towards the consensus, not one that drags the consensus towards
    whichever modality is furthest away.

    Args:
        modality_posteriors: Each modality's own posterior, before fusion.
        joint: The fused posterior.
        present: ``(B,)`` boolean per modality; absent subjects are skipped.

    Returns:
        A scalar, or zero if there is nothing to align.
    """
    if not modality_posteriors:
        return torch.zeros((), device=joint.mean.device, dtype=joint.mean.dtype)

    target = joint.mean.detach()
    terms = []
    for name, posterior in modality_posteriors.items():
        squared = (posterior.mean - target).pow(2).mean(dim=-1)
        mask = None if present is None else present.get(name)
        if mask is None:
            terms.append(squared.mean())
            continue
        weight = mask.to(device=squared.device, dtype=squared.dtype)
        count = weight.sum()
        terms.append((squared * weight).sum() / count if float(count) else (squared * weight).sum())
    return torch.stack(terms).mean()


def binary_cross_entropy(prediction: Tensor, targets: Tensor, pos_weight: Tensor | None = None) -> Tensor:
    """Cross-entropy on logits, for a binary endpoint.

    Logits rather than probabilities: the fused sigmoid-and-log form is the numerically
    stable one.

    Args:
        prediction: ``(B,)`` or ``(B, 1)`` logits.
        targets: ``(B,)`` targets, 1 positive and 0 negative.
        pos_weight: Weight on the positive class, for an imbalanced endpoint. It
            changes calibration and the optimisation path but not the ranking, so its
            effect on ROC-AUC is indirect.

    Returns:
        A scalar: the mean loss per subject.
    """
    return nn.functional.binary_cross_entropy_with_logits(
        prediction.reshape(-1), targets.reshape(-1).float(), pos_weight=pos_weight
    )


def cross_entropy(prediction: Tensor, targets: Tensor, class_weight: Tensor | None = None) -> Tensor:
    """Cross-entropy on logits, for a multiclass endpoint.

    Args:
        prediction: ``(B, num_classes)`` logits.
        targets: ``(B,)`` class indices. Floats are accepted and rounded, because a
            target read from a numeric column arrives as a float.
        class_weight: ``(num_classes,)`` weight per class.

    Returns:
        A scalar: the mean loss per subject.

    Raises:
        ValueError: If ``prediction`` is not 2-D with more than one column.
    """
    if prediction.dim() != 2 or prediction.shape[1] < 2:
        raise ValueError(f"expected (batch, num_classes >= 2) logits, got {tuple(prediction.shape)}")
    return nn.functional.cross_entropy(prediction, targets.reshape(-1).long(), weight=class_weight)


def mean_squared_error(prediction: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error, for a continuous endpoint.

    Args:
        prediction: ``(B,)`` or ``(B, 1)`` predictions.
        targets: ``(B,)`` targets.

    Returns:
        A scalar: the mean squared error per subject.
    """
    return nn.functional.mse_loss(prediction.reshape(-1), targets.reshape(-1).float())


def class_weight_from_labels(labels: Tensor, num_classes: int = 2) -> Tensor:
    """Inverse-frequency weights from a training split's labels.

    A cardiac endpoint is often imbalanced -- a few hundred subjects, a fifth of them
    positive -- and this is the standard counterweight. It is computed from the
    *training* labels and nowhere else: frequencies taken over the whole cohort carry
    the test split's class balance into the loss.

    Args:
        labels: ``(N,)`` class indices from the training split.
        num_classes: How many classes there are.

    Returns:
        ``(num_classes,)`` weights, normalised to mean one. A class absent from the
        split gets weight one rather than infinity.

    Raises:
        ValueError: If ``num_classes`` is below 2.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")

    counts = torch.bincount(labels.reshape(-1).long(), minlength=num_classes).float()
    weights = torch.where(counts > 0, counts.sum() / counts.clamp_min(1.0), torch.ones_like(counts))
    return weights / weights.mean()


def positive_weight_from_labels(labels: Tensor) -> Tensor:
    """The ratio of negatives to positives, for :func:`binary_cross_entropy`.

    Args:
        labels: ``(N,)`` binary labels from the training split.

    Returns:
        A 0-d tensor. One when there are no positives, since a weight is meaningless
        then and a division by zero would poison the whole loss.
    """
    values = labels.reshape(-1).float()
    positives = float(values.sum())
    negatives = float(values.numel() - positives)
    return torch.tensor(negatives / positives if positives else 1.0, dtype=torch.float32)
