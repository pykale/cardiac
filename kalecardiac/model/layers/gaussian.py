"""Combining Gaussian experts, and the sampling that follows.

Every multimodal VAE in this package is the same model with a different answer to one
question: *given one diagonal Gaussian per modality, what is the joint posterior?*
This module holds the answers, as functions of stacked tensors that know nothing about
modalities, leads or cohorts.

.. code-block:: text

    product_of_experts     precision-weighted product; a confident expert dominates
    mixture_of_experts     weighted arithmetic mean; every expert contributes
    hierarchical_experts   product within groups, mixture across them  (HiME)
    mean_of_experts        unweighted mean, the plainest baseline
    wasserstein_barycenter mean of means and of standard deviations

Naming the mechanisms rather than the papers is what makes them comparable: a
product-of-experts VAE, a mixture-of-experts VAE and a hierarchical one differ by the
string passed to :func:`build_expert_fusion` and by nothing else, so an ablation is a
configuration change.

**Every function takes the same shapes.** ``mean`` and ``log_var`` are ``(M, B, D)``
for ``M`` experts, and the optional ``mask`` is ``(M, B)``, one where that expert is
present for that sample. A masked-out expert contributes nothing -- negligible
precision to a product, zero weight to a mixture -- which is how a model trained on
twelve leads runs on six without being retrained.

:func:`product_of_experts` delegates to :class:`kale.embed.multimodal_fusion.ProductOfExperts`,
PyKale's implementation of the same closed form.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from kale.embed.multimodal_fusion import ProductOfExperts
from torch import Tensor

#: Log-variance assigned to an absent expert. ``exp(20)`` is a variance of about 5e8,
#: so the expert's precision is negligible beside any real one and it drops out of a
#: product without special-casing. Large but finite: ``inf`` would make the precision
#: exactly zero and the gradient a NaN.
ABSENT_LOG_VAR = 20.0

#: Floor on a variance, keeping a reciprocal finite.
EPS = 1e-8

_poe = ProductOfExperts()


@dataclass(frozen=True)
class GaussianPosterior:
    """A diagonal Gaussian over the latent space.

    What every encoder in this package returns and every fusion consumes, so that
    "the posterior" is one value rather than a pair a caller might swap.

    Attributes:
        mean (Tensor): ``(B, D)`` or ``(M, B, D)`` mean.
        log_var (Tensor): Log-variance, the same shape as ``mean``. Log rather than
            variance because an encoder emits an unconstrained real number, and
            exponentiating is what keeps the variance positive without a clamp.
    """

    mean: Tensor
    log_var: Tensor

    def __post_init__(self) -> None:
        if self.mean.shape != self.log_var.shape:
            raise ValueError(
                f"mean and log_var must have the same shape, got {self.mean.shape} and {self.log_var.shape}"
            )

    @property
    def variance(self) -> Tensor:
        """The variance, ``exp(log_var)``."""
        return torch.exp(self.log_var)

    @property
    def stddev(self) -> Tensor:
        """The standard deviation, ``exp(log_var / 2)``."""
        return torch.exp(0.5 * self.log_var)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        """Draw one sample with the reparameterisation trick; see :func:`reparameterise`."""
        return reparameterise(self.mean, self.log_var, generator=generator)

    def detach(self) -> GaussianPosterior:
        """A copy detached from the graph, for interpretation or logging."""
        return GaussianPosterior(self.mean.detach(), self.log_var.detach())

    def to(self, device: torch.device | str) -> GaussianPosterior:
        """A copy on ``device``."""
        return GaussianPosterior(self.mean.to(device), self.log_var.to(device))


def reparameterise(mean: Tensor, log_var: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Sample from ``N(mean, exp(log_var))`` so that the draw is differentiable.

    ``z = mean + eps * std`` with ``eps ~ N(0, 1)``: the randomness is an input rather
    than an operation, so a gradient reaches ``mean`` and ``log_var``.

    Unlike the two cardiac implementations this refactors, this does **not** return the
    mean when the module is in evaluation mode. Whether to sample is a property of what
    the caller is doing -- an ELBO needs a sample, a downstream embedding wants the
    mean -- not of a module flag, and tying it to ``training`` silently changes the
    objective the moment a validation pass runs. A caller that wants the mean asks for
    ``posterior.mean``, which is what :class:`~kalecardiac.model.embed.LatentEmbedder`
    does.

    Args:
        mean: ``(..., D)`` mean.
        log_var: ``(..., D)`` log-variance.
        generator: RNG, for a reproducible draw.

    Returns:
        A sample of the same shape as ``mean``.
    """
    std = torch.exp(0.5 * log_var)
    noise = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
    return mean + noise * std


def prior_expert(shape: Sequence[int], device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
    """The standard-normal expert, as ``(mean, log_var)`` of zeros.

    Included in a product so that a sample with *no* present modality still has a
    defined posterior rather than a zero-divided-by-zero one, and so that the product
    of a single expert with the prior is a proper posterior rather than that expert's
    own distribution. This is the convention multimodal VAEs have used since Wu and
    Goodman (2018).

    Args:
        shape: Shape of the expert, usually ``(1, batch, latent_dim)``.
        device: Where to allocate it.
        dtype: Element type.

    Returns:
        ``(mean, log_var)``, both zeros of ``shape``.
    """
    zeros = torch.zeros(tuple(shape), device=device, dtype=dtype)
    return zeros, zeros.clone()


def _check(mean: Tensor, log_var: Tensor, mask: Tensor | None) -> Tensor:
    """Validate expert tensors and return a usable ``(M, B)`` float mask."""
    if mean.shape != log_var.shape:
        raise ValueError(f"mean and log_var must have the same shape, got {mean.shape} and {log_var.shape}")
    if mean.dim() != 3:
        raise ValueError(f"expected (num_experts, batch, latent_dim) tensors, got shape {tuple(mean.shape)}")
    if mean.shape[0] == 0:
        raise ValueError("need at least one expert to fuse")
    if mask is None:
        return torch.ones(mean.shape[:2], device=mean.device, dtype=mean.dtype)
    if mask.shape != mean.shape[:2]:
        raise ValueError(f"mask must be {tuple(mean.shape[:2])}, got {tuple(mask.shape)}")
    return mask.to(device=mean.device, dtype=mean.dtype)


def _mask_experts(mean: Tensor, log_var: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Give absent experts a negligible precision, so a product ignores them."""
    absent = (1.0 - mask).unsqueeze(-1)
    return mean * (1.0 - absent), log_var * (1.0 - absent) + ABSENT_LOG_VAR * absent


def _normalised_share(mask: Tensor, weights: Tensor | None) -> Tensor:
    """Per-expert weights that sum to one over the expert axis, honouring ``mask``.

    Shared by every averaging fusion, so that "an absent expert takes no share" is
    implemented once.

    Raises:
        ValueError: If ``weights`` has the wrong shape or a negative entry.
    """
    if weights is None:
        share = mask
    else:
        weights = weights.to(device=mask.device, dtype=mask.dtype)
        # Checked before expanding: expand_as would otherwise raise a RuntimeError
        # about tensor sizes, which does not say that the expert count is wrong.
        if weights.dim() == 1 and weights.shape[0] == mask.shape[0]:
            weights = weights.unsqueeze(1).expand_as(mask)
        if weights.shape != mask.shape:
            raise ValueError(f"weights must be ({mask.shape[0]},) or {tuple(mask.shape)}, got {tuple(weights.shape)}")
        if bool((weights < 0).any()):
            raise ValueError("weights must be non-negative")
        share = weights * mask

    # A sample with nothing present would divide by zero; falling back to equal weights
    # over every expert returns an average rather than a NaN, and the caller's mask
    # already says the result is not evidence.
    share = torch.where(share.sum(dim=0, keepdim=True) > 0, share, torch.ones_like(share))
    return share / share.sum(dim=0, keepdim=True).clamp_min(EPS)


def product_of_experts(mean: Tensor, log_var: Tensor, mask: Tensor | None = None) -> GaussianPosterior:
    """Fuse experts as a product of Gaussians.

    The product of diagonal Gaussians is itself Gaussian, with precision the sum of
    the experts' precisions and mean their precision-weighted average. The consequence
    worth knowing is that the joint is *never less certain than any expert*: adding a
    modality can only narrow the posterior, which is what makes a product the right
    mechanism when the modalities are views of one underlying state -- as the leads of
    one heartbeat are.

    Args:
        mean: ``(M, B, D)`` expert means.
        log_var: ``(M, B, D)`` expert log-variances.
        mask: ``(M, B)``, one where the expert is present. ``None`` treats all as
            present.

    Returns:
        The ``(B, D)`` joint posterior.

    Raises:
        ValueError: If the shapes are inconsistent.
    """
    mask = _check(mean, log_var, mask)
    masked_mean, masked_log_var = _mask_experts(mean, log_var, mask)
    joint_mean, joint_log_var = _poe(masked_mean, masked_log_var, eps=EPS)
    return GaussianPosterior(joint_mean, joint_log_var)


def mixture_of_experts(
    mean: Tensor,
    log_var: Tensor,
    mask: Tensor | None = None,
    weights: Tensor | None = None,
) -> GaussianPosterior:
    """Fuse experts as a weighted arithmetic mean of their parameters.

    Averaging the parameters, not mixing the distributions: a true mixture of Gaussians
    is not Gaussian, and the moment-matched average is what a VAE needs to keep the KL
    term in closed form. The trade against a product is that no expert can dominate --
    a confident one and an uninformative one contribute equally -- which keeps a weak
    modality from being ignored, and keeps a noisy one from being discounted.

    Args:
        mean: ``(M, B, D)`` expert means.
        log_var: ``(M, B, D)`` expert log-variances.
        mask: ``(M, B)``, one where the expert is present. Absent experts take no share
            of the weight, and a sample with no present expert falls back to the
            unweighted mean rather than dividing by zero.
        weights: ``(M,)`` or ``(M, B)`` relative weights, normalised internally.
            ``None`` weights the experts equally.

    Returns:
        The ``(B, D)`` joint posterior.

    Raises:
        ValueError: If the shapes are inconsistent, or a weight is negative.
    """
    mask = _check(mean, log_var, mask)
    share = _normalised_share(mask, weights).unsqueeze(-1)
    return GaussianPosterior((share * mean).sum(dim=0), (share * log_var).sum(dim=0))


def mean_of_experts(mean: Tensor, log_var: Tensor, mask: Tensor | None = None) -> GaussianPosterior:
    """Fuse experts as an unweighted mean of their parameters.

    :func:`mixture_of_experts` with equal weights, named separately because it is the
    plainest possible fusion and so the baseline an ablation compares against.

    Args:
        mean: ``(M, B, D)`` expert means.
        log_var: ``(M, B, D)`` expert log-variances.
        mask: ``(M, B)``, one where the expert is present.

    Returns:
        The ``(B, D)`` joint posterior.
    """
    return mixture_of_experts(mean, log_var, mask=mask)


def wasserstein_barycenter(
    mean: Tensor,
    log_var: Tensor,
    mask: Tensor | None = None,
    weights: Tensor | None = None,
) -> GaussianPosterior:
    """Fuse experts as their 2-Wasserstein barycenter.

    For diagonal Gaussians the barycenter has the weighted mean of the means and the
    weighted mean of the *standard deviations* -- not of the variances, and not of the
    precisions. It therefore sits between a product and a mixture: unlike a product it
    cannot become more certain than its experts, and unlike a parameter-averaged
    mixture it averages in the scale a distance is measured in.

    Args:
        mean: ``(M, B, D)`` expert means.
        log_var: ``(M, B, D)`` expert log-variances.
        mask: ``(M, B)``, one where the expert is present.
        weights: ``(M,)`` or ``(M, B)`` relative weights. ``None`` weights equally.

    Returns:
        The ``(B, D)`` joint posterior.
    """
    mask = _check(mean, log_var, mask)
    share = _normalised_share(mask, weights).unsqueeze(-1)
    barycentre_mean = (share * mean).sum(dim=0)
    # The scale is averaged as a standard deviation, which is what makes this a
    # barycenter rather than a variance- or precision-weighted average.
    barycentre_stddev = (share * torch.exp(0.5 * log_var)).sum(dim=0)
    return GaussianPosterior(barycentre_mean, 2.0 * torch.log(barycentre_stddev.clamp_min(EPS)))


def expert_groups(num_experts: int, num_groups: int) -> list[list[int]]:
    """Divide ``num_experts`` expert indices into ``num_groups`` contiguous groups.

    Every expert lands in exactly one group, and the groups differ in size by at most
    one. That last property is a deliberate correction to the implementation this
    refactors, which computed ``group_size = num_experts // num_groups`` and took
    ``num_groups`` slices of that width: with thirteen experts (twelve leads and the
    prior) and four groups it took four groups of three and **silently discarded the
    thirteenth**, which was the prior. See :doc:`the model reference </models>` for the
    consequence.

    Args:
        num_experts: How many experts there are.
        num_groups: How many groups to form. Clamped to ``num_experts`` so that asking
            for more groups than experts gives one expert per group rather than empty
            groups.

    Returns:
        Expert indices per group, in order.

    Raises:
        ValueError: If either argument is below one.
    """
    if num_experts < 1:
        raise ValueError(f"need at least one expert, got {num_experts}")
    if num_groups < 1:
        raise ValueError(f"need at least one group, got {num_groups}")

    num_groups = min(num_groups, num_experts)
    size, remainder = divmod(num_experts, num_groups)
    groups, start = [], 0
    for index in range(num_groups):
        width = size + (1 if index < remainder else 0)
        groups.append(list(range(start, start + width)))
        start += width
    return groups


def hierarchical_experts(
    mean: Tensor,
    log_var: Tensor,
    mask: Tensor | None = None,
    num_groups: int = 4,
) -> GaussianPosterior:
    """Fuse experts hierarchically: a product within groups, a mixture across them.

    The hierarchical modality expert (HiME) mechanism of LS-EMVAE. Experts that should
    reinforce one another are multiplied, and the group summaries are then averaged, so
    the joint gains the sharpness of a product without letting one over-confident group
    determine the latent on its own.

    Grouping is contiguous over the expert axis, so the *order* experts are stacked in
    decides which reinforce which. For an ECG stacked in the conventional lead order
    that puts the limb leads in the early groups and the precordial leads in the later
    ones, which is anatomically sensible; a caller stacking leads in another order is
    choosing another grouping, and should say so.

    A group whose experts are all absent takes no share of the mixture.

    Args:
        mean: ``(M, B, D)`` expert means.
        log_var: ``(M, B, D)`` expert log-variances.
        mask: ``(M, B)``, one where the expert is present.
        num_groups: Groups to divide the experts into.

    Returns:
        The ``(B, D)`` joint posterior.

    Raises:
        ValueError: If the shapes are inconsistent, or ``num_groups`` is below one.
    """
    mask = _check(mean, log_var, mask)
    groups = expert_groups(mean.shape[0], num_groups)

    group_means, group_log_vars, group_present = [], [], []
    for indices in groups:
        index = torch.tensor(indices, device=mean.device, dtype=torch.long)
        fused = product_of_experts(
            mean.index_select(0, index), log_var.index_select(0, index), mask.index_select(0, index)
        )
        group_means.append(fused.mean)
        group_log_vars.append(fused.log_var)
        # A group contributes if any of its experts does, so a subject missing one lead
        # still has that lead's group represented by its siblings.
        group_present.append(mask.index_select(0, index).amax(dim=0))

    return mixture_of_experts(torch.stack(group_means), torch.stack(group_log_vars), mask=torch.stack(group_present))


#: Latent fusion mechanisms, by name. Each takes ``(mean, log_var, mask)`` and any
#: mechanism-specific keyword, and returns a :class:`GaussianPosterior`.
EXPERT_FUSIONS: dict[str, Callable[..., GaussianPosterior]] = {
    "poe": product_of_experts,
    "moe": mixture_of_experts,
    "hime": hierarchical_experts,
    "mean": mean_of_experts,
    "barycenter": wasserstein_barycenter,
}


def build_expert_fusion(method: str, **kwargs) -> Callable[..., GaussianPosterior]:
    """Return a latent fusion by name, with its options bound.

    Args:
        method: One of :data:`EXPERT_FUSIONS`.
        **kwargs: Mechanism-specific options, e.g. ``num_groups`` for ``"hime"``.
            Options a mechanism does not take are rejected here rather than at the
            first forward pass, so a configuration typo fails at construction.

    Returns:
        A callable taking ``(mean, log_var, mask)``.

    Raises:
        KeyError: If ``method`` is not registered.
        TypeError: If an option does not apply to ``method``.
    """
    if method not in EXPERT_FUSIONS:
        raise KeyError(f"unknown latent fusion {method!r}; available: {sorted(EXPERT_FUSIONS)}")
    fusion = EXPERT_FUSIONS[method]

    accepted = {"hime": {"num_groups"}, "moe": {"weights"}, "barycenter": {"weights"}}.get(method, set())
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise TypeError(f"latent fusion {method!r} takes no option(s) {unknown}; it accepts {sorted(accepted)}")

    def fuse(mean: Tensor, log_var: Tensor, mask: Tensor | None = None) -> GaussianPosterior:
        return fusion(mean, log_var, mask=mask, **kwargs)

    fuse.__name__ = f"{method}_fusion"
    return fuse


class GaussianHead(torch.nn.Module):
    """Project features to the mean and log-variance of a diagonal Gaussian.

    The last layer of every variational encoder here: two linear maps from a shared
    feature vector, which is what turns any backbone into a variational one.

    Args:
        in_features: Width of the incoming features.
        latent_dim: Width of the latent space.

    Raises:
        ValueError: If either width is not positive.
    """

    def __init__(self, in_features: int, latent_dim: int) -> None:
        super().__init__()
        if in_features < 1 or latent_dim < 1:
            raise ValueError(f"widths must be positive, got in_features={in_features}, latent_dim={latent_dim}")
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.to_mean = torch.nn.Linear(in_features, latent_dim)
        self.to_log_var = torch.nn.Linear(in_features, latent_dim)

    def forward(self, features: Tensor) -> GaussianPosterior:
        """Map ``(B, in_features)`` features to a ``(B, latent_dim)`` posterior.

        Raises:
            ValueError: If ``features`` is not ``(B, in_features)``.
        """
        if features.dim() != 2 or features.shape[1] != self.in_features:
            raise ValueError(f"expected (batch, {self.in_features}) features, got {tuple(features.shape)}")
        return GaussianPosterior(self.to_mean(features), self.to_log_var(features))
