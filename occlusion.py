"""Trait groups used to prevent fertile/sterile shortcut leakage."""

import torch


PAIRED_TRAITS = (("Ferstipe", "SterileStipe"), ("FertileFrond", "SterileFrond"))


def trait_groups(names, protocol="paired"):
    """Partition trait indices, independently of values and missingness."""
    if protocol not in {"paired", "individual"}:
        raise ValueError(f"Unknown occlusion protocol: {protocol}")
    names = list(names)
    groups = []
    assigned = set()
    if protocol == "paired":
        for pair in PAIRED_TRAITS:
            present = [name in names for name in pair]
            if any(present) and not all(present):
                raise ValueError(f"Incomplete leakage-prone trait pair: {pair}")
            if all(present):
                group = tuple(names.index(name) for name in pair)
                groups.append(group)
                assigned.update(group)
    groups.extend((i,) for i in range(len(names)) if i not in assigned)
    return groups


def group_for_trait(groups, index):
    return next(group for group in groups if index in group)


def grouped_random_mask(observed, groups, ratio, generator=None):
    """Sample one decision per species/group, scoring only observed cells."""
    mask = torch.zeros_like(observed, dtype=torch.bool)
    for group in groups:
        selected = torch.rand(observed.size(0), device=observed.device, generator=generator) < ratio
        mask[:, list(group)] = selected[:, None] & observed[:, list(group)]
    return mask
