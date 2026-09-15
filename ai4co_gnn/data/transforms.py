import torch

from rl4co.data.transforms import dihedral_8_augmentation
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

log = get_pylogger(__name__)


def shift_and_dihedral_augmentation(
    xy: Tensor, num_augment: int = 8 * 8, first_augment: bool = True
) -> Tensor:
    # dihedral augmentation on the originals
    orig_xy = xy[: xy.shape[0] // (num_augment), ...] if first_augment else xy
    dihedral_aug = dihedral_8_augmentation(orig_xy)

    # shifts
    # Ensure `num_augment` is compatible with one dihedral augmentation (8 variants)
    if num_augment % 8 != 0:
        raise ValueError(
            "num_augment must be a multiple of 8 to be compatible with dihedral augmentation."
        )
    # Choose the number of random shifts so that:
    #   8 (dihedral aug on originals) + 8 * n_random_shift (aug on shifted copies)
    #   equals the requested `num_augment`
    n_random_shift = num_augment // 8 - 1
    shift_max = 0.2  # maximum shift in the range [0, 1]

    # generate all random shifts at once and apply in batch
    # shape trick: orig_xy.ndim−1 ones so that shift broadcasts over all but last dim
    shifts = (
        2 * torch.rand(n_random_shift, *([1] * (orig_xy.ndim - 1)), 2, device=xy.device)
        - 1
    ) * shift_max

    # orig_xy.unsqueeze(0) is (1, N, ..., 2), + shifts → (S, N, ..., 2)
    # then flatten the first two dims into batch
    shifted = (orig_xy.unsqueeze(0) + shifts).reshape(-1, *orig_xy.shape[1:])

    # one more dihedral call on all shifted copies
    augmented = torch.cat((dihedral_aug, dihedral_8_augmentation(shifted)), dim=0)
    assert (
        augmented.shape[0] == num_augment * orig_xy.shape[0]
    ), f"Expected {num_augment * orig_xy.shape[0]} augmentations, got {augmented.shape[0]}"
    return augmented
