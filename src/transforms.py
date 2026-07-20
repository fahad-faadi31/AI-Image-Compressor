"""
Augmentation pipeline for DIV2K patch training.

Design notes:
- RandomHorizontalFlip / RandomVerticalFlip: standard geometry-preserving augmentation.
- RandomDiscreteRotation: rotates by exactly 0/90/180/270 degrees. We deliberately do NOT
  use torchvision.transforms.RandomRotation here, because that samples a *continuous* angle
  and requires interpolation, which introduces blur/artifacts that would corrupt the exact
  pixel statistics we want the compression model to learn from. Rotating by multiples of
  90 degrees on a square crop is a pure pixel permutation -- no interpolation needed.
- ToTensor(): converts a uint8 PIL image in [0, 255] to a float32 tensor in [0, 1].
  This is where our [0,1] normalization decision (paired with Sigmoid at the decoder
  output) is actually implemented.
"""

import random
from torchvision import transforms


class RandomDiscreteRotation:
    """Rotates a PIL image by a random choice of 0, 90, 180, or 270 degrees.

    Unlike transforms.RandomRotation, this never interpolates pixels -- it's a
    lossless permutation, which matters for a compression pipeline where we want
    training patches to reflect real pixel statistics.
    """

    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        if angle == 0:
            return img
        return img.rotate(angle, expand=True)


def get_train_transform():
    """Augmentation + normalization pipeline used during training."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomDiscreteRotation(),
        transforms.ToTensor(),  # uint8 [0,255] HWC -> float32 [0,1] CHW
    ])


def get_eval_transform():
    """No augmentation for validation/evaluation -- just tensor conversion.

    We don't want random flips/rotations polluting evaluation metrics (PSNR/SSIM),
    since we want those to reflect the model's real reconstruction quality on
    unmodified validation crops.
    """
    return transforms.Compose([
        transforms.ToTensor(),
    ])
