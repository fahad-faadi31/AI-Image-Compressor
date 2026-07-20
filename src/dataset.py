"""
DIV2K Dataset for patch-based training.

Design notes (see project discussion for full reasoning):
- We store only file paths in __init__, not loaded images -- loading 800+ 2K-resolution
  images into memory upfront would be slow to start and wasteful, since we only ever
  need small random crops from each.
- __getitem__ takes a *random* crop each time it's called, rather than a fixed
  non-overlapping grid. With only ~800 training images, random cropping acts as
  implicit data augmentation (the model effectively never sees the exact same input
  twice across epochs) and avoids the model learning artificial patch-boundary
  artifacts that a fixed grid would introduce.
- Guard clause: if an image is smaller than crop_size in either dimension (shouldn't
  happen with DIV2K, but real pipelines don't assume "it probably won't happen"), we
  upscale it, preserving aspect ratio, until the smaller side reaches crop_size.
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_image_paths(root_dir):
    """Return a sorted list of image file paths in root_dir (non-recursive)."""
    paths = [
        os.path.join(root_dir, fname)
        for fname in os.listdir(root_dir)
        if fname.lower().endswith(IMG_EXTENSIONS)
    ]
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No images found in '{root_dir}'. "
            f"Did you place the DIV2K images there? See README.md."
        )
    return sorted(paths)


class DIV2KDataset(Dataset):
    def __init__(self, image_paths, crop_size=128, transform=None):
        """
        Args:
            image_paths (list[str]): paths to full-resolution DIV2K images.
            crop_size (int): side length of the square random crop.
            transform: a torchvision-style transform applied to the PIL crop
                (augmentation + ToTensor). See transforms.py.
        """
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        # Note: this is the number of *source images*, not the number of unique
        # patches -- because crops are random, "one epoch" here is a loose notion.
        # This is standard practice for patch-based training.
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        width, height = img.size

        # Guard: upscale (preserving aspect ratio) if image is smaller than crop_size.
        if width < self.crop_size or height < self.crop_size:
            scale = self.crop_size / min(width, height)
            new_w, new_h = int(width * scale) + 1, int(height * scale) + 1
            img = img.resize((new_w, new_h), Image.BICUBIC)
            width, height = img.size

        x = random.randint(0, width - self.crop_size)
        y = random.randint(0, height - self.crop_size)
        img = img.crop((x, y, x + self.crop_size, y + self.crop_size))

        if self.transform:
            img = self.transform(img)

        return img
