"""
DIV2K Dataset and preprocessing/augmentation transforms.

Contract:
    DIV2KDataset(root_dir, crop_size, augment=True) -> torch.utils.data.Dataset
    __getitem__ returns a single tensor (3, crop_size, crop_size), float32, [0,1]
    (autoencoders are self-supervised: input == target, no labels needed)

Preprocessing steps:
    - Load image with PIL, convert to RGB
    - Random crop to `crop_size` (training) / center crop (validation, no augment)
    - If the image is smaller than crop_size in either dimension (shouldn't
      happen with DIV2K, but real-world safety net), reflect-pad first
    - Augmentations (train only): horizontal flip, rotation, color jitter
    - Normalize to [0, 1] (NOT ImageNet mean/std — we want raw pixel
      reconstruction; ImageNet stats would bias toward classification
      features we don't need here)
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

# DIV2K images are large (~2K px), so random cropping many patches per
# image per epoch is standard practice — otherwise 800 images is a tiny
# dataset for a conv net.
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class DIV2KDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int, augment: bool = True,
                 augmentation_config: dict = None):
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.augment = augment

        # Defaults mirror configs/config.yaml -> dataset.augmentation
        cfg = augmentation_config or {}
        self.horizontal_flip = cfg.get("horizontal_flip", True)
        self.vertical_flip = cfg.get("vertical_flip", False)
        self.random_rotation_deg = cfg.get("random_rotation_deg", 0)
        self.color_jitter = cfg.get("color_jitter", False)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}. "
                f"Did you copy DIV2K images into this folder?"
            )

        self.image_paths = sorted(
            p for p in self.root_dir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir}. Expected .png/.jpg files "
                f"copied directly from DIV2K_train_HR or DIV2K_valid_HR."
            )

        self._jitter = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        ) if self.color_jitter else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, idx: int) -> Image.Image:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return img

    def _pad_if_needed(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        pad_w = max(0, self.crop_size - w)
        pad_h = max(0, self.crop_size - h)
        if pad_w > 0 or pad_h > 0:
            # reflect padding avoids introducing flat/black borders that
            # would leak into the learned reconstruction
            img = TF.pad(img, padding=[0, 0, pad_w, pad_h], padding_mode="reflect")
        return img

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._load_image(idx)
        img = self._pad_if_needed(img)

        if self.augment:
            # Random crop location differs every epoch -> effectively
            # infinite unique training patches from 800 source images
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(self.crop_size, self.crop_size)
            )
            img = TF.crop(img, i, j, h, w)

            if self.horizontal_flip and torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
            if self.vertical_flip and torch.rand(1).item() < 0.5:
                img = TF.vflip(img)
            if self.random_rotation_deg > 0:
                angle = (torch.rand(1).item() * 2 - 1) * self.random_rotation_deg
                # reflect fill avoids black corner artifacts from rotation
                img = TF.rotate(img, angle, fill=0)
                # rotation can reintroduce below-crop-size edges after
                # black-corner crop; re-pad+crop defensively
                img = self._pad_if_needed(img)
                img = TF.center_crop(img, [self.crop_size, self.crop_size])
            if self._jitter is not None:
                img = self._jitter(img)
        else:
            # Validation: deterministic center crop, no augmentation —
            # we want stable, comparable metrics epoch to epoch
            img = TF.center_crop(img, [self.crop_size, self.crop_size])

        tensor = TF.to_tensor(img)  # -> (3, H, W), float32, already scaled to [0,1]
        return tensor


def get_dataloaders(config: dict):
    """
    Builds train and val DataLoaders directly from the config dict
    (configs/config.yaml). This is the single call training/train.py
    needs to make — keeps train.py free of dataset construction details.
    """
    from torch.utils.data import DataLoader

    ds_cfg = config["dataset"]
    train_cfg = config["training"]

    train_dataset = DIV2KDataset(
        root_dir=ds_cfg["train_dir"],
        crop_size=ds_cfg["crop_size"],
        augment=True,
        augmentation_config=ds_cfg["augmentation"],
    )
    val_dataset = DIV2KDataset(
        root_dir=ds_cfg["val_dir"],
        crop_size=ds_cfg["crop_size"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick smoke test — run directly to sanity check your DIV2K copy:
    #   python src/dataset.py
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "dataset/train"
    ds = DIV2KDataset(root_dir=root, crop_size=256, augment=True)
    print(f"Found {len(ds)} images in {root}")
    sample = ds[0]
    print(f"Sample tensor shape: {tuple(sample.shape)}, "
          f"dtype: {sample.dtype}, range: [{sample.min():.3f}, {sample.max():.3f}]")