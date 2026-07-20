"""
DataLoader construction for train/validation splits.

DIV2K ships with pre-separated train/valid folders, so no manual splitting logic
is needed -- we just point each split at its own directory.
"""

from torch.utils.data import DataLoader

from src.dataset import DIV2KDataset, list_image_paths
from src.transforms import get_train_transform, get_eval_transform


def get_train_loader(
    data_dir="data/DIV2K_train_HR",
    crop_size=128,
    batch_size=32,
    num_workers=4,
    shuffle=True,
):
    paths = list_image_paths(data_dir)
    dataset = DIV2KDataset(paths, crop_size=crop_size, transform=get_train_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # keeps batch shapes consistent, simplifies later model code
    )


def get_val_loader(
    data_dir="data/DIV2K_valid_HR",
    crop_size=128,
    batch_size=32,
    num_workers=2,
):
    paths = list_image_paths(data_dir)
    dataset = DIV2KDataset(paths, crop_size=crop_size, transform=get_eval_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # no need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Quick smoke test: python -m src.dataloader
    train_loader = get_train_loader()
    batch = next(iter(train_loader))
    print(f"Train batch shape: {batch.shape}")  # expect [B, 3, crop_size, crop_size]
    print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")  # expect ~[0, 1]
