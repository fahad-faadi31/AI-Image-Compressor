"""
Adaptive tiling pipeline — handles arbitrary input resolution.

This is the module that lets the API accept ANY image size (small icons up
to 8K photos) without a fixed input resolution requirement. See the
tiling.* section of configs/config.yaml.

Contract:
    split_into_tiles(image: np.ndarray, tile_size: int, overlap: int)
        -> (tiles: list[np.ndarray], layout: TileLayout)

    merge_tiles(tiles: list[np.ndarray], layout: TileLayout)
        -> np.ndarray  # full reconstructed image, feathered at overlaps

Steps implemented here (next step):
    1. If max(H, W) < tiling.min_size_for_tiling: skip tiling, process whole
       image directly (padded to a multiple of 8 for the conv stride).
    2. Otherwise: reflect-pad image to a multiple of tile_size, split into
       overlapping tiles, track original coordinates per tile.
    3. merge_tiles: place reconstructed tiles back, linearly feather the
       overlap regions so tile boundaries aren't visible, crop padding.

NOTE: Stub only. This is high-value, dependency-light code — good candidate
to implement early since it doesn't require the trained model to test
(can be unit tested with plain numpy arrays).
"""

import numpy as np


class TileLayout:
    """Stores tile coordinates + original/padded image dimensions for merging."""
    def __init__(self, original_size, padded_size, tile_size, overlap, positions):
        self.original_size = original_size
        self.padded_size = padded_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.positions = positions  # list of (y, x) top-left coords


def split_into_tiles(image: np.ndarray, tile_size: int, overlap: int):
    raise NotImplementedError


def merge_tiles(tiles: list, layout: TileLayout) -> np.ndarray:
    raise NotImplementedError
