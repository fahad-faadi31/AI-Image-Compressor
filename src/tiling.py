"""
Adaptive tiling pipeline — handles arbitrary input resolution.

This is the module that lets the API accept ANY image size (small icons up
to 8K photos) without a fixed input resolution requirement.

Design: every extracted tile is exactly `tile_size x tile_size` (matching
what the model was trained on — see configs/config.yaml dataset.crop_size),
using an overlapping stride rather than padding tiles to a larger size.
Overlapping regions are blended on merge with a feathered (linear ramp)
weight mask, so tile boundaries aren't visible as seams in the final image.

Contract:
    split_into_tiles(image, tile_size, overlap)
        -> (tiles: list[np.ndarray] of shape (tile_size, tile_size, C), layout: TileLayout)

    merge_tiles(tiles, layout)
        -> np.ndarray  # full reconstructed image, feathered at overlaps, cropped to original size

    adaptive_prepare(image, tile_size, overlap, min_size_for_tiling)
        -> decides whether tiling is needed at all; small images skip tiling
           entirely (see tiling.min_size_for_tiling in configs/config.yaml)
"""

import numpy as np


class TileLayout:
    """Stores everything needed to reassemble tiles back into the full image."""

    def __init__(self, original_size, padded_size, tile_size, overlap, positions):
        self.original_size = original_size   # (H, W) before any padding
        self.padded_size = padded_size       # (H, W) after reflect-padding
        self.tile_size = tile_size
        self.overlap = overlap
        self.positions = positions           # list of (y, x) top-left coords in padded image


def _compute_tile_positions(padded_length: int, tile_size: int, stride: int) -> list:
    """1D helper: starting positions of tiles along one axis, guaranteed to
    cover the full padded length, with the last tile flush against the end
    (may overlap its neighbor more than `stride` if it doesn't divide evenly)."""
    if padded_length <= tile_size:
        return [0]
    positions = list(range(0, padded_length - tile_size + 1, stride))
    last_covered_end = positions[-1] + tile_size
    if last_covered_end < padded_length:
        positions.append(padded_length - tile_size)  # flush final tile against the edge
    return positions


def split_into_tiles(image: np.ndarray, tile_size: int, overlap: int):
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")

    h, w = image.shape[:2]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size})")

    # Pad so every tile position has enough room to extract a full tile_size
    # window. reflect padding avoids introducing artificial black borders
    # that the model would otherwise have to "compress" too.
    pad_h = max(0, tile_size - h) if h < tile_size else 0
    pad_w = max(0, tile_size - w) if w < tile_size else 0

    # Also ensure the last tile position + tile_size reaches the image edge
    # cleanly; _compute_tile_positions handles uneven coverage, but we still
    # need at least tile_size total extent to extract from.
    padded_h = max(h + pad_h, tile_size)
    padded_w = max(w + pad_w, tile_size)

    if padded_h > h or padded_w > w:
        image = np.pad(
            image,
            ((0, padded_h - h), (0, padded_w - w), (0, 0)),
            mode="reflect",
        )

    positions_y = _compute_tile_positions(image.shape[0], tile_size, stride)
    positions_x = _compute_tile_positions(image.shape[1], tile_size, stride)

    tiles = []
    positions = []
    for y in positions_y:
        for x in positions_x:
            tiles.append(image[y:y + tile_size, x:x + tile_size].copy())
            positions.append((y, x))

    layout = TileLayout(
        original_size=(h, w),
        padded_size=(image.shape[0], image.shape[1]),
        tile_size=tile_size,
        overlap=overlap,
        positions=positions,
    )
    return tiles, layout


def _axis_weights(positions: list, tile_size: int) -> list:
    """For a sorted list of 1D tile start positions, compute a per-position
    weight vector of length tile_size such that overlapping tiles' weights
    sum to exactly 1 at every world coordinate — including at the true
    boundary tiles, which get NO ramp on their outward-facing edge (there's
    no neighbor there to blend with, so that edge should keep full weight).

    This is what the original uniform-ramp-on-every-edge version got wrong:
    it also ramped down at the image's true outer boundary, where there's
    no neighboring tile to compensate, corrupting the edges.
    """
    n = len(positions)
    weights = []
    for i, p in enumerate(positions):
        w = np.ones(tile_size, dtype=np.float32)

        if i > 0:
            overlap_left = positions[i - 1] + tile_size - p
            if overlap_left > 0:
                fade_in = np.linspace(0, 1, overlap_left, endpoint=False, dtype=np.float32)
                w[:overlap_left] = fade_in

        if i < n - 1:
            overlap_right = p + tile_size - positions[i + 1]
            if overlap_right > 0:
                # Complement of the NEXT tile's fade-in on this same physical
                # region, guaranteeing the two sum to exactly 1 everywhere.
                fade_in_next = np.linspace(0, 1, overlap_right, endpoint=False, dtype=np.float32)
                w[tile_size - overlap_right:] = 1.0 - fade_in_next

        weights.append(w)
    return weights


def merge_tiles(tiles: list, layout: TileLayout) -> np.ndarray:
    if len(tiles) != len(layout.positions):
        raise ValueError(
            f"Got {len(tiles)} tiles but layout expects {len(layout.positions)}"
        )

    padded_h, padded_w = layout.padded_size
    channels = tiles[0].shape[2]
    canvas = np.zeros((padded_h, padded_w, channels), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w, 1), dtype=np.float32)

    # Recover the distinct sorted y/x positions from the grid to compute
    # boundary-aware per-axis weights (see _axis_weights docstring).
    positions_y = sorted(set(p[0] for p in layout.positions))
    positions_x = sorted(set(p[1] for p in layout.positions))
    y_weights = dict(zip(positions_y, _axis_weights(positions_y, layout.tile_size)))
    x_weights = dict(zip(positions_x, _axis_weights(positions_x, layout.tile_size)))

    for tile, (y, x) in zip(tiles, layout.positions):
        mask = np.outer(y_weights[y], x_weights[x])[:, :, None]
        canvas[y:y + layout.tile_size, x:x + layout.tile_size] += tile.astype(np.float32) * mask
        weight_sum[y:y + layout.tile_size, x:x + layout.tile_size] += mask

    weight_sum = np.maximum(weight_sum, 1e-8)  # guard divide-by-zero
    merged = canvas / weight_sum

    h, w = layout.original_size
    return merged[:h, :w]


def adaptive_prepare(image: np.ndarray, tile_size: int, overlap: int,
                      min_size_for_tiling: int):
    """
    Decides whether an image needs tiling at all. Small images are padded
    once and processed whole — no point tiling a 128x128 icon.

    Returns:
        ("whole", padded_image, original_size) or
        ("tiled", tiles, layout)
    """
    h, w = image.shape[:2]
    if max(h, w) < min_size_for_tiling:
        # Pad to a multiple of 8 (the model's stride requirement — see
        # src/encoder.py), not necessarily tile_size, so small images
        # aren't wastefully padded all the way up to a full tile.
        pad_h = (-h) % 8
        pad_w = (-w) % 8
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        return "whole", image, (h, w)

    tiles, layout = split_into_tiles(image, tile_size, overlap)
    return "tiled", tiles, layout


if __name__ == "__main__":
    # Self-test — run as a module from the project root:
    #   python -m src.tiling
    print("Test 1: small image bypasses tiling")
    small = np.random.rand(100, 150, 3).astype(np.float32)
    mode, result, meta = adaptive_prepare(small, tile_size=256, overlap=16,
                                           min_size_for_tiling=512)
    print(f"  mode={mode}, shape={result.shape} (expected multiple of 8)")
    assert mode == "whole"
    assert result.shape[0] % 8 == 0 and result.shape[1] % 8 == 0

    print("\nTest 2: large image gets tiled, round-trip reconstructs exactly")
    large = np.random.rand(1000, 1300, 3).astype(np.float32)
    mode, tiles, layout = adaptive_prepare(large, tile_size=256, overlap=16,
                                            min_size_for_tiling=512)
    print(f"  mode={mode}, num_tiles={len(tiles)}, "
          f"grid positions sample={layout.positions[:3]}")
    assert mode == "tiled"
    for t in tiles:
        assert t.shape == (256, 256, 3), t.shape

    # Round trip with the ORIGINAL (unprocessed) tiles should perfectly
    # reconstruct the original image — this proves split+merge geometry is
    # correct, independent of any model. (In real use, tiles get replaced
    # by the model's reconstructions between split and merge.)
    reconstructed = merge_tiles(tiles, layout)
    print(f"  reconstructed shape={reconstructed.shape}, original shape={large.shape}")
    assert reconstructed.shape == large.shape
    max_error = np.abs(reconstructed - large).max()
    print(f"  max pixel error vs original: {max_error:.6f} (expected ~0)")
    assert max_error < 1e-4, "Tiling geometry is wrong — round trip doesn't match"

    print("\nTest 3: odd, non-tile-size-multiple dimensions")
    odd = np.random.rand(777, 1234, 3).astype(np.float32)
    mode, tiles, layout = adaptive_prepare(odd, tile_size=256, overlap=16,
                                            min_size_for_tiling=512)
    reconstructed = merge_tiles(tiles, layout)
    assert reconstructed.shape == odd.shape
    max_error = np.abs(reconstructed - odd).max()
    print(f"  shape={odd.shape}, num_tiles={len(tiles)}, max pixel error: {max_error:.6f}")
    assert max_error < 1e-4

    print("\nAll tiling self-tests PASSED")