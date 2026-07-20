"""
Saves side-by-side Original | Reconstructed comparison grids during validation.

Used periodically (every --vis_every epochs) so training progress can be
visually inspected, not just tracked via loss/PSNR numbers -- a model can have
a "good" loss number while still producing visually obvious artifacts, and
this catches that.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend -- required for Colab/servers with no display
import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def save_reconstruction_grid(originals, reconstructions, epoch, save_dir, writer=None, n_examples=4):
    """
    Args:
        originals: tensor [B, 3, H, W], values in [0, 1].
        reconstructions: tensor [B, 3, H, W], values in [0, 1] (already on CPU,
            already detached from the graph -- caller's responsibility).
        epoch: current epoch number, used in the filename and TensorBoard step.
        save_dir: directory to save the PNG grid to.
        writer: optional torch.utils.tensorboard.SummaryWriter -- if given, the
            same figure is also logged to TensorBoard under "Reconstructions".
        n_examples: how many image pairs to show in the grid.

    Returns:
        Path to the saved PNG file.
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(n_examples, originals.shape[0])

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i in range(n):
        orig = originals[i].permute(1, 2, 0).cpu().numpy()
        recon = reconstructions[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        axes[i, 0].imshow(orig)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(recon)
        axes[i, 1].axis("off")

    axes[0, 0].set_title("Original", fontsize=12)
    axes[0, 1].set_title("Reconstructed", fontsize=12)
    fig.suptitle(f"Epoch {epoch}", fontsize=14)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=120)

    if writer is not None:
        writer.add_figure("Reconstructions", fig, global_step=epoch)

    plt.close(fig)
    return save_path
