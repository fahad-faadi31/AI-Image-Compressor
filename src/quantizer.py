"""
Straight-Through Estimator (STE) quantization for the compression bottleneck.

Pipeline (encoder side):
    conv_output --Tanh--> y in [-1, 1] --scale by 127--> y_scaled in [-127, 127]
    --STE round--> y_quantized (integer-valued floats, forward = round(y_scaled))

Pipeline (decoder side):
    y_quantized --divide by 127--> y_dequantized (~[-1, 1]) --> feed into decoder

Why round() is applied to y_scaled, not y:
    y (post-Tanh) lives in [-1, 1]. Rounding a value in that tiny range would only
    ever produce -1, 0, or 1 -- three symbols total, nowhere near enough for 8-bit
    quantization (up to 255 distinct levels). We must scale up to the target
    integer range FIRST, then round -- rounding is only meaningful once neighbouring
    integers are spaced far enough apart to matter.

Why scale by 127 (not 128):
    int8 technically spans [-128, 127], but Tanh's output is symmetric around 0.
    Scaling by 127 keeps the quantization grid symmetric too (127 positive levels,
    127 negative levels, plus 0), rather than wasting an asymmetric extra negative
    level that a symmetric activation would never produce values for anyway.

Why no branching is needed between train/eval mode:
    The STE formula `y + (round(y) - y).detach()` has forward value round(y)
    regardless of whether autograd is tracking gradients. At eval/inference time,
    code typically runs under torch.no_grad() anyway, so the .detach() call is a
    no-op in practice (there's no graph to detach from) and the forward numerical
    result -- real hard rounding -- is identical in both modes. nn.Module's
    self.training flag isn't needed here; the same forward() works correctly
    whether or not gradients are being computed.
"""

import torch
import torch.nn as nn


class Quantizer(nn.Module):
    def __init__(self, levels=127):
        """
        Args:
            levels (int): scale factor mapping Tanh's [-1, 1] output to an integer
                range [-levels, levels]. levels=127 corresponds to (near) full use
                of a signed 8-bit representation.
        """
        super().__init__()
        self.levels = levels

    def forward(self, y):
        """
        Args:
            y: tensor already passed through Tanh, values in [-1, 1].

        Returns:
            y_quantized: tensor of the same shape, values are integers (as floats)
                in [-levels, levels]. Forward pass = real rounding.
                Backward pass = identity gradient (STE), so gradients still flow
                into the encoder despite round() being technically non-differentiable.
        """
        y_scaled = y * self.levels

        # Straight-Through Estimator:
        # forward:  y_scaled + (round(y_scaled) - y_scaled) = round(y_scaled)
        # backward: gradient of the detached term is 0, so d(output)/d(y_scaled) = 1
        y_quantized = y_scaled + (torch.round(y_scaled) - y_scaled).detach()

        return y_quantized

    def dequantize(self, y_quantized):
        """Maps quantized integer-valued tensor back to ~[-1, 1] for the decoder."""
        return y_quantized / self.levels
