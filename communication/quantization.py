# communication/quantization.py — N-bit uniform quantization of model parameters

import numpy as np
from config import QUANTIZATION_BITS


def quantize(params, bits=QUANTIZATION_BITS):
    """
    Uniform linear quantization of a list of numpy arrays.
    Reduces each array to `bits`-bit integer representation.
    Returns: (quantized_params, scale_factors, zero_points)
    """
    quantized = []
    scales    = []
    zeros     = []

    levels = 2 ** bits - 1      # e.g. 255 for 8-bit

    for p in params:
        p_min = p.min()
        p_max = p.max()

        # Avoid zero-range collapse
        p_range = p_max - p_min if p_max != p_min else 1e-8

        scale = p_range / levels
        zero  = p_min

        q = np.round((p - zero) / scale).astype(np.int32)
        q = np.clip(q, 0, levels)

        quantized.append(q)
        scales.append(scale)
        zeros.append(zero)

    return quantized, scales, zeros


def dequantize(quantized_params, scales, zeros):
    """Reconstruct float32 arrays from quantized representation."""
    return [
        q.astype(np.float32) * s + z
        for q, s, z in zip(quantized_params, scales, zeros)
    ]


def quantize_and_restore(params, bits=QUANTIZATION_BITS):
    """Full round-trip: quantize → dequantize. Returns compressed params."""
    q, scales, zeros = quantize(params, bits)
    restored = dequantize(q, scales, zeros)
    return restored


def compute_compression_ratio(original_params, bits=QUANTIZATION_BITS):
    """Return compression ratio relative to float32 (32 bits)."""
    return 32.0 / bits
