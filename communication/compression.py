# communication/compression.py
# ─────────────────────────────────────────────────────────────────
# RESEARCH NOVELTY: Adaptive Hybrid Compression for Federated Edge
# Improved for stability + better accuracy trade-off
# ─────────────────────────────────────────────────────────────────

import numpy as np
from communication.quantization   import quantize, dequantize
from communication.sparsification import sparsify, desparsify
from config import (
    COMPRESSION_METHOD,
    QUANTIZATION_BITS,
    SPARSIFICATION_RATIO,
)

# 🔧 UPDATED: More stable threshold
_LARGE_LAYER_THRESHOLD = 2000   # Increased from 1000


def compress(params, method=COMPRESSION_METHOD):
    if method == "quantization":
        q_params, scales, zeros = quantize(params, bits=QUANTIZATION_BITS)
        return {
            "method": "quantization",
            "data":   q_params,
            "meta":   {"scales": scales, "zeros": zeros},
        }

    elif method == "sparsification":
        # 🔧 Reduced aggressiveness
        ratio = max(SPARSIFICATION_RATIO, 0.4)
        sp_params, masks = sparsify(params, ratio=ratio)
        return {
            "method": "sparsification",
            "data":   sp_params,
            "meta":   {"masks": masks},
        }

    elif method == "combined":
        return _adaptive_compress(params)

    else:
        return {"method": "none", "data": params, "meta": {}}


def decompress(compressed_data):
    method = compressed_data["method"]
    data   = compressed_data["data"]
    meta   = compressed_data["meta"]

    if method == "quantization":
        return dequantize(data, meta["scales"], meta["zeros"])

    elif method == "sparsification":
        return desparsify(data, meta["masks"])

    elif method == "combined":
        return _adaptive_decompress(data, meta)

    else:
        return data


# ── Adaptive Compression ─────────────────────────────────────────

def _adaptive_compress(params):
    strategies = []

    q_data, q_scales, q_zeros = [], [], []
    sp_data, sp_masks         = [], []
    raw_data                  = []

    for p in params:
        size = p.size

        if size < 100:
            # Bias layers → no compression
            strategies.append("raw")
            raw_data.append(p.copy())

        elif size >= _LARGE_LAYER_THRESHOLD:
            # Large layers → quantization
            strategies.append("quant")
            q, s, z = quantize([p], bits=QUANTIZATION_BITS)
            q_data.append(q[0])
            q_scales.append(s[0])
            q_zeros.append(z[0])

        else:
            # 🔧 Medium layers → LESS aggressive sparsification
            strategies.append("sparse")
            ratio = max(SPARSIFICATION_RATIO, 0.4)
            sp, msk = sparsify([p], ratio=ratio)
            sp_data.append(sp[0])
            sp_masks.append(msk[0])

    return {
        "method": "combined",
        "data": {
            "q_data":   q_data,
            "sp_data":  sp_data,
            "raw_data": raw_data,
        },
        "meta": {
            "strategies": strategies,
            "q_scales":   q_scales,
            "q_zeros":    q_zeros,
            "sp_masks":   sp_masks,   # ✅ FIXED (important)
        },
    }


def _adaptive_decompress(data, meta):
    strategies = meta["strategies"]
    q_scales   = meta["q_scales"]
    q_zeros    = meta["q_zeros"]
    sp_masks   = meta["sp_masks"]

    q_idx = sp_idx = raw_idx = 0
    result = []

    for strat in strategies:
        if strat == "raw":
            result.append(data["raw_data"][raw_idx].astype(np.float32))
            raw_idx += 1

        elif strat == "quant":
            r = dequantize(
                [data["q_data"][q_idx]],
                [q_scales[q_idx]],
                [q_zeros[q_idx]],
            )
            result.append(r[0])
            q_idx += 1

        elif strat == "sparse":
            # 🔧 FIXED: use actual mask instead of (data != 0)
            r = desparsify(
                [data["sp_data"][sp_idx]],
                [sp_masks[sp_idx]],
            )
            result.append(r[0])
            sp_idx += 1

    return result


# ── Communication Cost Estimation ───────────────────────────────

def estimate_bytes(params, compressed_data):
    original_bytes = sum(p.nbytes for p in params)

    method = compressed_data["method"]

    if method == "quantization":
        bits = QUANTIZATION_BITS
        compressed_bytes = sum(
            p.size * bits // 8 for p in compressed_data["data"]
        )

    elif method == "sparsification":
        compressed_bytes = sum(
            int(np.sum(m)) * 4 for m in compressed_data["meta"]["masks"]
        )

    elif method == "combined":
        q_bytes = sum(
            a.size * QUANTIZATION_BITS // 8
            for a in compressed_data["data"]["q_data"]
        )

        sp_bytes = sum(
            int(np.sum(m)) * 4
            for m in compressed_data["meta"]["sp_masks"]
        )

        raw_bytes = sum(
            a.nbytes for a in compressed_data["data"]["raw_data"]
        )

        compressed_bytes = q_bytes + sp_bytes + raw_bytes

    else:
        compressed_bytes = original_bytes

    ratio = original_bytes / max(compressed_bytes, 1)
    return original_bytes, compressed_bytes, ratio
