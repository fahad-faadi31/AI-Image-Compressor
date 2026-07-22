"""
Entropy coding for quantized latent codes.

Design decision: uses Python's built-in zlib/lzma (general-purpose,
battle-tested compressors) rather than a hand-written arithmetic/range
coder. A custom arithmetic coder could theoretically hit the exact
Shannon entropy bound measured by scripts/measure_entropy.py, but a subtle
bug in hand-rolled arithmetic coding is exactly the kind of error that
silently corrupts data rather than crashing loudly -- a serious risk for
a production image compression service. zlib/lzma are extensively tested,
used in production everywhere, and still capture a real, substantial
portion of the available redundancy via their internal Huffman coding
stage, which does adapt to the actual byte frequency distribution.

Contract:
    encode_codes(codes: np.ndarray) -> bytes
    decode_codes(data: bytes, shape: tuple, dtype) -> np.ndarray
"""

import heapq
import json
import lzma
import zlib
from collections import Counter

import numpy as np


def encode_codes(codes: np.ndarray, method: str = "lzma") -> bytes:
    """Compress integer latent codes to bytes.

    codes: array of small integers (0..255 for 8-bit quantization).
    method: "lzma" (better ratio, slower) or "zlib" (faster, still good).
    """
    raw_bytes = codes.astype(np.uint8).tobytes()
    if method == "lzma":
        return lzma.compress(raw_bytes, preset=6)
    elif method == "zlib":
        return zlib.compress(raw_bytes, level=9)
    else:
        raise ValueError(f"Unknown method: {method}")


def decode_codes(data: bytes, shape: tuple, method: str = "lzma") -> np.ndarray:
    """Inverse of encode_codes -- reconstructs the exact integer code array."""
    if method == "lzma":
        raw_bytes = lzma.decompress(data)
    elif method == "zlib":
        raw_bytes = zlib.decompress(data)
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape).copy()


class HuffmanTable:
    """A static Huffman code built from a fixed symbol frequency table.

    Static (not per-image adaptive) is deliberate: the model always
    produces roughly the same code distribution (see
    scripts/measure_entropy.py), so we build the table ONCE from a large
    validation sample and ship it as part of the model artifact -- this
    avoids transmitting a frequency table with every single compressed
    image, which would eat into the savings for small images.
    """

    def __init__(self, codes_by_symbol: dict):
        self.codes_by_symbol = codes_by_symbol  # {symbol: bitstring}
        self.symbol_by_code = {v: k for k, v in codes_by_symbol.items()}

    @classmethod
    def from_counts(cls, counts: dict, num_symbols: int = 256):
        # Laplace smoothing: every symbol needs count >= 1, or the Huffman
        # tree can't build a valid code for it (and an unseen value in a
        # future image would be unencodable otherwise).
        freqs = {s: counts.get(s, 0) + 1 for s in range(num_symbols)}

        heap = [[freq, [symbol, ""]] for symbol, freq in freqs.items()]
        heapq.heapify(heap)

        if len(heap) == 1:
            # Degenerate case: only one distinct symbol possible at all
            only = heap[0][1][0]
            return cls({only: "0"})

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = "0" + pair[1]
            for pair in hi[1:]:
                pair[1] = "1" + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        codes_by_symbol = {symbol: code for symbol, code in heap[0][1:]}
        return cls(codes_by_symbol)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.codes_by_symbol, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            raw = json.load(f)
        # JSON keys are always strings; symbols need to be ints again
        codes_by_symbol = {int(k): v for k, v in raw.items()}
        return cls(codes_by_symbol)


def huffman_encode(codes: np.ndarray, table: HuffmanTable) -> bytes:
    flat = codes.astype(np.uint8).flatten()
    bitstring = "".join(table.codes_by_symbol[int(s)] for s in flat)
    # Pad to a whole number of bytes; store the pad amount in the first byte
    pad = (8 - len(bitstring) % 8) % 8
    bitstring = bitstring + "0" * pad
    raw = bytearray([pad])
    for i in range(0, len(bitstring), 8):
        raw.append(int(bitstring[i:i + 8], 2))
    return bytes(raw)


def huffman_decode(data: bytes, shape: tuple, table: HuffmanTable) -> np.ndarray:
    pad = data[0]
    bitstring = "".join(f"{byte:08b}" for byte in data[1:])
    if pad:
        bitstring = bitstring[:-pad]

    symbols = []
    current = ""
    num_symbols = int(np.prod(shape))
    for bit in bitstring:
        current += bit
        if current in table.symbol_by_code:
            symbols.append(table.symbol_by_code[current])
            current = ""
            if len(symbols) == num_symbols:
                break

    return np.array(symbols, dtype=np.uint8).reshape(shape)


if __name__ == "__main__":
    def compute_entropy_of(arr: np.ndarray) -> float:
        counts = Counter(arr.flatten().tolist())
        total = arr.size
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        return entropy

    # Self-test — run as a module from the project root:
    #   python -m src.entropy
    # Simulates a symbol distribution matching what scripts/measure_entropy.py
    # actually measured on the trained model (clustered around 127-128,
    # ~4.56 bits/symbol entropy), to get an honest empirical comparison
    # against the theoretical bound -- not just a round-trip correctness check.

    print("Test 1: round-trip correctness (random data)")
    rng = np.random.default_rng(42)
    random_codes = rng.integers(0, 256, size=(32, 32, 32), dtype=np.uint8)
    for method in ["zlib", "lzma"]:
        encoded = encode_codes(random_codes, method=method)
        decoded = decode_codes(encoded, random_codes.shape, method=method)
        assert np.array_equal(random_codes, decoded), f"{method} round-trip FAILED"
    print("  round-trip exact match: PASSED (both zlib and lzma)")

    print("\nTest 2: realistic distribution (Gaussian-like, matching measured "
          "entropy of ~4.56 bits/symbol from your trained model)")
    # Approximate the measured distribution: values clustered around 127-128
    # with std roughly tuned to produce ~4.5-4.6 bits of entropy over 256 levels
    gaussian_codes = np.clip(
        rng.normal(loc=127.5, scale=5.5, size=(32, 32, 32)), 0, 255
    ).astype(np.uint8)

    original_bits = gaussian_codes.size * 8
    print(f"  Original (raw 8-bit): {original_bits/8:.0f} bytes "
          f"({original_bits} bits, {8.0:.3f} bits/symbol)")

    for method in ["zlib", "lzma"]:
        encoded = encode_codes(gaussian_codes, method=method)
        decoded = decode_codes(encoded, gaussian_codes.shape, method=method)
        assert np.array_equal(gaussian_codes, decoded), f"{method} round-trip FAILED"

        compressed_bits = len(encoded) * 8
        bits_per_symbol = compressed_bits / gaussian_codes.size
        ratio_vs_raw = original_bits / compressed_bits
        print(f"  [{method}] compressed: {len(encoded)} bytes, "
              f"{bits_per_symbol:.3f} bits/symbol, "
              f"{ratio_vs_raw:.2f}x smaller than raw 8-bit codes")

    print("\nTest 3: Huffman coding, built from this exact distribution's "
          "measured frequencies (this is what scripts/build_huffman_table.py "
          "will do against your real model's validation-set codes)")
    counts = Counter(gaussian_codes.flatten().tolist())
    table = HuffmanTable.from_counts(dict(counts), num_symbols=256)

    huff_encoded = huffman_encode(gaussian_codes, table)
    huff_decoded = huffman_decode(huff_encoded, gaussian_codes.shape, table)
    assert np.array_equal(gaussian_codes, huff_decoded), "Huffman round-trip FAILED"

    huff_bits_per_symbol = len(huff_encoded) * 8 / gaussian_codes.size
    huff_ratio = original_bits / (len(huff_encoded) * 8)
    print(f"  [huffman] compressed: {len(huff_encoded)} bytes, "
          f"{huff_bits_per_symbol:.3f} bits/symbol, "
          f"{huff_ratio:.2f}x smaller than raw 8-bit codes")
    print(f"  (theoretical entropy bound for this data: "
          f"{compute_entropy_of(gaussian_codes):.3f} bits/symbol)")

    print("\nAll entropy coding self-tests PASSED")