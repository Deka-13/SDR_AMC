import numpy as np

def _validate_bits(bits: np.ndarray) -> np.ndarray:
    """Validate a non-empty, one-dimensional binary sequence."""
    bits = np.asarray(bits)

    if bits.ndim != 1:
        raise ValueError("Input must be a one-dimensional array.")

    if bits.size == 0:
        raise ValueError("Input must not be empty.")

    if not np.issubdtype(bits.dtype, np.number) or np.issubdtype(
        bits.dtype, np.complexfloating
    ):
        raise ValueError("Input must contain only bits 0 and 1.")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Input must contain only bits 0 and 1.")

    return bits.astype(np.uint8)


def _validate_symbols(symbols: np.ndarray) -> np.ndarray:
    """Validate a non-empty array of finite numeric symbols."""
    symbols = np.asarray(symbols)

    if symbols.ndim != 1:
        raise ValueError("Symbols must be a one-dimensional array.")

    if symbols.size == 0:
        raise ValueError("Symbols must not be empty.")

    if not np.issubdtype(symbols.dtype, np.number):
        raise ValueError("Symbols must be numeric.")

    if not np.all(np.isfinite(symbols)):
        raise ValueError("Symbols must contain only finite values.")

    return symbols


