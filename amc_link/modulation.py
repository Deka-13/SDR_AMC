import numpy as np
from amc_link.validation import _validate_bits, _validate_symbols

def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    """Maps bits 0 -> +1 and 1 -> -1."""
    bits = _validate_bits(bits)
    return (1.0 - 2.0 * bits.astype(float)).astype(np.complex128)


def bpsk_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision BPSK demodulation."""
    symbols = _validate_symbols(symbols)
    return (symbols.real < 0).astype(np.uint8)


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    """Gray-coded QPSK; bits are grouped as I bit, Q bit."""
    bits = _validate_bits(bits)

    if bits.size % 2 != 0:
        raise ValueError("QPSK requires an even number of bits.")

    pairs = bits.reshape(-1, 2).astype(float)
    i = 1.0 - 2.0 * pairs[:, 0]
    q = 1.0 - 2.0 * pairs[:, 1]

    return (i + 1j * q) / np.sqrt(2.0)


def qpsk_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision QPSK demodulation."""
    symbols = _validate_symbols(symbols)

    bits = np.empty(2 * symbols.size, dtype=np.uint8)
    bits[0::2] = (symbols.real < 0).astype(np.uint8)
    bits[1::2] = (symbols.imag < 0).astype(np.uint8)

    return bits
