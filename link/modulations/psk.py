import numpy as np
from link.validation import _validate_bits, _validate_symbols

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


def _bits_to_psk_generic(bits: np.ndarray, M: int) -> np.ndarray:
    """Generic Gray-coded M-PSK modulator."""
    bits = _validate_bits(bits)
    k = int(np.log2(M))

    if bits.size % k != 0:
        raise ValueError(f"{M}-PSK requires a multiple of {k} bits.")

    groups = bits.reshape(-1, k)
    binary = np.zeros(groups.shape[0], dtype=np.int64)
    for j in range(k):
        binary += groups[:, j].astype(np.int64) * (1 << (k - 1 - j))

    gray = binary ^ (binary >> 1)
    phase = 2.0 * np.pi * gray.astype(float) / float(M)

    return np.cos(phase) + 1j * np.sin(phase)


def _psk_to_bits_generic(symbols: np.ndarray, M: int) -> np.ndarray:
    """Generic hard-decision Gray-coded M-PSK demodulator."""
    symbols = _validate_symbols(symbols)
    k = int(np.log2(M))

    phase = np.mod(np.angle(symbols), 2.0 * np.pi)
    gray = np.floor(
        phase * float(M) / (2.0 * np.pi) + 0.5
    ).astype(np.int64) % M

    binary = gray.copy()
    mask = binary >> 1
    while np.any(mask > 0):
        binary ^= mask
        mask >>= 1

    bits = np.empty(k * symbols.size, dtype=np.uint8)
    for j in range(k):
        bits[j::k] = (binary >> (k - 1 - j)) & 1

    return bits


def bits_to_psk8(bits: np.ndarray) -> np.ndarray:
    """Gray-coded 8-PSK."""
    return _bits_to_psk_generic(bits, 8)


def psk8_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision Gray-coded 8-PSK demodulation."""
    return _psk_to_bits_generic(symbols, 8)


def bits_to_psk16(bits: np.ndarray) -> np.ndarray:
    """Gray-coded 16-PSK (added for family completeness)."""
    return _bits_to_psk_generic(bits, 16)


def psk16_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision Gray-coded 16-PSK demodulation."""
    return _psk_to_bits_generic(symbols, 16)


def bits_to_psk32(bits: np.ndarray) -> np.ndarray:
    """Gray-coded 32-PSK (added for family completeness)."""
    return _bits_to_psk_generic(bits, 32)


def psk32_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision Gray-coded 32-PSK demodulation."""
    return _psk_to_bits_generic(symbols, 32)
