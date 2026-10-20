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

def bits_to_psk8(bits: np.ndarray) -> np.ndarray:
    """Gray-coded 8-PSK; bits are grouped MSB first."""
    bits = _validate_bits(bits)

    if bits.size % 3 != 0:
        raise ValueError("8-PSK requires a multiple of 3 bits.")

    groups = bits.reshape(-1, 3)

    binary = (
        4 * groups[:, 0]
        + 2 * groups[:, 1]
        + groups[:, 2]
    )

    gray = binary ^ (binary >> 1)

    phase = 2.0 * np.pi * gray / 8.0

    return np.cos(phase) + 1j * np.sin(phase)


def psk8_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision Gray-coded 8-PSK demodulation."""
    symbols = _validate_symbols(symbols)

    phase = np.mod(np.angle(symbols), 2.0 * np.pi)
    gray = np.floor(
        phase * 8.0 / (2.0 * np.pi) + 0.5
    ).astype(np.int64) % 8

    binary = gray.copy()
    binary ^= binary >> 1
    binary ^= binary >> 2

    bits = np.empty(3 * symbols.size, dtype=np.uint8)
    bits[0::3] = (binary >> 2) & 1
    bits[1::3] = (binary >> 1) & 1
    bits[2::3] = binary & 1

    return bits


def bits_to_qam16(bits: np.ndarray) -> np.ndarray:
    """Gray-coded square 16-QAM; bits are grouped I MSB, I LSB, Q MSB, Q LSB."""
    bits = _validate_bits(bits)

    if bits.size % 4 != 0:
        raise ValueError("16-QAM requires a multiple of 4 bits.")

    groups = bits.reshape(-1, 4)

    i = (1.0 - 2.0 * groups[:, 0]) * (
        3.0 - 2.0 * groups[:, 1]
    )

    q = (1.0 - 2.0 * groups[:, 2]) * (
        3.0 - 2.0 * groups[:, 3]
    )

    return (i + 1j * q) / np.sqrt(10.0)


def qam16_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision Gray-coded square 16-QAM demodulation."""
    symbols = _validate_symbols(symbols)

    i = symbols.real * np.sqrt(10.0)
    q = symbols.imag * np.sqrt(10.0)

    i_level = np.where(i < -2.0, -3,
               np.where(i < 0.0, -1,
               np.where(i < 2.0, 1, 3)))

    q_level = np.where(q < -2.0, -3,
               np.where(q < 0.0, -1,
               np.where(q < 2.0, 1, 3)))

    def levels_to_bits(levels: np.ndarray) -> np.ndarray:
        first = (levels < 0).astype(np.uint8)
        second = np.where(
            np.abs(levels) == 1, 1, 0
        ).astype(np.uint8)

        return np.column_stack((first, second))

    i_bits = levels_to_bits(i_level)
    q_bits = levels_to_bits(q_level)

    bits = np.empty(4 * symbols.size, dtype=np.uint8)
    bits[0::4] = i_bits[:, 0]
    bits[1::4] = i_bits[:, 1]
    bits[2::4] = q_bits[:, 0]
    bits[3::4] = q_bits[:, 1]

    return bits
