import numpy as np
from link.validation import _validate_bits, _validate_symbols

def bits_to_qam16(bits: np.ndarray) -> np.ndarray:
    """Gray-coded square 16-QAM."""
    bits = _validate_bits(bits)

    if bits.size % 4 != 0:
        raise ValueError("16-QAM requires a multiple of 4 bits.")

    groups = bits.reshape(-1, 4)

    i = (1.0 - 2.0 * groups[:, 0]) * (3.0 - 2.0 * groups[:, 1])
    q = (1.0 - 2.0 * groups[:, 2]) * (3.0 - 2.0 * groups[:, 3])

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
        second = np.where(np.abs(levels) == 1, 1, 0).astype(np.uint8)
        return np.column_stack((first, second))

    i_bits = levels_to_bits(i_level)
    q_bits = levels_to_bits(q_level)

    bits = np.empty(4 * symbols.size, dtype=np.uint8)
    bits[0::4] = i_bits[:, 0]
    bits[1::4] = i_bits[:, 1]
    bits[2::4] = q_bits[:, 0]
    bits[3::4] = q_bits[:, 1]

    return bits


def bits_to_qam64(bits: np.ndarray) -> np.ndarray:
    """Gray-coded square 64-QAM."""
    bits = _validate_bits(bits)

    if bits.size % 6 != 0:
        raise ValueError("64-QAM requires a multiple of 6 bits.")

    groups = bits.reshape(-1, 6)

    def bits_to_level(level_bits: np.ndarray) -> np.ndarray:
        sign = 1.0 - 2.0 * level_bits[:, 0]
        gray = (
            2 * level_bits[:, 1].astype(np.int64)
            + level_bits[:, 2].astype(np.int64)
        )
        binary = gray.copy()
        binary ^= binary >> 1
        magnitude = 7.0 - 2.0 * binary
        return sign * magnitude

    i = bits_to_level(groups[:, :3])
    q = bits_to_level(groups[:, 3:])

    return (i + 1j * q) / np.sqrt(42.0)


def qam64_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision Gray-coded square 64-QAM demodulation."""
    symbols = _validate_symbols(symbols)

    i = symbols.real * np.sqrt(42.0)
    q = symbols.imag * np.sqrt(42.0)

    def level_to_bits(values: np.ndarray) -> np.ndarray:
        level_index = np.clip(
            np.floor((values + 8.0) / 2.0), 0, 7
        ).astype(np.int64)

        sign_bit = (level_index < 4).astype(np.uint8)
        magnitude_index = np.where(
            level_index < 4, level_index, 7 - level_index
        )
        gray = magnitude_index ^ (magnitude_index >> 1)

        bits = np.empty((values.size, 3), dtype=np.uint8)
        bits[:, 0] = sign_bit
        bits[:, 1] = (gray >> 1) & 1
        bits[:, 2] = gray & 1

        return bits

    i_bits = level_to_bits(i)
    q_bits = level_to_bits(q)

    bits = np.empty(6 * symbols.size, dtype=np.uint8)
    bits[0::6] = i_bits[:, 0]
    bits[1::6] = i_bits[:, 1]
    bits[2::6] = i_bits[:, 2]
    bits[3::6] = q_bits[:, 0]
    bits[4::6] = q_bits[:, 1]
    bits[5::6] = q_bits[:, 2]

    return bits


def bits_to_qam256(bits: np.ndarray) -> np.ndarray:
    """Gray-coded square 256-QAM (added for family completeness)."""
    bits = _validate_bits(bits)

    if bits.size % 8 != 0:
        raise ValueError("256-QAM requires a multiple of 8 bits.")

    groups = bits.reshape(-1, 8)

    def bits_to_level(level_bits: np.ndarray) -> np.ndarray:
        sign = 1.0 - 2.0 * level_bits[:, 0]
        # 3 bits of Gray code for magnitude: levels are 1, 3, 5, 7, 9, 11, 13, 15
        gray = (
            4 * level_bits[:, 1].astype(np.int64)
            + 2 * level_bits[:, 2].astype(np.int64)
            + level_bits[:, 3].astype(np.int64)
        )
        binary = gray.copy()
        binary ^= binary >> 2
        binary ^= binary >> 1
        magnitude = 15.0 - 2.0 * binary
        return sign * magnitude

    i = bits_to_level(groups[:, :4])
    q = bits_to_level(groups[:, 4:])

    # Average energy normalization factor for 256-QAM is sqrt(170)
    return (i + 1j * q) / np.sqrt(170.0)



def qam256_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision demodulation for the matching Gray-coded 256-QAM encoder."""

    symbols = _validate_symbols(symbols)

    # Undo transmitter energy normalization.
    i = symbols.real * np.sqrt(170.0)
    q = symbols.imag * np.sqrt(170.0)

    def pam16_level_to_bits(values: np.ndarray) -> np.ndarray:
        # Recover sign bit: 0 = positive, 1 = negative.
        sign = (values < 0).astype(np.uint8)

        # Recover binary magnitude index (0 through 7).
        # Encoder magnitude = 15 - 2 * binary_index.
        magnitude_binary = np.rint(
            (15.0 - np.abs(values)) / 2.0
        ).astype(np.int64)

        magnitude_binary = np.clip(
            magnitude_binary, 0, 7
        )

        # Convert binary magnitude index back to Gray code.
        gray = magnitude_binary ^ (magnitude_binary >> 1)

        bits = np.empty((values.size, 4), dtype=np.uint8)

        bits[:, 0] = sign
        bits[:, 1] = (gray >> 2) & 1
        bits[:, 2] = (gray >> 1) & 1
        bits[:, 3] = gray & 1

        return bits

    i_bits = pam16_level_to_bits(i)
    q_bits = pam16_level_to_bits(q)

    # Restore original I-then-Q bit ordering.
    bits = np.empty((symbols.size, 8), dtype=np.uint8)

    bits[:, :4] = i_bits
    bits[:, 4:] = q_bits

    return bits.reshape(-1)

