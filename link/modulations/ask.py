import numpy as np
from link.validation import _validate_bits, _validate_symbols

def bits_to_ask2(bits: np.ndarray) -> np.ndarray:
    """Binary Amplitude Shift Keying (2-ASK / On-Off Keying): maps 0 -> 0.0, 1 -> 1.0."""
    bits = _validate_bits(bits)
    return bits.astype(np.complex128)


def ask2_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision 2-ASK (OOK) demodulation."""
    symbols = _validate_symbols(symbols)
    return (symbols.real >= 0.5).astype(np.uint8)


def bits_to_ask4(bits: np.ndarray) -> np.ndarray:
    """Gray-coded 4-ASK (Amplitude Shift Keying)."""
    bits = _validate_bits(bits)
    if bits.size % 2 != 0:
        raise ValueError("4-ASK requires a multiple of 2 bits.")
    
    groups = bits.reshape(-1, 2)
    binary = 2 * groups[:, 0].astype(np.int64) + groups[:, 1].astype(np.int64)
    gray = binary ^ (binary >> 1)
    
    # Levels for 4-ASK normalized: -3, -1, +1, +3 scaled by 1/sqrt(5)
    levels = -3.0 + 2.0 * gray.astype(float)
    return (levels + 0j) / np.sqrt(5.0)


def ask4_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision Gray-coded 4-ASK demodulation."""
    symbols = _validate_symbols(symbols)
    val = symbols.real * np.sqrt(5.0)
    
    level_index = np.clip(np.floor((val + 4.0) / 2.0), 0, 3).astype(np.int64)
    gray = level_index ^ (level_index >> 1)
    
    bits = np.empty(2 * symbols.size, dtype=np.uint8)
    bits[0::2] = (gray >> 1) & 1
    bits[1::2] = gray & 1
    return bits
