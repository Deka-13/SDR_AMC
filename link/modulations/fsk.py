import numpy as np
from link.validation import _validate_bits, _validate_symbols

def bits_to_bfsk(bits: np.ndarray, h: float = 0.5, samples_per_symbol: int = 8) -> np.ndarray:
    """
    Binary Frequency Shift Keying (BFSK) continuous-phase modulator.
    Returns complex baseband samples: exp(j * theta(t)).
    """
    bits = _validate_bits(bits)
    data = 1.0 - 2.0 * bits.astype(float)
    
    expanded = np.repeat(data, samples_per_symbol)
    phase_increment = np.pi * h * expanded / samples_per_symbol
    phase = np.cumsum(phase_increment)
    
    return np.exp(1j * phase).astype(np.complex128)

def bfsk_to_bits(
    symbols: np.ndarray,
    samples_per_symbol: int = 8,
    h: float = 0.5
) -> np.ndarray:
    """Hard-decision continuous-phase BFSK demodulation."""

    symbols = _validate_symbols(symbols)

    if (
        not isinstance(samples_per_symbol, (int, np.integer))
        or samples_per_symbol < 2
    ):
        raise ValueError("samples_per_symbol must be an integer >= 2")

    if not np.isfinite(h) or not 0 < h < samples_per_symbol:
        raise ValueError("h must satisfy 0 < h < samples_per_symbol")

    if symbols.size % samples_per_symbol != 0:
        raise ValueError("Incomplete BFSK symbol block")

    num_symbols = symbols.size // samples_per_symbol

    if num_symbols == 0:
        return np.zeros(0, dtype=np.uint8)

    reshaped = symbols.reshape(
        num_symbols, samples_per_symbol
    )

    # Adjacent-sample phase differences.
    phase_products = (
        reshaped[:, 1:] * np.conj(reshaped[:, :-1])
    )

    # Average the phase differences.
    estimated_phase = np.angle(
        np.sum(phase_products, axis=1)
    )

    # Positive frequency = bit 0.
    # Negative frequency = bit 1.
    return (estimated_phase < 0).astype(np.uint8)



def bits_to_fsk4(bits: np.ndarray, h: float = 0.5, samples_per_symbol: int = 8) -> np.ndarray:
    """
    Gray-coded 4-FSK continuous-phase modulator.
    Groups bits in pairs, maps to 4 frequency tones.
    """
    bits = _validate_bits(bits)
    if bits.size % 2 != 0:
        raise ValueError("4-FSK requires a multiple of 2 bits.")
    
    groups = bits.reshape(-1, 2)
    binary = 2 * groups[:, 0].astype(np.int64) + groups[:, 1].astype(np.int64)
    gray = binary ^ (binary >> 1)
    
    # Map Gray code {0, 1, 2, 3} to normalized frequency levels: {-1.5, -0.5, 0.5, 1.5}
    levels = -1.5 + gray.astype(float)
    
    expanded = np.repeat(levels, samples_per_symbol)
    phase_increment = (np.pi * h / 1.5) * expanded / samples_per_symbol
    phase = np.cumsum(phase_increment)
    
    return np.exp(1j * phase).astype(np.complex128)



def fsk4_to_bits(
    symbols: np.ndarray,
    h: float = 0.5,
    samples_per_symbol: int = 8
) -> np.ndarray:
    """Hard-decision Gray-coded 4-FSK demodulation."""

    symbols = _validate_symbols(symbols)

    if samples_per_symbol < 2:
        raise ValueError("samples_per_symbol must be at least 2")

    if h <= 0:
        raise ValueError("h must be positive")

    num_symbols = symbols.size // samples_per_symbol

    if num_symbols == 0:
        return np.zeros(0, dtype=np.uint8)

    # Divide received signal into symbol blocks.
    trimmed = symbols[:num_symbols * samples_per_symbol]
    reshaped = trimmed.reshape(
        num_symbols, samples_per_symbol
    )

    # Calculate phase differences between adjacent samples.
    phase_products = (
        reshaped[:, 1:] * np.conj(reshaped[:, :-1])
    )

    # Estimate average phase increment for each symbol.
    estimated_phase = np.angle(
        np.sum(phase_products, axis=1)
    )

    # Reconstruct the four expected phase increments.
    gray_indices = np.arange(4)
    levels = -1.5 + gray_indices

    expected_phase = (
        np.pi * h * levels
        / (1.5 * samples_per_symbol)
    )

    # Calculate circular phase distance to each expected tone.
    phase_errors = np.angle(
        np.exp(
            1j * (
                estimated_phase[:, None]
                - expected_phase[None, :]
            )
        )
    )

    # Select the closest frequency tone.
    detected_gray = np.argmin(
        np.abs(phase_errors),
        axis=1
    )

    # Convert detected Gray indices back to binary indices.
    binary = detected_gray ^ (detected_gray >> 1)

    # Recover original MSB-first bit pairs.
    bits = np.empty((num_symbols, 2), dtype=np.uint8)

    bits[:, 0] = (binary >> 1) & 1
    bits[:, 1] = binary & 1

    return bits.reshape(-1)

