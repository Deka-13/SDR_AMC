import numpy as np
from link.validation import _validate_symbols


def _is_valid_int(val) -> bool:
    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


def design_rrc_filter(
    num_taps: int = 101,
    alpha: float = 0.35,
    samples_per_symbol: int = 8,
) -> np.ndarray:
    if not isinstance(alpha, (float, int)) or not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be a float between 0 and 1.")
    if not _is_valid_int(samples_per_symbol) or samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be an integer >= 1.")
    if not _is_valid_int(num_taps) or num_taps < 3:
        raise ValueError("num_taps must be an integer >= 3.")
    
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.arange(num_taps) - (num_taps - 1) / 2.0
    h_rrc = np.zeros(num_taps, dtype=float)

    for i, val in enumerate(t):
        x = val / float(samples_per_symbol)
        if x == 0.0:
            h_rrc[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif alpha != 0 and np.isclose(np.abs(x), 1.0 / (4.0 * alpha), atol=1e-7):
            h_rrc[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            numerator = np.sin(np.pi * x * (1.0 - alpha)) + 4.0 * alpha * x * np.cos(np.pi * x * (1.0 + alpha))
            denominator = np.pi * x * (1.0 - (4.0 * alpha * x) ** 2)
            h_rrc[i] = numerator / denominator

    h_rrc = h_rrc / np.sqrt(np.sum(h_rrc ** 2))
    return h_rrc


def apply_pulse_shaping(
    symbols: np.ndarray,
    num_taps: int = 101,
    alpha: float = 0.35,
    samples_per_symbol: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    symbols = _validate_symbols(symbols).astype(np.complex128)
    h_rrc = design_rrc_filter(num_taps=num_taps, alpha=alpha, samples_per_symbol=samples_per_symbol)
    
    upsampled = np.zeros(symbols.size * samples_per_symbol, dtype=np.complex128)
    upsampled[::samples_per_symbol] = symbols

    shaped_signal = np.convolve(upsampled, h_rrc)
    return shaped_signal, h_rrc
