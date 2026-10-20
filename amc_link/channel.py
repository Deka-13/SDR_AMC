import numpy as np
from amc_link.validation import _validate_symbols


def add_awgn(
    symbols: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add complex AWGN at a specified average symbol SNR."""

    symbols = _validate_symbols(symbols).astype(np.complex128)

    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")

    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(symbols) ** 2)

    if signal_power <= 0:
        raise ValueError("Signal must have positive average power.")

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(symbols.size)
        + 1j * rng.standard_normal(symbols.size)
    )

    return symbols + noise
