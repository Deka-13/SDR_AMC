import numpy as np
from link.validation import _validate_symbols


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


def add_phase_jitter(
    symbols: np.ndarray,
    jitter_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Adds per-symbol phase jitter (phi_k) modeled as zero-mean Gaussian distributed phase shifts,
    in accordance with the signal model in Section 2[cite: 14].
    """
    symbols = _validate_symbols(symbols).astype(np.complex128)
    if jitter_std <= 0:
        return symbols

    if rng is None:
        rng = np.random.default_rng()

    phi_k = rng.normal(0.0, jitter_std, size=symbols.size)
    return symbols * np.exp(1j * phi_k)


def add_carrier_frequency_offset(
    symbols: np.ndarray,
    cfo: float,
    samples_per_symbol: int = 1,
) -> np.ndarray:
    """
    Adds Carrier Frequency Offset (CFO, Delta f) term exp(j * 2 * pi * Delta_f * t).
    """
    symbols = _validate_symbols(symbols).astype(np.complex128)
    if cfo == 0.0:
        return symbols

    n = np.arange(symbols.size, dtype=float) / float(samples_per_symbol)
    cfo_vector = np.exp(1j * 2.0 * np.pi * cfo * n)
    return symbols * cfo_vector


def add_carrier_phase(
    symbols: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Adds a time-invariant carrier phase offset (Theta)."""
    symbols = _validate_symbols(symbols).astype(np.complex128)
    if theta == 0.0:
        return symbols
    return symbols * np.exp(1j * theta)


def add_interference(
    symbols: np.ndarray,
    sir_db: float,
    interference_type: str = "tone",
    tone_freq: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Adds cochannel interference or jamming signals to model the aggregate
    interference component present in n(t)[cite: 15].
    """
    symbols = _validate_symbols(symbols).astype(np.complex128)
    if not np.isfinite(sir_db):
        raise ValueError("sir_db must be finite.")

    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(symbols) ** 2)
    if signal_power <= 0:
        raise ValueError("Signal must have positive average power.")

    sir_linear = 10.0 ** (sir_db / 10.0)
    interference_power = signal_power / sir_linear

    n = np.arange(symbols.size, dtype=float)
    if interference_type == "tone":
        # Single-tone co-channel jammer / interference
        interference = np.sqrt(interference_power) * np.exp(
            1j * (2.0 * np.pi * tone_freq * n + rng.uniform(0.0, 2.0 * np.pi))
        )
    elif interference_type == "noise":
        # Bandlimited Gaussian co-channel interference
        interference = np.sqrt(interference_power / 2.0) * (
            rng.standard_normal(symbols.size) + 1j * rng.standard_normal(symbols.size)
        )
    else:
        raise ValueError("Unsupported interference_type. Choose 'tone' or 'noise'.")

    return symbols + interference


def apply_complete_channel(
    symbols: np.ndarray,
    snr_db: float,
    sir_db: float | None = None,
    jitter_std: float = 0.0,
    cfo: float = 0.0,
    theta: float = 0.0,
    interference_type: str = "tone",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Applies the comprehensive channel impairment chain:
    Phase Jitter -> CFO & Carrier Phase -> Interference/Jamming -> AWGN.
    Based on the unified signal model (Section 2, IEE_COM0176)[cite: 14, 15].
    """
    if rng is None:
        rng = np.random.default_rng()

    out = _validate_symbols(symbols)

    # 1. Per-symbol phase jitter (phi_k)
    if jitter_std > 0:
        out = add_phase_jitter(out, jitter_std, rng=rng)

    # 2. Carrier frequency offset and phase offset (Delta f, Theta)
    if cfo != 0.0:
        out = add_carrier_frequency_offset(out, cfo)
    if theta != 0.0:
        out = add_carrier_phase(out, theta)

    # 3. Cochannel interference and jamming (part of aggregate channel/noise model)
    if sir_db is not None:
        out = add_interference(out, sir_db, interference_type=interference_type, rng=rng)

    # 4. Receiver Additive White Gaussian Noise (AWGN)
    out = add_awgn(out, snr_db, rng=rng)

    return out
