import numpy as np


def _is_valid_int(val) -> bool:
    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


def apply_multipath_fading(
    signal: np.ndarray,
    path_delays: list[int] | None = None,
    path_gains: list[complex] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.size == 0:
        raise ValueError("Input signal cannot be empty.")

    if rng is None:
        rng = np.random.default_rng()

    if path_delays is None:
        path_delays = [0, 2, 5]
    
    if len(path_delays) == 0:
        raise ValueError("At least one multipath delay must be provided.")
    if not all(_is_valid_int(d) and d >= 0 for d in path_delays):
        raise ValueError("All path delays must be non-negative integers.")
    
    if path_gains is not None:
        path_gains = np.asarray(path_gains, dtype=np.complex128)
        if path_gains.size != len(path_delays):
            raise ValueError("path_delays and path_gains must have equal lengths.")
        if not np.all(np.isfinite(path_gains)):
            raise ValueError("Path gains must be finite complex numbers.")

    max_delay = max(path_delays)
    channel_impulse = np.zeros(max_delay + 1, dtype=np.complex128)

    if path_gains is None:
        num_paths = len(path_delays)
        raw_amps = rng.rayleigh(scale=1.0, size=num_paths)
        raw_amps = raw_amps / np.sqrt(np.sum(raw_amps ** 2))
        phases = rng.uniform(0.0, 2.0 * np.pi, size=num_paths)
        path_gains = raw_amps * np.exp(1j * phases)

    for delay, gain in zip(path_delays, path_gains):
        channel_impulse[delay] += gain

    return np.convolve(signal, channel_impulse)


def apply_timing_offset(
    signal: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.size == 0:
        raise ValueError("Input signal cannot be empty.")
    if not np.isfinite(epsilon) or not (0.0 <= float(epsilon) < 1.0):
        raise ValueError("epsilon must be finite and in the range [0, 1).")
    if epsilon == 0.0:
        return signal.copy()
    
    padded = np.pad(signal, (1, 1), mode="constant", constant_values=(0.0, 0.0))
    n = np.arange(signal.size, dtype=float)
    shifted_indices = n + 1.0 - epsilon
    
    return (
        np.interp(shifted_indices, np.arange(padded.size), padded.real)
        + 1j * np.interp(shifted_indices, np.arange(padded.size), padded.imag)
    )


def add_awgn_to_signal(
    signal: np.ndarray,
    ref_power: float,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not np.isfinite(ref_power) or ref_power <= 0:
        raise ValueError("ref_power must be finite and positive.")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")
    
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = ref_power / snr_linear

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(signal.size) + 1j * rng.standard_normal(signal.size)
    )
    return signal + noise


def add_phase_jitter(
    signal: np.ndarray,
    jitter_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if jitter_std <= 0:
        return signal
    phi = rng.normal(0.0, jitter_std, size=signal.size)
    return signal * np.exp(1j * phi)


def add_carrier_frequency_offset(
    signal: np.ndarray,
    cfo: float,
    samples_per_symbol: int,
) -> np.ndarray:
    if cfo == 0.0:
        return signal
    n = np.arange(signal.size, dtype=float) / float(samples_per_symbol)
    return signal * np.exp(1j * 2.0 * np.pi * cfo * n)


def add_carrier_phase(signal: np.ndarray, theta: float) -> np.ndarray:
    if theta == 0.0:
        return signal
    return signal * np.exp(1j * theta)


def add_interference(
    signal: np.ndarray,
    ref_power: float,
    sir_db: float,
    interference_type: str = "tone",
    tone_freq: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if not np.isfinite(ref_power) or ref_power <= 0:
        raise ValueError("ref_power must be finite and positive.")
    if not np.isfinite(sir_db):
        raise ValueError("sir_db must be finite.")
    if rng is None:
        rng = np.random.default_rng()

    sir_linear = 10.0 ** (sir_db / 10.0)
    interference_power = ref_power / sir_linear

    n = np.arange(signal.size, dtype=float)
    if interference_type == "tone":
        interference = np.sqrt(interference_power) * np.exp(
            1j * (2.0 * np.pi * tone_freq * n + rng.uniform(0.0, 2.0 * np.pi))
        )
    elif interference_type == "noise":
        interference = np.sqrt(interference_power / 2.0) * (
            rng.standard_normal(signal.size) + 1j * rng.standard_normal(signal.size)
        )
    else:
        raise ValueError("Unsupported interference_type.")
    return signal + interference
