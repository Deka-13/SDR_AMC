import numpy as np
from link.validation import _validate_symbols
from link.pulse_shaping import apply_pulse_shaping
from link.channel import (
    apply_multipath_fading,
    apply_timing_offset,
    add_awgn_to_signal,
    add_phase_jitter,
    add_carrier_frequency_offset,
    add_carrier_phase,
    add_interference,
)
from link.receiver import receiver_synchronization


def simulate_complete_link(
    symbols: np.ndarray,
    snr_db: float,
    sir_db: float | None = None,
    jitter_std: float = 0.0,
    cfo: float = 0.0,
    theta: float = 0.0,
    epsilon: float = 0.0,
    use_multipath: bool = False,
    path_delays: list[int] | None = None,
    path_gains: list[complex] | None = None,
    interference_type: str = "tone",
    alpha: float = 0.35,
    num_taps: int = 101,
    samples_per_symbol: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Executes the full pipeline with strict size checking, synchronization, and reference power scaling."""
    if rng is None:
        rng = np.random.default_rng()

    out = _validate_symbols(symbols)

    # --- TRANSMITTER SIDE ---
    shaped_signal, h_rrc = apply_pulse_shaping(
        out, num_taps=num_taps, alpha=alpha, samples_per_symbol=samples_per_symbol
    )

    ref_power = np.mean(np.abs(shaped_signal) ** 2)
    if not np.isfinite(ref_power) or ref_power <= 0:
        raise ValueError("Signal reference power must be finite and positive.")

    # --- CHANNEL ENVIRONMENT ---
    if use_multipath:
        shaped_signal = apply_multipath_fading(
            shaped_signal, path_delays=path_delays, path_gains=path_gains, rng=rng
        )

    if epsilon != 0.0:
        shaped_signal = apply_timing_offset(shaped_signal, epsilon)

    if jitter_std > 0:
        shaped_signal = add_phase_jitter(shaped_signal, jitter_std, rng=rng)

    if cfo != 0.0:
        shaped_signal = add_carrier_frequency_offset(shaped_signal, cfo, samples_per_symbol=samples_per_symbol)
    
    if theta != 0.0:
        shaped_signal = add_carrier_phase(shaped_signal, theta)

    if sir_db is not None:
        shaped_signal = add_interference(
            shaped_signal, ref_power=ref_power, sir_db=sir_db, 
            interference_type=interference_type, rng=rng
        )

    shaped_signal = add_awgn_to_signal(
        shaped_signal, ref_power=ref_power, snr_db=snr_db, rng=rng
    )

    # --- RECEIVER SIDE ---
    recovered_symbols = receiver_synchronization(
        shaped_signal, h_rrc, samples_per_symbol=samples_per_symbol
    )

    if recovered_symbols.size < symbols.size:
        raise RuntimeError(
            f"Receiver produced fewer symbols ({recovered_symbols.size}) "
            f"than were transmitted ({symbols.size}). Check synchronization/filtering lengths."
        )

    return recovered_symbols[:symbols.size]
