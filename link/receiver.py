import numpy as np


def _is_valid_int(val) -> bool:
    """Helper to strictly validate integer types while rejecting booleans and floats."""
    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


def receiver_synchronization(
    received_signal: np.ndarray,
    h_rrc: np.ndarray,
    expected_num_symbols: int,
    samples_per_symbol: int = 8,
) -> np.ndarray:
    """
    Performs receiver processing with active integer-sample symbol timing synchronization:
    1. Exhaustive input validation (1D signal check, finite checks, floating/complex dtypes, robust norm filter energy, integer bounds).
    2. General Matched Filtering using the conjugate time-reversed impulse response: h_MF[n] = h*[-n].
    3. Active Integer Symbol Timing Synchronization: Estimates the optimal integer sampling phase 
       (0 to samples_per_symbol - 1) by evaluating average signal energy across a uniform observation window 
       of exactly expected_num_symbols for each candidate phase, eliminating sample-count or trailing transient bias.
       
    METHODOLOGY & THESIS LIMITATIONS:
    - This estimator uses an average peak energy heuristic ($\arg\max_\tau E_{\text{avg}}(\tau)$). 
    - CRITICAL THEORETICAL CAVEAT: Maximum average output energy does not guarantee minimum inter-symbol 
      interference (ISI). Because energy depends on the transmitted data sequence, pulse shape, and noise, 
      this estimator can exhibit data-pattern jitter.
    - Group delay calculations assume matched transmit and receive filters of length L, yielding a 
      combined nominal group delay of L - 1 in the absence of additional channel propagation delay.
    - This function estimates the intra-symbol timing phase ($\tau \in \{0, 1, \ldots, N_s-1\}$), 
      not absolute frame/packet boundary synchronization or multi-symbol propagation delay.
    """
    # Validate received signal
    received_signal = np.asarray(received_signal, dtype=np.complex128)
    if received_signal.size == 0:
        raise ValueError("Received signal cannot be empty.")
    if received_signal.ndim != 1:
        raise ValueError("received_signal must be one-dimensional.")
    if not np.all(np.isfinite(received_signal)):
        raise ValueError("received_signal must contain only finite values (no NaNs or infinities).")

    # Validate expected symbol count
    if not _is_valid_int(expected_num_symbols) or expected_num_symbols < 1:
        raise ValueError("expected_num_symbols must be a positive integer.")

    # Validate filter taps (accepting standard and extended NumPy floating/complex-floating types)
    h_rrc = np.asarray(h_rrc)
    if h_rrc.ndim != 1 or h_rrc.size == 0:
        raise ValueError("h_rrc must be a non-empty one-dimensional filter.")
    
    if not np.issubdtype(h_rrc.dtype, np.floating) and not np.issubdtype(h_rrc.dtype, np.complexfloating):
        raise ValueError("h_rrc must have a floating-point or complex-floating data type.")

    if not np.all(np.isfinite(h_rrc)):
        raise ValueError("h_rrc must contain only finite values.")
    
    # Robust norm-based energy check
    filter_energy = np.linalg.norm(h_rrc)
    if not np.isfinite(filter_energy) or filter_energy == 0.0:
        raise ValueError("h_rrc must have finite, non-zero energy.")

    # Validate samples per symbol
    if not _is_valid_int(samples_per_symbol) or samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be a positive integer.")

    # 1. Matched Filter using general conjugate time-reversed impulse response: h_MF[n] = h*[-n]
    h_mf = np.conj(h_rrc[::-1])
    filtered_signal = np.convolve(received_signal, h_mf)

    # 2. Active Symbol Timing Synchronization (Uniform-Window Peak Average Energy Search)
    group_delay = h_mf.size - 1
    best_offset = 0
    max_avg_energy = -1.0

    for phase in range(samples_per_symbol):
        # Evaluate every phase over an equivalent observation interval of expected_num_symbols
        candidate_indices = (
            group_delay
            + phase
            + np.arange(expected_num_symbols) * samples_per_symbol
        )
        
        if candidate_indices[-1] >= filtered_signal.size:
            raise ValueError(
                f"Insufficient samples in the received waveform to evaluate "
                f"expected_num_symbols={expected_num_symbols} at phase offset {phase}."
            )

        avg_energy = np.mean(np.abs(filtered_signal[candidate_indices]) ** 2)
        if avg_energy > max_avg_energy:
            max_avg_energy = avg_energy
            best_offset = phase

    # 3. Decimation at the optimal estimated timing phase, slicing to exact expected length
    optimal_start_idx = group_delay + best_offset
    decimated_symbols = filtered_signal[
        optimal_start_idx : optimal_start_idx + expected_num_symbols * samples_per_symbol : samples_per_symbol
    ]
    
    return decimated_symbols
