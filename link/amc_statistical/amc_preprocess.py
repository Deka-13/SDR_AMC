"""Signal detection, blind parameter estimation, CFO/timing sync, and matched filtering."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    signal_present: bool
    estimated_noise_power: float
    estimated_total_power: float
    snr_db: float
    cyclic_peak_freq_hz: float
    cyclic_peak_ratio: float
    reason: str


@dataclass
class PreprocessResult:
    raw_iq: np.ndarray
    conditioned_iq: np.ndarray
    cfo_corrected_iq: np.ndarray
    matched_iq: np.ndarray
    symbols: np.ndarray
    estimated_snr_db: float
    estimated_cfo_norm: float
    estimated_sps: int
    best_timing_phase: int
    detection: DetectionResult


def design_rrc_taps(num_taps: int = 101, alpha: float = 0.35, sps: int = 8) -> np.ndarray:
    """Root-Raised Cosine (RRC) filter normalized to unit L2 energy."""
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    if sps < 1 or num_taps < 3:
        raise ValueError("sps must be >= 1 and num_taps >= 3.")
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.arange(num_taps, dtype=float) - (num_taps - 1) / 2.0
    h = np.zeros(num_taps, dtype=float)
    for i, val in enumerate(t):
        x = val / float(sps)
        if x == 0.0:
            h[i] = 1.0 - alpha + (4.0 * alpha / math.pi)
        elif alpha != 0.0 and math.isclose(abs(x), 1.0 / (4.0 * alpha), rel_tol=0.0, abs_tol=1e-7):
            h[i] = (alpha / math.sqrt(2.0)) * (
                (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * alpha))
                + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * alpha))
            )
        else:
            num = math.sin(math.pi * x * (1.0 - alpha)) + 4.0 * alpha * x * math.cos(math.pi * x * (1.0 + alpha))
            den = math.pi * x * (1.0 - (4.0 * alpha * x) ** 2)
            h[i] = num / den if abs(den) > 1e-15 else 0.0

    norm = float(np.linalg.norm(h))
    return h / norm if norm > 0.0 else h


def estimate_noise_floor_mad(x: np.ndarray) -> float:
    """
    Robust blind noise variance estimator using real/imag median absolute deviation (MAD)
    on first differences (high-frequency noise floor proxy).
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    if arr.size < 8:
        return 1e-9
    d = np.diff(arr)
    # Scale factor for complex Gaussian first difference: Var(d_r) = 2 * sigma_r^2
    mad_r = float(np.median(np.abs(np.real(d) - np.median(np.real(d)))))
    mad_i = float(np.median(np.abs(np.imag(d) - np.median(np.imag(d)))))
    sigma_r = (mad_r / 0.67448975) / math.sqrt(2.0)
    sigma_i = (mad_i / 0.67448975) / math.sqrt(2.0)
    noise_pwr = max((sigma_r ** 2) + (sigma_i ** 2), 1e-12)
    return float(noise_pwr)


def estimate_snr_db_blind(x: np.ndarray) -> float:
    """Blind SNR (Es/N0 or total SNR) estimate in dB from signal power vs MAD noise floor."""
    arr = np.asarray(x, dtype=np.complex128).ravel()
    tot_pwr = float(np.mean(np.abs(arr) ** 2))
    noise_pwr = estimate_noise_floor_mad(arr)
    sig_pwr = max(tot_pwr - noise_pwr, 1e-12)
    snr_lin = max(sig_pwr / noise_pwr, 1e-4)
    return float(10.0 * math.log10(snr_lin))


def detect_signal_energy_and_cyclostationary(
    x: np.ndarray,
    sample_rate: float = 1.0,
    pfa_threshold: float = 2.5,
) -> DetectionResult:
    """
    Energy + cyclostationary feature detector (PySDR Detection & Cyclostationary chapters).
    Determines whether a modulated signal is present above the noise floor.
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    tot_pwr = float(np.mean(np.abs(arr) ** 2))
    noise_pwr = estimate_noise_floor_mad(arr)
    snr_lin = tot_pwr / max(noise_pwr, 1e-12)
    snr_db = float(10.0 * math.log10(max(snr_lin, 1e-4)))

    # Lag-1 conjugate product spectral peak (cyclostationary baud/carrier signature)
    if arr.size >= 16:
        prod = arr[:-1] * np.conj(arr[1:])
        prod = prod - np.mean(prod)
        nfft = 1 << int(math.ceil(math.log2(max(prod.size, 64))))
        mag = np.abs(np.fft.fft(prod, n=nfft))
        freqs = np.fft.fftfreq(nfft, d=1.0 / max(float(sample_rate), 1e-9))
        pos = freqs > 0.0
        if np.any(pos):
            sub_m = mag[pos]
            sub_f = freqs[pos]
            idx = int(np.argmax(sub_m))
            cyc_freq = float(sub_f[idx])
            med_m = float(np.median(sub_m)) + 1e-15
            cyc_ratio = float(sub_m[idx] / med_m)
        else:
            cyc_freq, cyc_ratio = 0.0, 0.0
    else:
        cyc_freq, cyc_ratio = 0.0, 0.0

    present = bool((snr_db >= pfa_threshold) or (cyc_ratio >= 8.0 and snr_db >= 0.0))
    reason = (
        f"SNR={snr_db:.1f} dB (thr={pfa_threshold:.1f} dB), cyclic_peak_ratio={cyc_ratio:.1f}"
    )
    return DetectionResult(
        signal_present=present,
        estimated_noise_power=noise_pwr,
        estimated_total_power=tot_pwr,
        snr_db=snr_db,
        cyclic_peak_freq_hz=cyc_freq,
        cyclic_peak_ratio=cyc_ratio,
        reason=reason,
    )


def estimate_coarse_cfo_normalized(x: np.ndarray, powers: tuple[int, ...] = (2, 4)) -> float:
    """
    Blind normalized CFO estimate (cycles per sample, fs=1) using M-th power
    spectral line search: f_hat = argmax_f |sum(x^M * exp(-j*2*pi*f*M*n))| / M.
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    if arr.size < 16:
        return 0.0

    best_peak = -1.0
    best_cfo = 0.0
    nfft = 1 << int(math.ceil(math.log2(max(arr.size * 2, 1024))))
    n_idx = np.arange(arr.size, dtype=float)

    for m in powers:
        z = arr ** m
        z = z - np.mean(z)
        spec = np.abs(np.fft.fft(z, n=nfft))
        idx = int(np.argmax(spec))
        # FFT bin frequency f_m in cycles/sample for z = x^m corresponds to CFO = f_m / m
        f_m = idx / float(nfft)
        if f_m > 0.5:
            f_m -= 1.0
        cfo_est = f_m / float(m)
        # Score using coherent de-rotation magnitude
        derot = arr * np.exp(-1j * 2.0 * math.pi * cfo_est * n_idx)
        score = float(np.abs(np.mean(derot ** m)))
        if score > best_peak:
            best_peak = score
            best_cfo = float(cfo_est)

    return best_cfo


def estimate_symbol_rate_hz(x: np.ndarray, sample_rate: float = 1.0, min_sps: int = 2, max_sps: int = 32) -> tuple[float, int]:
    """
    Estimate symbol rate (Hz) and integer samples-per-symbol (sps) from the squared
    instantaneous envelope |x|^2 spectrum (baud-line cyclostationarity).
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    fs = max(float(sample_rate), 1e-9)
    if arr.size < 16:
        sps_fb = max(min(8, max_sps), min_sps)
        return fs / sps_fb, sps_fb

    env_sq = np.abs(arr) ** 2
    env_sq = env_sq - np.mean(env_sq)
    nfft = 1 << int(math.ceil(math.log2(max(env_sq.size * 2, 1024))))
    mag = np.abs(np.fft.fft(env_sq, n=nfft))
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)

    f_low = fs / float(max_sps)
    f_high = fs / float(max_sps) * (float(max_sps) / float(min_sps))
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        sps_fb = max(min(8, max_sps), min_sps)
        return fs / sps_fb, sps_fb

    sub_f = freqs[mask]
    sub_m = mag[mask]
    baud_hz = float(sub_f[int(np.argmax(sub_m))])
    sps_est = int(round(fs / max(baud_hz, 1e-9)))
    sps_est = max(min(sps_est, max_sps), min_sps)
    return float(fs / sps_est), sps_est


def synchronize_and_decimate(
    cfo_corrected_iq: np.ndarray,
    h_rrc: np.ndarray,
    sps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Matched filtering + uniform-window peak average energy timing synchronization
    (matching your receiver_synchronization design with safe fallback for short buffers).
    """
    x = np.asarray(cfo_corrected_iq, dtype=np.complex128).ravel()
    h_mf = np.conj(np.asarray(h_rrc, dtype=np.complex128)[::-1])
    filtered = np.convolve(x, h_mf)

    group_delay = max(h_mf.size - 1, 0)
    avail = filtered.size - group_delay
    if avail <= 0:
        return filtered, x, 0

    expected_syms = max(1, avail // sps)
    best_phase = 0
    best_energy = -1.0

    for phase in range(sps):
        idxs = group_delay + phase + np.arange(expected_syms) * sps
        if idxs.size > 0 and idxs[-1] < filtered.size:
            e = float(np.mean(np.abs(filtered[idxs]) ** 2))
            if e > best_energy:
                best_energy = e
                best_phase = phase

    start = group_delay + best_phase
    syms = filtered[start : start + expected_syms * sps : sps]
    if syms.size == 0:
        syms = x[:: max(sps, 1)]
    return filtered, syms, best_phase


def preprocess_iq_stream(
    iq: np.ndarray,
    sample_rate: float = 1.0,
    known_sps: int | None = None,
    alpha: float = 0.35,
    num_taps: int = 101,
    estimate_cfo: bool = True,
) -> PreprocessResult:
    """
    Full signal conditioning pipeline:
      1. DC removal & RMS normalization
      2. Detection (energy + cyclic peak) & blind SNR estimate
      3. Coarse CFO estimation & derotation
      4. Symbol rate / SPS estimation, RRC matched filtering & symbol timing sync
    """
    raw = np.asarray(iq, dtype=np.complex128).ravel()
    if raw.size < 8:
        raise ValueError("Need at least 8 IQ samples for preprocessing.")

    det = detect_signal_energy_and_cyclostationary(raw, sample_rate=sample_rate)

    # 1. Remove DC offset and normalize unit RMS power
    centered = raw - np.mean(raw)
    rms = math.sqrt(float(np.mean(np.abs(centered) ** 2)))
    cond = centered / rms if rms > 1e-15 else centered

    # 2. Coarse CFO estimation & correction
    cfo_norm = estimate_coarse_cfo_normalized(cond) if estimate_cfo else 0.0
    n_idx = np.arange(cond.size, dtype=float)
    cfo_corr = cond * np.exp(-1j * 2.0 * math.pi * cfo_norm * n_idx)

    # 3. SPS & matched filtering + timing synchronization
    if known_sps is not None:
        if isinstance(known_sps, bool) or not isinstance(known_sps, (int, np.integer)) or int(known_sps) < 1:
            raise ValueError("known_sps must be an integer >= 1.")
        sps_val = int(known_sps)
    else:
        _, sps_val = estimate_symbol_rate_hz(cfo_corr, sample_rate=sample_rate)

    taps_eff = min(num_taps, max(3, cond.size if cond.size % 2 == 1 else cond.size - 1))
    h_rrc = design_rrc_taps(num_taps=taps_eff, alpha=alpha, sps=sps_val)
    matched, syms, best_phase = synchronize_and_decimate(cfo_corr, h_rrc, sps=sps_val)

    return PreprocessResult(
        raw_iq=raw,
        conditioned_iq=cond,
        cfo_corrected_iq=cfo_corr,
        matched_iq=matched,
        symbols=syms,
        estimated_snr_db=det.snr_db,
        estimated_cfo_norm=cfo_norm,
        estimated_sps=sps_val,
        best_timing_phase=best_phase,
        detection=det,
    )
