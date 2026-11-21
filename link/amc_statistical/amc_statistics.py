"""Moments, cumulants, cyclostationary features, instantaneous/HWT features, and LB tests."""

from __future__ import annotations

import math
import numpy as np


def _logsumexp_1d(arr: np.ndarray) -> float:
    m = float(np.max(arr))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(arr - m))))


def sample_moment(x: np.ndarray, n: int, q: int) -> complex:
    """
    Nth-order / q-conjugate sample moment:
      m_{x, n, q} = (1 / N) * sum(x^(n - q) * conj(x)^q).
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    if arr.size == 0:
        raise ValueError("Input array cannot be empty.")
    if n < 1 or not (0 <= q <= n):
        raise ValueError("Require n >= 1 and 0 <= q <= n.")
    term = (arr ** (n - q)) * (np.conj(arr) ** q)
    return complex(np.mean(term))


def sample_cumulant_zero_lag(x: np.ndarray, n: int, q: int) -> complex:
    """
    Compute zero-lag sample cumulant c_{x, n, q} for orders n in {1, 2, 4, 6, 8}
    using Leonov-Shiryaev / moment-to-cumulant relations for stationary zero-mean signals.
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    if arr.size == 0:
        raise ValueError("Input array cannot be empty.")
    xc = arr - np.mean(arr)

    m = lambda p, r: sample_moment(xc, p, r)

    if n == 1:
        return complex(np.mean(arr))
    if n == 2:
        return m(2, q)
    if n == 4:
        if q == 0:
            return m(4, 0) - 3.0 * (m(2, 0) ** 2)
        if q == 1:
            return m(4, 1) - 3.0 * m(2, 0) * m(2, 1)
        if q == 2:
            return m(4, 2) - (abs(m(2, 0)) ** 2) - 2.0 * (m(2, 1) ** 2)
        return sample_cumulant_zero_lag(xc, 4, 4 - q)
    if n == 6:
        # Standard symmetric-constellation 6th-order cumulant forms (Dobre et al. Table IV)
        if q in (0, 6):
            return m(6, 0) - 15.0 * m(4, 0) * m(2, 0) + 30.0 * (m(2, 0) ** 3)
        if q in (1, 5):
            return (
                m(6, 1)
                - 3.0 * m(4, 0) * m(2, 1)
                - 12.0 * m(4, 1) * m(2, 0)
                + 30.0 * (m(2, 0) ** 2) * m(2, 1)
            )
        if q in (2, 4):
            return (
                m(6, 2)
                - 6.0 * m(4, 1) * m(2, 1)
                - 9.0 * m(4, 2) * m(2, 0)
                + 18.0 * m(2, 0) * (m(2, 1) ** 2)
                + 12.0 * (abs(m(2, 0)) ** 2) * m(2, 0)
            )
        if q == 3:
            return (
                m(6, 3)
                - 9.0 * m(4, 2) * m(2, 1)
                - 6.0 * m(4, 1) * np.conj(m(2, 0))
                + 12.0 * (m(2, 1) ** 3)
                + 24.0 * m(2, 1) * (abs(m(2, 0)) ** 2)
            )
    if n == 8:
        # 8th-order zero-lag cumulant estimator for symmetric constellations
        if q in (0, 8):
            return m(8, 0) - 28.0 * m(6, 0) * m(2, 0) - 35.0 * (m(4, 0) ** 2) + 420.0 * m(4, 0) * (m(2, 0) ** 2) - 630.0 * (m(2, 0) ** 4)
        if q in (2, 6):
            return m(8, 2) - 15.0 * m(6, 2) * m(2, 0) - 13.0 * m(6, 1) * m(2, 1) - 10.0 * (m(4, 2) ** 2) + 120.0 * m(4, 2) * m(2, 0) * m(2, 1)
        if q == 4:
            return m(8, 4) - 16.0 * m(6, 3) * m(2, 1) - 18.0 * (m(4, 2) ** 2) + 144.0 * m(4, 2) * (m(2, 1) ** 2) - 144.0 * (m(2, 1) ** 4)
        return sample_cumulant_zero_lag(xc, 8, 8 - q)

    raise ValueError(f"Unsupported cumulant order n={n} or conjugate index q={q}.")


def higher_order_cumulants_zero_lag(x: np.ndarray) -> dict[str, complex]:
    """Compute key zero-lag cumulants and phase-invariant normalized cumulants."""
    c20 = sample_cumulant_zero_lag(x, 2, 0)
    c21 = sample_cumulant_zero_lag(x, 2, 1)
    c40 = sample_cumulant_zero_lag(x, 4, 0)
    c41 = sample_cumulant_zero_lag(x, 4, 1)
    c42 = sample_cumulant_zero_lag(x, 4, 2)
    c63 = sample_cumulant_zero_lag(x, 6, 3)
    c80 = sample_cumulant_zero_lag(x, 8, 0)
    c84 = sample_cumulant_zero_lag(x, 8, 4)

    denom2 = (c21 ** 2) if abs(c21) > 1e-15 else (1e-15 + 0j)
    denom3 = (c21 ** 3) if abs(c21) > 1e-15 else (1e-15 + 0j)
    denom4 = (c21 ** 4) if abs(c21) > 1e-15 else (1e-15 + 0j)

    return {
        "c20": c20,
        "c21": c21,
        "c40": c40,
        "c41": c41,
        "c42": c42,
        "c63": c63,
        "c80": c80,
        "c84": c84,
        "c40_norm": c40 / denom2,
        "c42_norm": c42 / denom2,
        "c63_norm": c63 / denom3,
        "c80_norm": c80 / denom4,
        "c84_norm": c84 / denom4,
    }


def cyclic_autocorrelation_scan(
    x: np.ndarray,
    sample_rate: float = 1.0,
    nfft: int = 1024,
) -> dict[str, float]:
    """
    Scan cyclic autocorrelation magnitudes for both squared (x^2) and lag-shifted
    conjugate (x[n]*conj(x[n+tau])) cyclostationary features.
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    fs = max(float(sample_rate), 1e-9)
    pwr = float(np.mean(np.abs(arr) ** 2)) + 1e-15

    if arr.size < 8:
        return {"cyc_sq_peak_norm": 0.0, "cyc_sq_freq_hz": 0.0, "cyc_lag1_peak_norm": 0.0, "cyc_lag1_freq_hz": 0.0}

    fft_len = max(int(nfft), 1 << int(math.ceil(math.log2(arr.size))))
    freqs = np.fft.fftfreq(fft_len, d=1.0 / fs)
    pos = freqs > 0.0
    sub_f = freqs[pos]

    z_sq = (arr ** 2) - np.mean(arr ** 2)
    spec_sq = np.abs(np.fft.fft(z_sq, n=fft_len))[pos] / max(arr.size, 1)
    idx_sq = int(np.argmax(spec_sq))

    z_lag = (arr[:-1] * np.conj(arr[1:])) - np.mean(arr[:-1] * np.conj(arr[1:]))
    spec_lag = np.abs(np.fft.fft(z_lag, n=fft_len))[pos] / max(arr.size - 1, 1)
    idx_lag = int(np.argmax(spec_lag))

    return {
        "cyc_sq_freq_hz": float(sub_f[idx_sq]),
        "cyc_sq_peak_norm": float(spec_sq[idx_sq] / pwr),
        "cyc_lag1_freq_hz": float(sub_f[idx_lag]),
        "cyc_lag1_peak_norm": float(spec_lag[idx_lag] / pwr),
    }


def cyclic_cumulant_magnitude(
    x: np.ndarray,
    cycle_freq_norm: float,
    order: int = 4,
    conj_count: int = 0,
) -> float:
    """
    Estimate magnitude of nth-order / q-conjugate cyclic cumulant at normalized cycle
    frequency alpha in [0, 0.5) and zero lag vector (Dobre et al. Eqs. 33-36).
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    if arr.size < 8:
        return 0.0
    xc = arr - np.mean(arr)
    n_idx = np.arange(xc.size, dtype=float)
    carrier = np.exp(-1j * 2.0 * math.pi * float(cycle_freq_norm) * n_idx)

    m_cyc = lambda p, r: complex(np.mean(((xc ** (p - r)) * (np.conj(xc) ** r)) * carrier))
    m_stat = lambda p, r: sample_moment(xc, p, r)

    if order == 2:
        val = m_cyc(2, conj_count)
    elif order == 4:
        if conj_count == 0:
            val = m_cyc(4, 0) - 3.0 * m_cyc(2, 0) * m_stat(2, 0)
        elif conj_count == 2:
            val = m_cyc(4, 2) - 2.0 * m_cyc(2, 1) * m_stat(2, 1) - m_cyc(2, 0) * np.conj(m_stat(2, 0))
        else:
            val = m_cyc(4, conj_count) - 3.0 * m_cyc(2, 0) * m_stat(2, 1)
    else:
        val = complex(np.mean(((xc ** (order - conj_count)) * (np.conj(xc) ** conj_count)) * carrier))

    pwr = float(np.mean(np.abs(xc) ** 2)) + 1e-15
    return float(abs(val) / (pwr ** (order / 2.0)))


def haar_wavelet_detail_stats(x: np.ndarray) -> dict[str, float]:
    """
    Haar wavelet detail magnitude statistics (Ho et al. [36]-[38], Dobre et al. Sec. 4.1-4.3):
    d[k] = 0.5 * (x[2k+1] - x[2k]). Constant envelope PSK has flat detail away from transitions;
    FSK/QAM exhibit stronger amplitude/frequency step variations.
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    if arr.size < 4:
        return {"hwt_std": 0.0, "hwt_peak_ratio": 0.0}
    even = arr[0 : (arr.size // 2) * 2 : 2]
    odd = arr[1 : (arr.size // 2) * 2 : 2]
    detail_mag = np.abs(0.5 * (odd - even))
    med = float(np.median(detail_mag)) + 1e-15
    return {
        "hwt_std": float(np.std(detail_mag)),
        "hwt_peak_ratio": float(np.max(detail_mag) / med),
    }


def instantaneous_and_transform_features(x: np.ndarray) -> dict[str, float]:
    """Extract instantaneous amplitude, phase, frequency, and Haar wavelet statistics."""
    arr = np.asarray(x, dtype=np.complex128).ravel()
    amp = np.abs(arr)
    mean_amp = float(np.mean(amp)) + 1e-15
    amp_norm = amp / mean_amp
    amp_cent = amp_norm - float(np.mean(amp_norm))
    amp_std = float(np.std(amp_cent))
    amp_kurt = float(np.mean(amp_cent ** 4) / ((amp_std ** 4) + 1e-15))

    phase = np.unwrap(np.angle(arr))
    phase_cent = phase - float(np.mean(phase))
    phase_std = float(np.std(phase_cent))

    inst_freq = np.diff(phase)
    inst_freq_cent = inst_freq - float(np.mean(inst_freq)) if inst_freq.size > 0 else np.array([0.0])
    freq_std = float(np.std(inst_freq_cent))
    abs_freq_cent = np.abs(inst_freq_cent) - float(np.mean(np.abs(inst_freq_cent)))
    abs_freq_var = float(np.var(abs_freq_cent))

    hwt = haar_wavelet_detail_stats(arr)

    return {
        "amp_std": amp_std,
        "amp_kurt": amp_kurt,
        "phase_std": phase_std,
        "freq_std": freq_std,
        "abs_freq_var": abs_freq_var,
        **hwt,
    }


def quasi_alrt_psk_metrics(symbols: np.ndarray) -> dict[str, float]:
    """Polydoros / Quasi-ALRT M-th order phase-blind metrics: v_M = |sum(s_n^M)| / K."""
    s = np.asarray(symbols, dtype=np.complex128).ravel()
    k = max(s.size, 1)
    rms = math.sqrt(float(np.mean(np.abs(s) ** 2))) + 1e-15
    sn = s / rms
    return {
        "v2_psk": float(np.abs(np.sum(sn ** 2)) / k),
        "v4_psk": float(np.abs(np.sum(sn ** 4)) / k),
        "v8_psk": float(np.abs(np.sum(sn ** 8)) / k),
    }


def quasi_alrt_qam_metric(symbols: np.ndarray, a_coeff: float = 0.0135, b_coeff: float = -0.0246) -> float:
    """Quasi-ALRT fourth-order QAM metric (Long et al. [7], Dobre et al. Eq. 23)."""
    s = np.asarray(symbols, dtype=np.complex128).ravel()
    if s.size == 0:
        return 0.0
    rms = math.sqrt(float(np.mean(np.abs(s) ** 2))) + 1e-15
    sn = s / rms
    return float(a_coeff * abs(np.sum(sn ** 4)) + b_coeff * abs(np.sum((np.abs(sn) ** 4))))


def quasi_alrt_fsk_hoc_score(x: np.ndarray, snr_linear: float = 10.0) -> float:
    """
    Higher-order correlation (HOC) Quasi-ALRT score for FSK-like constant-envelope /
    frequency-stepped waveforms (Beidas & Weber [9], Dobre et al. Eq. 25).
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    if arr.size < 8:
        return 0.0
    rms = math.sqrt(float(np.mean(np.abs(arr) ** 2))) + 1e-15
    xn = arr / rms
    c1_0 = float(np.mean(np.abs(xn) ** 2))
    c1_lag1 = complex(np.mean(np.conj(xn[:-1]) * xn[1:]))
    c2_lag1 = complex(np.mean((np.abs(xn[:-1]) ** 2) * (np.abs(xn[1:]) ** 2)))
    snr = max(float(snr_linear), 1e-3)
    score = (
        1.0
        + 0.25 * snr * c1_0
        + 0.125 * snr * (abs(c1_lag1) ** 2)
        + 0.05 * snr * float(np.real(c1_lag1 * np.conj(c2_lag1)))
    )
    return float(score)


def alrt_known_channel_loglik(symbols: np.ndarray, constellation: np.ndarray, snr_linear: float) -> float:
    """ALRT / ML log-likelihood when carrier phase and gain are aligned (Dobre et al. Eq. 10)."""
    y = np.asarray(symbols, dtype=np.complex128).ravel()
    c = np.asarray(constellation, dtype=np.complex128).ravel()
    snr = max(float(snr_linear), 1e-4)
    # y[:, None] - c[None, :] -> (K, M)
    dist2 = np.abs(y[:, None] - c[None, :]) ** 2
    scaled = -snr * dist2
    m_max = np.max(scaled, axis=1, keepdims=True)
    sym_ll = m_max[:, 0] + np.log(np.mean(np.exp(scaled - m_max), axis=1))
    return float(np.sum(sym_ll))


def alrt_phase_marginalized_loglik(
    symbols: np.ndarray,
    constellation: np.ndarray,
    snr_linear: float,
    phase_grid_steps: int = 72,
) -> float:
    """
    ALRT log-likelihood averaging over unknown carrier phase theta ~ Uniform[-pi, pi)
    and equiprobable symbols (Dobre et al. Eq. 13 / Eq. 38).
    """
    y = np.asarray(symbols, dtype=np.complex128).ravel()
    c = np.asarray(constellation, dtype=np.complex128).ravel()
    snr = max(float(snr_linear), 1e-4)
    thetas = np.linspace(-math.pi, math.pi, int(phase_grid_steps), endpoint=False)

    # Compute per-phase log-likelihood across symbols: shape (P, K)
    phase_lls = np.empty((thetas.size, y.size), dtype=float)
    for idx_p, th in enumerate(thetas):
        rot_c = c * np.exp(1j * th)
        dist2 = np.abs(y[:, None] - rot_c[None, :]) ** 2
        scaled = -snr * dist2
        m_max = np.max(scaled, axis=1, keepdims=True)
        phase_lls[idx_p, :] = m_max[:, 0] + np.log(np.mean(np.exp(scaled - m_max), axis=1))

    tot_per_phase = np.sum(phase_lls, axis=1)
    return _logsumexp_1d(tot_per_phase) - math.log(float(thetas.size))


def hlrt_phase_profiled_loglik(
    symbols: np.ndarray,
    constellation: np.ndarray,
    snr_linear: float,
    phase_grid_steps: int = 180,
) -> float:
    """
    Hybrid Likelihood Ratio Test (HLRT) log-likelihood: averages over unknown symbols
    while maximizing over deterministic unknown carrier phase theta (Dobre et al. Eq. 27).
    """
    y = np.asarray(symbols, dtype=np.complex128).ravel()
    c = np.asarray(constellation, dtype=np.complex128).ravel()
    snr = max(float(snr_linear), 1e-4)
    thetas = np.linspace(-math.pi, math.pi, int(phase_grid_steps), endpoint=False)

    best_ll = -math.inf
    for th in thetas:
        rot_c = c * np.exp(1j * th)
        dist2 = np.abs(y[:, None] - rot_c[None, :]) ** 2
        scaled = -snr * dist2
        m_max = np.max(scaled, axis=1, keepdims=True)
        ll = float(np.sum(m_max[:, 0] + np.log(np.mean(np.exp(scaled - m_max), axis=1))))
        if ll > best_ll:
            best_ll = ll
    return best_ll


def glrt_joint_phase_symbol_loglik(
    symbols: np.ndarray,
    constellation: np.ndarray,
    snr_linear: float,
    phase_grid_steps: int = 180,
) -> float:
    """
    Generalized Likelihood Ratio Test (GLRT) log-likelihood: maximizes jointly over
    carrier phase theta and per-symbol constellation point s_k (Dobre et al. Eq. 26).
    """
    y = np.asarray(symbols, dtype=np.complex128).ravel()
    c = np.asarray(constellation, dtype=np.complex128).ravel()
    snr = max(float(snr_linear), 1e-4)
    thetas = np.linspace(-math.pi, math.pi, int(phase_grid_steps), endpoint=False)

    best_ll = -math.inf
    for th in thetas:
        rot_c = c * np.exp(1j * th)
        dist2 = np.abs(y[:, None] - rot_c[None, :]) ** 2
        min_dist2 = np.min(dist2, axis=1)
        ll = float(-snr * np.sum(min_dist2))
        if ll > best_ll:
            best_ll = ll
    return best_ll
