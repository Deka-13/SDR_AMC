"""Multi-statistic modulation classifier with CLI for files, SigMF, simulations, and Pluto SDR."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .amc_io import ArrayIQSource, FileIQSource, PlutoSDRSource, detect_clipping_and_saturation
from .amc_preprocess import PreprocessResult, preprocess_iq_stream
from .amc_statistics import (
    alrt_phase_marginalized_loglik,
    cyclic_autocorrelation_scan,
    cyclic_cumulant_magnitude,
    higher_order_cumulants_zero_lag,
    hlrt_phase_profiled_loglik,
    instantaneous_and_transform_features,
    quasi_alrt_fsk_hoc_score,
    quasi_alrt_psk_metrics,
    quasi_alrt_qam_metric,
)


@dataclass
class ClassificationResult:
    predicted_modulation: str
    confidence: float
    scores: dict[str, float]
    estimated_snr_db: float
    estimated_cfo_norm: float
    estimated_sps: int
    signal_detected: bool
    warnings: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)


def build_default_constellations() -> dict[str, np.ndarray]:
    """Build unit-energy (E[|s|^2] = 1) reference constellations for PSK, QAM, and ASK."""
    qpsk_scale = 1.0 / math.sqrt(2.0)
    qam16_levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=float) / math.sqrt(10.0)
    qam64_levels = np.arange(-7, 8, 2, dtype=float) / math.sqrt(42.0)
    ask4_levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=float) / math.sqrt(5.0)

    return {
        "BPSK": np.array([-1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
        "QPSK": np.array(
            [
                complex(qpsk_scale, qpsk_scale),
                complex(-qpsk_scale, qpsk_scale),
                complex(-qpsk_scale, -qpsk_scale),
                complex(qpsk_scale, -qpsk_scale),
            ],
            dtype=np.complex128,
        ),
        "8-PSK": np.exp(1j * 2.0 * math.pi * np.arange(8) / 8.0).astype(np.complex128),
        "16-QAM": np.array([complex(re, im) for re in qam16_levels for im in qam16_levels], dtype=np.complex128),
        "64-QAM": np.array([complex(re, im) for re in qam64_levels for im in qam64_levels], dtype=np.complex128),
        "4-ASK": ask4_levels.astype(np.complex128),
    }


class StatisticalModulationClassifier:
    """
    Multi-statistic classifier combining:
      1. Signal presence & cyclostationary / baud detection
      2. Zero-lag moments & normalized cumulants (2nd, 4th, 6th, 8th orders)
      3. Cyclic autocorrelation & cyclic cumulants
      4. Instantaneous amplitude/phase/frequency & Haar wavelet features
      5. HLRT / ALRT / Quasi-ALRT likelihood metrics
    """

    # Reference c_{4,0} / c_{2,1}^2 targets from Dobre et al. Table IV (unit-variance constellations)
    CUMULANT_C40_TARGETS: dict[str, complex] = {
        "BPSK": -2.0 + 0.0j,
        "QPSK": 1.0 + 0.0j,
        "8-PSK": 0.0 + 0.0j,
        "16-QAM": -0.68 + 0.0j,
        "64-QAM": -0.619 + 0.0j,
        "4-ASK": -1.36 + 0.0j,
    }

    def __init__(
        self,
        constellations: dict[str, np.ndarray] | None = None,
        include_fsk: bool = True,
        include_noise_class: bool = True,
    ) -> None:
        self.constellations = build_default_constellations() if constellations is None else dict(constellations)
        self.include_fsk = bool(include_fsk)
        self.include_noise_class = bool(include_noise_class)

    def classify_preprocessed(
        self,
        prep: PreprocessResult,
        sample_rate: float = 1.0,
        snr_override_db: float | None = None,
    ) -> ClassificationResult:
        warnings: list[str] = detect_clipping_and_saturation(prep.raw_iq)

        snr_db = float(prep.estimated_snr_db if snr_override_db is None else snr_override_db)
        snr_lin = max(min(10.0 ** (snr_db / 10.0), 1000.0), 0.05)

        syms = prep.symbols if prep.symbols.size >= 8 else prep.cfo_corrected_iq
        # Normalize symbol power to unit average power for scale-invariant likelihoods
        sym_pwr = float(np.mean(np.abs(syms) ** 2))
        syms_norm = syms / math.sqrt(sym_pwr) if sym_pwr > 1e-15 else syms

        hos = higher_order_cumulants_zero_lag(syms_norm)
        inst = instantaneous_and_transform_features(prep.cfo_corrected_iq)
        cyc = cyclic_autocorrelation_scan(prep.conditioned_iq, sample_rate=sample_rate)
        cyc_freq_norm = (cyc["cyc_lag1_freq_hz"] / max(float(sample_rate), 1e-9)) % 0.5
        cc40_mag = cyclic_cumulant_magnitude(prep.conditioned_iq, cyc_freq_norm, order=4, conj_count=0)
        cc42_mag = cyclic_cumulant_magnitude(prep.conditioned_iq, cyc_freq_norm, order=4, conj_count=2)

        psk_q = quasi_alrt_psk_metrics(syms_norm)
        qam_q = quasi_alrt_qam_metric(syms_norm)
        fsk_hoc = quasi_alrt_fsk_hoc_score(prep.cfo_corrected_iq, snr_linear=snr_lin)

        c40_norm = hos["c40_norm"]
        c42_norm = hos["c42_norm"]
        c63_norm = hos["c63_norm"]

        # Weight likelihood higher as SNR increases; weight cumulant/cyclostationary higher at low SNR or with CFO/fading
        w_lik = max(0.25, min(1.35, (snr_db + 8.0) / 14.0))
        w_cum = max(0.35, min(1.25, 1.35 - 0.03 * max(snr_db, -5.0)))

        raw_scores: dict[str, float] = {}
        k_len = max(syms_norm.size, 1)

        for mod_name, const in self.constellations.items():
            hlrt_ll = hlrt_phase_profiled_loglik(syms_norm, const, snr_linear=snr_lin, phase_grid_steps=90) / k_len
            alrt_ll = alrt_phase_marginalized_loglik(syms_norm, const, snr_linear=snr_lin, phase_grid_steps=48) / k_len
            lik_term = 0.7 * hlrt_ll + 0.3 * alrt_ll

            target_c40 = self.CUMULANT_C40_TARGETS.get(mod_name, 0.0 + 0.0j)
            cum_dist = abs(c40_norm - target_c40) ** 2

            # Constellations with constant amplitude (BPSK, QPSK, 8-PSK) have low amplitude variance
            amp_pen = 0.0
            if mod_name in ("BPSK", "QPSK", "8-PSK"):
                amp_pen = 3.5 * (inst["amp_std"] ** 2)
            elif mod_name in ("16-QAM", "64-QAM", "4-ASK"):
                # Multi-amplitude constellations have non-zero amplitude variation
                amp_pen = 2.0 * max(0.0, 0.14 - inst["amp_std"]) ** 2

            # PSK phase-blind moments (v2, v4, v8) distinguish order
            order_bonus = 0.0
            if mod_name == "BPSK":
                order_bonus = 0.9 * psk_q["v2_psk"] - 0.4 * inst["amp_std"]
            elif mod_name == "QPSK":
                order_bonus = 0.7 * psk_q["v4_psk"] - 0.5 * psk_q["v2_psk"]
            elif mod_name == "8-PSK":
                order_bonus = 0.6 * psk_q["v8_psk"] - 0.4 * psk_q["v4_psk"]
            elif mod_name == "4-ASK":
                # 4-ASK is real-valued with strong amplitude spread and near-zero imaginary part
                imag_ratio = float(np.mean(np.abs(np.imag(syms_norm))) / (np.mean(np.abs(np.real(syms_norm))) + 1e-12))
                order_bonus = -1.2 * imag_ratio + 0.8 * inst["amp_std"]

            raw_scores[mod_name] = float(w_lik * lik_term - w_cum * cum_dist - amp_pen + order_bonus)

        if self.include_fsk:
            # FSK waveforms show frequency variation & constant/near-constant envelope with HOC response
            fsk_score = (
                0.35 * fsk_hoc
                + 1.8 * inst["freq_std"]
                + 1.2 * inst["abs_freq_var"]
                + 0.8 * inst["hwt_std"]
                - 3.0 * (inst["amp_std"] ** 2)
                - 0.6
            )
            raw_scores["M-FSK"] = float(fsk_score)

        if self.include_noise_class:
            # Noise / unmodulated baseline score when SNR is very low and no cyclic tone is detected
            noise_score = -0.35 * (snr_db - 1.0) - 0.4 * cyc["cyc_sq_peak_norm"] - 0.4 * cyc["cyc_lag1_peak_norm"]
            if not prep.detection.signal_present:
                noise_score += 1.5
            raw_scores["NOISE_OR_UW"] = float(noise_score)

        keys = list(raw_scores.keys())
        vals = np.array([raw_scores[k] for k in keys], dtype=float)
        shifted = vals - float(np.max(vals))
        exp_vals = np.exp(shifted)
        probs = exp_vals / float(np.sum(exp_vals))

        best_idx = int(np.argmax(probs))
        best_mod = keys[best_idx]
        conf = float(probs[best_idx])

        feat_summary = {
            "c40_real": float(np.real(c40_norm)),
            "c40_imag": float(np.imag(c40_norm)),
            "c40_abs": float(abs(c40_norm)),
            "c42_real": float(np.real(c42_norm)),
            "c63_abs": float(abs(c63_norm)),
            "cc40_mag": float(cc40_mag),
            "cc42_mag": float(cc42_mag),
            "v2_psk": float(psk_q["v2_psk"]),
            "v4_psk": float(psk_q["v4_psk"]),
            "v8_psk": float(psk_q["v8_psk"]),
            "qam_quasi_alrt": float(qam_q),
            "fsk_hoc": float(fsk_hoc),
            "amp_std": float(inst["amp_std"]),
            "amp_kurt": float(inst["amp_kurt"]),
            "phase_std": float(inst["phase_std"]),
            "freq_std": float(inst["freq_std"]),
            "hwt_std": float(inst["hwt_std"]),
            "cyc_sq_peak_norm": float(cyc["cyc_sq_peak_norm"]),
            "cyc_lag1_peak_norm": float(cyc["cyc_lag1_peak_norm"]),
        }

        return ClassificationResult(
            predicted_modulation=best_mod,
            confidence=conf,
            scores={k: float(p) for k, p in zip(keys, probs)},
            estimated_snr_db=snr_db,
            estimated_cfo_norm=float(prep.estimated_cfo_norm),
            estimated_sps=int(prep.estimated_sps),
            signal_detected=bool(prep.detection.signal_present),
            warnings=warnings,
            features=feat_summary,
        )

    def classify_iq(
        self,
        iq_samples: np.ndarray,
        sample_rate: float = 1.0,
        known_sps: int | None = None,
        snr_override_db: float | None = None,
        estimate_cfo: bool = True,
    ) -> ClassificationResult:
        prep = preprocess_iq_stream(
            iq=iq_samples,
            sample_rate=sample_rate,
            known_sps=known_sps,
            estimate_cfo=estimate_cfo,
        )
        return self.classify_preprocessed(prep=prep, sample_rate=sample_rate, snr_override_db=snr_override_db)


def _synthesize_demo_iq(mod: str, num_symbols: int = 400, sps: int = 8, snr_db: float = 14.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    consts = build_default_constellations()
    if mod == "M-FSK":
        tones = np.array([-0.15, -0.05, 0.05, 0.15], dtype=float)
        idx = rng.integers(0, tones.size, size=num_symbols)
        freqs = np.repeat(tones[idx], sps)
        phase = 2.0 * math.pi * np.cumsum(freqs)
        syms = np.exp(1j * phase)
    else:
        c = consts[mod]
        idx = rng.integers(0, c.size, size=num_symbols)
        up = np.zeros(num_symbols * sps, dtype=np.complex128)
        up[::sps] = c[idx]
        t = np.arange(81, dtype=float) - 40.0
        sinc = np.sinc(t / float(sps))
        h = sinc / np.sqrt(np.sum(sinc ** 2))
        syms = np.convolve(up, h, mode="same")

    cfo_norm = 0.004
    n_idx = np.arange(syms.size, dtype=float)
    rot = syms * np.exp(1j * (2.0 * math.pi * cfo_norm * n_idx + 0.7))

    pwr = float(np.mean(np.abs(rot) ** 2))
    noise_pwr = pwr / (10.0 ** (snr_db / 10.0))
    noise = math.sqrt(noise_pwr / 2.0) * (rng.standard_normal(rot.size) + 1j * rng.standard_normal(rot.size))
    return rot + noise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Statistical Automatic Modulation Classifier")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--file", type=str, help="Path to raw IQ (.iq/.bin/.npy) or SigMF (.sigmf-meta) file")
    src_group.add_argument("--pluto", action="store_true", help="Capture live IQ samples from ADALM-PLUTO SDR")
    src_group.add_argument(
        "--demo",
        type=str,
        choices=["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM", "4-ASK", "M-FSK"],
        help="Run self-contained synthetic impairment demo for a given modulation",
    )

    parser.add_argument("--format", type=str, default="cf32_le", help="Raw IQ format (cf32_le, cf64_le, ci16_le, cu8, cs8, npy)")
    parser.add_argument("--sample-rate", type=float, default=1_000_000.0, help="Sample rate in Hz")
    parser.add_argument("--center-freq", type=float, default=915_000_000.0, help="Center LO frequency in Hz (for Pluto)")
    parser.add_argument("--pluto-uri", type=str, default="ip:192.168.2.1", help="Pluto IIO URI")
    parser.add_argument("--num-samples", type=int, default=8192, help="Number of IQ samples to capture or load")
    parser.add_argument("--sps", type=int, default=None, help="Optional known samples per symbol (auto-estimated if omitted)")
    parser.add_argument("--snr-db", type=float, default=14.0, help="SNR in dB for --demo mode")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")

    args = parser.parse_args(argv)
    clf = StatisticalModulationClassifier()

    if args.file:
        src = FileIQSource(args.file, fmt=args.format, sample_rate=args.sample_rate)
        batch = src.capture_batch(args.num_samples)
        fs = batch.sample_rate if batch.sample_rate is not None else args.sample_rate
        res = clf.classify_iq(batch.samples, sample_rate=fs, known_sps=args.sps)
    elif args.pluto:
        sdr_src = PlutoSDRSource(
            uri=args.pluto_uri,
            sample_rate=args.sample_rate,
            center_freq_hz=args.center_freq,
            rx_buffer_size=args.num_samples,
        )
        batch = sdr_src.capture_batch(args.num_samples)
        res = clf.classify_iq(batch.samples, sample_rate=args.sample_rate, known_sps=args.sps)
    else:
        demo_iq = _synthesize_demo_iq(args.demo, num_symbols=max(64, args.num_samples // 8), sps=args.sps or 8, snr_db=args.snr_db)
        arr_src = ArrayIQSource(demo_iq, sample_rate=args.sample_rate)
        batch = arr_src.capture_batch(demo_iq.size)
        res = clf.classify_iq(batch.samples, sample_rate=args.sample_rate, known_sps=args.sps)

    if args.json:
        print(json.dumps({
            "predicted_modulation": res.predicted_modulation,
            "confidence": res.confidence,
            "scores": res.scores,
            "estimated_snr_db": res.estimated_snr_db,
            "estimated_cfo_norm": res.estimated_cfo_norm,
            "estimated_sps": res.estimated_sps,
            "signal_detected": res.signal_detected,
            "warnings": res.warnings,
            "features": res.features,
        }, indent=2))
    else:
        print(f"Predicted Modulation : {res.predicted_modulation} (confidence={res.confidence:.3f})")
        print(f"Signal Detected      : {res.signal_detected} | Est. SNR={res.estimated_snr_db:.1f} dB | Est. CFO={res.estimated_cfo_norm:.5f} cyc/samp | Est. SPS={res.estimated_sps}")
        for w in res.warnings:
            print(f"WARNING              : {w}")
        ranked = sorted(res.scores.items(), key=lambda kv: kv[1], reverse=True)
        print("Top Candidate Scores : " + ", ".join(f"{k}={v:.3f}" for k, v in ranked[:4]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
