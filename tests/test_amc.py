"""Comprehensive pytest suite for IQ ingestion, preprocessing, statistics, and classification."""

import json
import math
import numpy as np
import pytest

from amc.amc_io import (
    ArrayIQSource,
    FileIQSource,
    detect_clipping_and_saturation,
    load_iq_file,
    load_sigmf,
)
from amc.amc_preprocess import (
    detect_signal_energy_and_cyclostationary,
    estimate_coarse_cfo_normalized,
    estimate_snr_db_blind,
    preprocess_iq_stream,
)
from amc.amc_statistics import (
    alrt_phase_marginalized_loglik,
    cyclic_autocorrelation_scan,
    glrt_joint_phase_symbol_loglik,
    higher_order_cumulants_zero_lag,
    hlrt_phase_profiled_loglik,
    quasi_alrt_psk_metrics,
    sample_cumulant_zero_lag,
    sample_moment,
)
from amc.amc_classifier import (
    StatisticalModulationClassifier,
    build_default_constellations,
    _synthesize_demo_iq,
)


def test_raw_and_sigmf_file_loading(tmp_path):
    rng = np.random.default_rng(10)
    iq = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64) * 0.25

    cf32_path = tmp_path / "capture.cf32"
    interleaved = np.empty(iq.size * 2, dtype="<f4")
    interleaved[0::2] = iq.real
    interleaved[1::2] = iq.imag
    interleaved.tofile(cf32_path)

    loaded, meta = load_iq_file(cf32_path, fmt="cf32_le", sample_rate=2.0e6)
    assert loaded.shape == (512,)
    assert meta["sample_rate"] == 2.0e6
    assert np.allclose(loaded, iq, atol=1e-6)

    # SigMF metadata + data
    sigmf_data = tmp_path / "test.sigmf-data"
    sigmf_meta = tmp_path / "test.sigmf-meta"
    interleaved.tofile(sigmf_data)
    sigmf_meta.write_text(
        json.dumps({
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 1.5e6,
                "core:description": "unit test capture",
            },
            "captures": [{"core:sample_start": 0, "core:frequency": 433.92e6}],
        }),
        encoding="utf-8",
    )

    s_iq, s_meta = load_sigmf(sigmf_meta)
    assert s_iq.shape == (512,)
    assert s_meta["sample_rate"] == 1.5e6
    assert s_meta["center_freq_hz"] == 433.92e6


def test_clipping_and_array_source():
    clipped = np.ones(128, dtype=np.complex128) * (1.0 + 1.0j)
    warns = detect_clipping_and_saturation(clipped)
    assert len(warns) >= 1

    src = ArrayIQSource(np.arange(10, dtype=np.complex128), loop=False)
    b1 = src.capture_batch(6)
    assert b1.samples.size == 6
    b2 = src.capture_batch(4)
    assert b2.samples.size == 4
    with pytest.raises(StopIteration):
        src.capture_batch(1)


def test_cumulants_and_likelihood_ordering():
    consts = build_default_constellations()
    rng = np.random.default_rng(7)
    bpsk_syms = consts["BPSK"][rng.integers(0, 2, size=2000)]
    qpsk_syms = consts["QPSK"][rng.integers(0, 4, size=2000)]

    c_bpsk = higher_order_cumulants_zero_lag(bpsk_syms)
    c_qpsk = higher_order_cumulants_zero_lag(qpsk_syms)

    # Theoretical c40 / c21^2: BPSK -> -2.0, QPSK -> +1.0
    assert abs(c_bpsk["c40_norm"] - (-2.0)) < 0.25
    assert abs(c_qpsk["c40_norm"] - (1.0)) < 0.25

    # Phase-blind v2_psk is close to 1.0 for BPSK and near 0 for QPSK
    v_bpsk = quasi_alrt_psk_metrics(bpsk_syms)
    v_qpsk = quasi_alrt_psk_metrics(qpsk_syms)
    assert v_bpsk["v2_psk"] > 0.85
    assert v_qpsk["v2_psk"] < 0.15

    # HLRT vs GLRT on BPSK symbols under phase rotation
    rot_bpsk = bpsk_syms * np.exp(1j * 0.45)
    hlrt_b = hlrt_phase_profiled_loglik(rot_bpsk, consts["BPSK"], snr_linear=20.0)
    hlrt_q = hlrt_phase_profiled_loglik(rot_bpsk, consts["QPSK"], snr_linear=20.0)
    assert hlrt_b > hlrt_q

    glrt_b = glrt_joint_phase_symbol_loglik(rot_bpsk, consts["BPSK"], snr_linear=20.0)
    assert math.isfinite(glrt_b)
    alrt_b = alrt_phase_marginalized_loglik(rot_bpsk, consts["BPSK"], snr_linear=20.0)
    assert math.isfinite(alrt_b)


def test_noise_detection_and_cfo_estimation():
    rng = np.random.default_rng(99)
    pure_noise = 1e-4 * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048))
    det = detect_signal_energy_and_cyclostationary(pure_noise, sample_rate=1e6)
    assert det.snr_db < 2.0

    # CFO estimation accuracy on BPSK with known CFO = 0.005 cycles/sample
    consts = build_default_constellations()
    syms = consts["BPSK"][rng.integers(0, 2, size=512)]
    up = np.zeros(512 * 4, dtype=np.complex128)
    up[::4] = syms
    true_cfo = 0.005
    n_idx = np.arange(up.size, dtype=float)
    rx = up * np.exp(1j * 2.0 * math.pi * true_cfo * n_idx) + 0.02 * (rng.standard_normal(up.size) + 1j * rng.standard_normal(up.size))
    est_cfo = estimate_coarse_cfo_normalized(rx, powers=(2,))
    assert abs(est_cfo - true_cfo) < 0.001


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM", "M-FSK"])
def test_classifier_on_synthesized_impairments(mod):
    clf = StatisticalModulationClassifier()
    iq = _synthesize_demo_iq(mod, num_symbols=600, sps=8, snr_db=18.0, seed=123)
    res = clf.classify_iq(iq, sample_rate=1_000_000.0, known_sps=8)
    assert res.predicted_modulation == mod
    assert 0.0 <= res.confidence <= 1.0
    assert math.isclose(sum(res.scores.values()), 1.0, rel_tol=1e-6)
