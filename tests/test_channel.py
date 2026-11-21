import time
import numpy as np
import pytest

from channel import (
    _is_valid_int,
    add_awgn_to_signal,
    add_carrier_frequency_offset,
    add_carrier_phase,
    add_interference,
    add_phase_jitter,
    apply_multipath_fading,
    apply_timing_offset,
)


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, True),
        (5, True),
        (-2, True),
        (np.int32(3), True),
        (np.int64(7), True),
        (True, False),
        (False, False),
        (np.bool_(True), False),
        (1.0, False),
        ("2", False),
        (None, False),
    ],
)
def test_is_valid_int(value, expected):
    assert _is_valid_int(value) is expected


# ---------------------------------------------------------------------------
# Multipath fading tests
# ---------------------------------------------------------------------------


def test_multipath_default_delays_and_output_shape():
    rng = np.random.default_rng(101)
    signal = np.ones(128, dtype=np.complex128)

    out = apply_multipath_fading(signal, rng=rng)

    # Default path_delays is [0, 2, 5] -> convolution length is N + 5
    assert out.size == signal.size + 5
    assert out.dtype == np.complex128
    assert np.all(np.isfinite(out))


def test_multipath_rng_reproducibility():
    signal = np.exp(1j * np.linspace(0, np.pi, 64))

    out_a = apply_multipath_fading(signal, rng=np.random.default_rng(777))
    out_b = apply_multipath_fading(signal, rng=np.random.default_rng(777))
    out_c = apply_multipath_fading(signal, rng=np.random.default_rng(778))

    np.testing.assert_array_equal(out_a, out_b)
    assert not np.array_equal(out_a, out_c)


def test_multipath_custom_gains_and_overlapping_delays():
    # Impulse input isolates the exact channel impulse response
    impulse = np.zeros(8, dtype=np.complex128)
    impulse[0] = 1.0 + 0.0j

    gains = [1.0 + 0.0j, 0.5j, 0.25 - 0.25j]
    delays = [0, 2, 2]  # delay=2 sums both gains: 0.25 + 0.25j

    out = apply_multipath_fading(impulse, path_delays=delays, path_gains=gains)

    assert out.size == impulse.size + 2
    assert np.isclose(out[0], 1.0 + 0.0j)
    assert np.isclose(out[1], 0.0 + 0.0j)
    assert np.isclose(out[2], 0.25 + 0.25j)


def test_multipath_random_gain_unit_expected_power():
    rng = np.random.default_rng(2026)
    impulse = np.zeros(16, dtype=np.complex128)
    impulse[0] = 1.0

    # For each draw, raw_amps is normalized so sum(|gain|^2) == 1
    powers = []
    for _ in range(200):
        resp = apply_multipath_fading(impulse, path_delays=[0, 2, 5], rng=rng)
        powers.append(np.sum(np.abs(resp) ** 2))

    np.testing.assert_allclose(powers, np.ones_like(powers), atol=1e-12)


@pytest.mark.parametrize(
    ("signal", "delays", "gains"),
    [
        (np.array([], dtype=np.complex128), [0, 1], None),
        (np.ones(10), [], None),
        (np.ones(10), [-1, 2], None),
        (np.ones(10), [0.5], None),
        (np.ones(10), [True], None),
        (np.ones(10), [0, 2], [1.0 + 0j]),
        (np.ones(10), [0, 2], [1.0 + 0j, np.nan + 0j]),
        (np.ones(10), [0, 2], [1.0 + 0j, np.inf + 0j]),
    ],
)
def test_multipath_invalid_inputs_raise(signal, delays, gains):
    with pytest.raises(ValueError):
        apply_multipath_fading(signal, path_delays=delays, path_gains=gains)


# ---------------------------------------------------------------------------
# Timing offset tests
# ---------------------------------------------------------------------------


def test_timing_offset_zero_returns_independent_copy():
    signal = np.arange(10, dtype=np.complex128) * (1.0 + 0.5j)
    shifted = apply_timing_offset(signal, epsilon=0.0)

    np.testing.assert_array_equal(shifted, signal)
    assert shifted is not signal
    shifted[0] = -999.0 + 0j
    assert signal[0] != -999.0 + 0j


@pytest.mark.parametrize("epsilon", [0.25, 0.5, 0.75])
def test_timing_offset_fractional_shift_on_linear_ramp(epsilon):
    # On a linear ramp x[n] = n, linear interpolation at n + 1 - epsilon
    # (with interior values) equals n + 1 - epsilon
    n = np.arange(16, dtype=float)
    signal = n + 1j * (2.0 * n)

    out = apply_timing_offset(signal, epsilon=epsilon)

    assert out.size == signal.size
    assert np.iscomplexobj(out)
    # Check interior samples away from zero-padded boundaries
    expected_interior = (n[2:-2] + 1.0 - epsilon) + 1j * (2.0 * (n[2:-2] + 1.0 - epsilon))
    np.testing.assert_allclose(out[2:-2], expected_interior, atol=1e-12)


@pytest.mark.parametrize("epsilon", [-0.1, 1.0, 1.5, np.nan, np.inf, -np.inf])
def test_timing_offset_invalid_epsilon_raises(epsilon):
    with pytest.raises(ValueError):
        apply_timing_offset(np.ones(16, dtype=np.complex128), epsilon=epsilon)


def test_timing_offset_empty_signal_raises():
    with pytest.raises(ValueError):
        apply_timing_offset(np.array([], dtype=np.complex128), epsilon=0.2)


# ---------------------------------------------------------------------------
# AWGN tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_samples", [16, 256, 4096])
def test_awgn_reproducibility(num_samples):
    signal = np.ones(num_samples, dtype=np.complex128)

    out_a = add_awgn_to_signal(signal, ref_power=1.0, snr_db=15.0, rng=np.random.default_rng(12))
    out_b = add_awgn_to_signal(signal, ref_power=1.0, snr_db=15.0, rng=np.random.default_rng(12))

    np.testing.assert_array_equal(out_a, out_b)


@pytest.mark.parametrize(
    ("ref_power", "snr_db"),
    [
        (1.0, 10.0),
        (4.0, 20.0),
        (0.25, 5.0),
    ],
)
def test_awgn_empirical_noise_power_and_circular_symmetry(ref_power, snr_db):
    rng = np.random.default_rng(42)
    # Use a low-power signal so ref_power controls the noise floor directly
    signal = 0.01 * np.ones(50_000, dtype=np.complex128)

    noisy = add_awgn_to_signal(signal, ref_power=ref_power, snr_db=snr_db, rng=rng)
    noise = noisy - signal

    expected_power = ref_power / (10.0 ** (snr_db / 10.0))
    measured_power = np.mean(np.abs(noise) ** 2)

    assert np.isclose(measured_power, expected_power, rtol=0.05)
    # Verify real and imaginary parts share noise power equally
    assert np.isclose(np.var(noise.real), expected_power / 2.0, rtol=0.08)
    assert np.isclose(np.var(noise.imag), expected_power / 2.0, rtol=0.08)


@pytest.mark.parametrize("ref_power", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_awgn_invalid_ref_power_raises(ref_power):
    with pytest.raises(ValueError):
        add_awgn_to_signal(np.ones(32, dtype=np.complex128), ref_power, 10.0, np.random.default_rng(1))


@pytest.mark.parametrize("snr_db", [np.nan, np.inf, -np.inf])
def test_awgn_invalid_snr_raises(snr_db):
    with pytest.raises(ValueError):
        add_awgn_to_signal(np.ones(32, dtype=np.complex128), 1.0, snr_db, np.random.default_rng(1))


# ---------------------------------------------------------------------------
# Phase jitter, CFO, and carrier phase tests
# ---------------------------------------------------------------------------


def test_phase_jitter_behavior_and_zero_std():
    signal = np.ones(20_000, dtype=np.complex128)

    assert add_phase_jitter(signal, jitter_std=0.0, rng=np.random.default_rng(1)) is signal
    assert add_phase_jitter(signal, jitter_std=-0.5, rng=np.random.default_rng(1)) is signal

    jitter_std = 0.15
    out = add_phase_jitter(signal, jitter_std=jitter_std, rng=np.random.default_rng(99))
    np.testing.assert_allclose(np.abs(out), 1.0, atol=1e-12)
    assert np.isclose(np.std(np.angle(out)), jitter_std, rtol=0.05)


def test_cfo_and_samples_per_symbol():
    signal = np.ones(12, dtype=np.complex128)
    assert add_carrier_frequency_offset(signal, cfo=0.0, samples_per_symbol=4) is signal

    cfo = 0.1
    sps = 4
    out = add_carrier_frequency_offset(signal, cfo=cfo, samples_per_symbol=sps)
    expected = np.exp(1j * 2.0 * np.pi * cfo * (np.arange(12) / float(sps)))
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_carrier_phase():
    signal = np.exp(1j * np.linspace(0, 1.0, 16))
    assert add_carrier_phase(signal, theta=0.0) is signal

    theta = np.pi / 3.0
    out = add_carrier_phase(signal, theta=theta)
    np.testing.assert_allclose(out, signal * np.exp(1j * theta), atol=1e-12)


# ---------------------------------------------------------------------------
# Interference tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interference_type", ["tone", "noise"])
@pytest.mark.parametrize(("ref_power", "sir_db"), [(1.0, 10.0), (2.5, 6.0)])
def test_interference_power_and_reproducibility(interference_type, ref_power, sir_db):
    signal = np.zeros(50_000, dtype=np.complex128)

    out_a = add_interference(
        signal,
        ref_power=ref_power,
        sir_db=sir_db,
        interference_type=interference_type,
        tone_freq=0.125,
        rng=np.random.default_rng(55),
    )
    out_b = add_interference(
        signal,
        ref_power=ref_power,
        sir_db=sir_db,
        interference_type=interference_type,
        tone_freq=0.125,
        rng=np.random.default_rng(55),
    )

    np.testing.assert_array_equal(out_a, out_b)

    expected_power = ref_power / (10.0 ** (sir_db / 10.0))
    measured_power = np.mean(np.abs(out_a) ** 2)
    assert np.isclose(measured_power, expected_power, rtol=0.05)


def test_tone_interference_frequency():
    n_samples = 1024
    tone_freq = 0.125  # normalized cycles/sample -> bin 128 of 1024
    signal = np.zeros(n_samples, dtype=np.complex128)

    out = add_interference(
        signal,
        ref_power=1.0,
        sir_db=0.0,
        interference_type="tone",
        tone_freq=tone_freq,
        rng=np.random.default_rng(123),
    )

    spectrum = np.abs(np.fft.fft(out))
    expected_bin = int(tone_freq * n_samples)
    assert int(np.argmax(spectrum)) == expected_bin


@pytest.mark.parametrize("ref_power", [0.0, -2.0, np.nan, np.inf])
def test_interference_invalid_ref_power_raises(ref_power):
    with pytest.raises(ValueError):
        add_interference(np.ones(16, dtype=np.complex128), ref_power, 10.0, rng=np.random.default_rng(1))


@pytest.mark.parametrize("sir_db", [np.nan, np.inf, -np.inf])
def test_interference_invalid_sir_raises(sir_db):
    with pytest.raises(ValueError):
        add_interference(np.ones(16, dtype=np.complex128), 1.0, sir_db, rng=np.random.default_rng(1))


def test_interference_unsupported_type_raises():
    with pytest.raises(ValueError):
        add_interference(
            np.ones(16, dtype=np.complex128),
            1.0,
            10.0,
            interference_type="chirp",
            rng=np.random.default_rng(1),
        )


# ---------------------------------------------------------------------------
# Performance benchmark tests
# ---------------------------------------------------------------------------


def test_performance_channel_pipeline_large_signal():
    """Verify end-to-end channel impairment functions run quickly on large sample buffers."""
    rng = np.random.default_rng(2026)
    num_samples = 250_000
    signal = (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)) / np.sqrt(2.0)

    # Warm-up call
    _ = add_awgn_to_signal(signal[:1024], ref_power=1.0, snr_db=20.0, rng=rng)

    start = time.perf_counter()

    x = apply_multipath_fading(signal, path_delays=[0, 3, 7, 12], rng=rng)
    x = apply_timing_offset(x, epsilon=0.35)
    x = add_carrier_frequency_offset(x, cfo=0.002, samples_per_symbol=4)
    x = add_carrier_phase(x, theta=0.4)
    x = add_phase_jitter(x, jitter_std=0.02, rng=rng)
    x = add_interference(x, ref_power=1.0, sir_db=25.0, interference_type="tone", rng=rng)
    x = add_awgn_to_signal(x, ref_power=1.0, snr_db=20.0, rng=rng)

    elapsed = time.perf_counter() - start

    assert x.size == num_samples + 12
    assert np.all(np.isfinite(x))
    assert elapsed < 1.0, f"Channel pipeline took {elapsed:.3f}s for {num_samples} samples (expected < 1.0s)"
