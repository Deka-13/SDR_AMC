import numpy as np
import pytest

from link.channel import (
    add_awgn,
    add_phase_jitter,
    add_carrier_frequency_offset,
    add_carrier_phase,
    add_interference,
    apply_complete_channel,
)


@pytest.mark.parametrize("num_symbols", [10, 100, 1_000, 10_000])
def test_awgn_is_reproducible(num_symbols):
    rng_a = np.random.default_rng(123)
    symbols = (
        rng_a.integers(0, 2, size=num_symbols) * 2 - 1
    ).astype(np.complex128)

    a = add_awgn(symbols, 10.0, np.random.default_rng(123))
    b = add_awgn(symbols, 10.0, np.random.default_rng(123))

    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("num_symbols", [10_000, 50_000])
def test_awgn_empirical_noise_power(num_symbols):
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=num_symbols, dtype=np.uint8)
    symbols = (1.0 - 2.0 * bits.astype(float)).astype(np.complex128)

    snr_db = 10.0
    received = add_awgn(symbols, snr_db, rng)
    noise = received - symbols

    signal_power = np.mean(np.abs(symbols) ** 2)
    measured_noise_power = np.mean(np.abs(noise) ** 2)
    expected_noise_power = signal_power / (10 ** (snr_db / 10.0))

    assert np.isclose(
        measured_noise_power,
        expected_noise_power,
        rtol=0.1,
    )


def test_phase_jitter_modifies_phase():
    symbols = np.ones(100, dtype=np.complex128)
    rng = np.random.default_rng(42)
    jittered = add_phase_jitter(symbols, jitter_std=0.2, rng=rng)
    
    # Magnitudes should remain unchanged (multiplication by exp(j*phi))
    np.testing.assert_allclose(np.abs(jittered), 1.0)
    # Phases should not all be zero
    assert not np.allclose(np.angle(jittered), 0.0)


def test_carrier_frequency_offset():
    symbols = np.ones(10, dtype=np.complex128)
    cfo = 0.05
    out = add_carrier_frequency_offset(symbols, cfo, samples_per_symbol=1)
    
    expected_phases = 2.0 * np.pi * cfo * np.arange(10)
    np.testing.assert_allclose(np.angle(out), expected_phases)


def test_carrier_phase():
    symbols = np.ones(10, dtype=np.complex128)
    theta = np.pi / 4.0
    out = add_carrier_phase(symbols, theta)
    np.testing.assert_allclose(np.angle(out), theta)


@pytest.mark.parametrize("interference_type", ["tone", "noise"])
def test_interference_power(interference_type):
    rng = np.random.default_rng(42)
    symbols = np.ones(10_000, dtype=np.complex128)
    sir_db = 10.0
    
    received = add_interference(symbols, sir_db, interference_type=interference_type, rng=rng)
    interference = received - symbols
    
    signal_power = np.mean(np.abs(symbols) ** 2)
    measured_int_power = np.mean(np.abs(interference) ** 2)
    expected_int_power = signal_power / (10 ** (sir_db / 10.0))
    
    assert np.isclose(measured_int_power, expected_int_power, rtol=0.15)


def test_apply_complete_channel():
    rng = np.random.default_rng(42)
    symbols = (rng.integers(0, 2, size=100) * 2 - 1).astype(np.complex128)
    
    out = apply_complete_channel(
        symbols,
        snr_db=15.0,
        sir_db=20.0,
        jitter_std=0.05,
        cfo=0.01,
        theta=0.5,
        interference_type="tone",
        rng=rng,
    )
    
    assert out.shape == symbols.shape
    assert out.dtype == np.complex128


@pytest.mark.parametrize(
    "symbols",
    [
        np.array([], dtype=np.complex128),
        np.array([np.nan + 0j]),
        np.array([np.inf + 0j]),
    ],
)
def test_awgn_rejects_invalid_symbols(symbols):
    with pytest.raises(ValueError):
        add_awgn(symbols, 10.0, np.random.default_rng(42))


@pytest.mark.parametrize("snr_db", [np.nan, np.inf, -np.inf])
def test_awgn_rejects_non_finite_snr(snr_db):
    symbols = np.ones(100, dtype=np.complex128)

    with pytest.raises(ValueError):
        add_awgn(symbols, snr_db, np.random.default_rng(42))


@pytest.mark.parametrize("sir_db", [np.nan, np.inf, -np.inf])
def test_interference_rejects_non_finite_sir(sir_db):
    symbols = np.ones(100, dtype=np.complex128)

    with pytest.raises(ValueError):
        add_interference(symbols, sir_db, "tone", rng=np.random.default_rng(42))
