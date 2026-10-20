import numpy as np
import pytest

from amc_link.channel import add_awgn


@pytest.mark.parametrize("num_symbols", [10, 100, 1_000, 10_000])
def test_awgn_is_reproducible(num_symbols):
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    symbols = (
        rng_a.integers(0, 2, size=num_symbols) * 2 - 1
    ).astype(np.complex128)

    a = add_awgn(symbols, 10.0, np.random.default_rng(123))
    b = add_awgn(symbols, 10.0, np.random.default_rng(123))

    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("num_symbols", [10_000, 100_000, 200_000])
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
        rtol=0.05,
    )


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
