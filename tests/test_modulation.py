import numpy as np
import pytest

from amc_link.modulation import (
    bits_to_bpsk,
    bpsk_to_bits,
    bits_to_qpsk,
    qpsk_to_bits,
)


@pytest.mark.parametrize("num_bits", [10, 100, 1000, 10000])
def test_bpsk_round_trip(num_bits):
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    symbols = bits_to_bpsk(bits)
    recovered = bpsk_to_bits(symbols)

    np.testing.assert_array_equal(recovered, bits)


@pytest.mark.parametrize("num_bits", [10, 100, 1000, 10000])
def test_qpsk_round_trip(num_bits):
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    symbols = bits_to_qpsk(bits)
    recovered = qpsk_to_bits(symbols)

    np.testing.assert_array_equal(recovered, bits)


@pytest.mark.parametrize("num_bits", [10, 100, 1000, 10000])
def test_average_symbol_energy(num_bits):
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    bpsk = bits_to_bpsk(bits)
    qpsk = bits_to_qpsk(bits)

    assert np.isclose(np.mean(np.abs(bpsk) ** 2), 1.0)
    assert np.isclose(np.mean(np.abs(qpsk) ** 2), 1.0)


@pytest.mark.parametrize(
    "modulate",
    [bits_to_bpsk, bits_to_qpsk],
)
@pytest.mark.parametrize(
    "bits",
    [
        np.array([], dtype=np.uint8),
        np.array([0, 1, 2], dtype=np.uint8),
        np.array([0, 1, 3], dtype=np.uint8),
        np.array([-1, 0, 1]),
    ],
)
def test_modulation_rejects_invalid_bits(modulate, bits):
    with pytest.raises(ValueError):
        modulate(bits)


def test_qpsk_rejects_odd_number_of_bits():
    bits = np.array([0, 1, 0], dtype=np.uint8)

    with pytest.raises(ValueError):
        bits_to_qpsk(bits)


@pytest.mark.parametrize(
    "demodulate",
    [bpsk_to_bits, qpsk_to_bits],
)
@pytest.mark.parametrize(
    "symbols",
    [
        np.array([], dtype=np.complex128),
        np.array([np.nan + 0j]),
        np.array([np.inf + 0j]),
    ],
)
def test_demodulation_rejects_invalid_symbols(demodulate, symbols):
    with pytest.raises(ValueError):
        demodulate(symbols)
