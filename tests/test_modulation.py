import numpy as np
import pytest

from amc_link.modulation import (
    bits_to_bpsk,
    bpsk_to_bits,
    bits_to_qpsk,
    qpsk_to_bits,
    bits_to_psk8,
    psk8_to_bits,
    bits_to_qam16,
    qam16_to_bits,
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

@pytest.mark.parametrize("num_bits", [12, 99, 999, 9999])
def test_8psk_round_trip(num_bits):
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    symbols = bits_to_psk8(bits)
    recovered = psk8_to_bits(symbols)

    np.testing.assert_array_equal(recovered, bits)

@pytest.mark.parametrize("num_bits", [12, 100, 1000, 10000])
def test_16qam_round_trip(num_bits):
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    symbols = bits_to_qam16(bits)
    recovered = qam16_to_bits(symbols)

    np.testing.assert_array_equal(recovered, bits)



def test_qpsk_rejects_odd_number_of_bits():
    bits = np.array([0, 1, 0], dtype=np.uint8)

    with pytest.raises(ValueError):
        bits_to_qpsk(bits)

def test_8psk_rejects_non_multiple_of_three_bits():
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)

    with pytest.raises(ValueError):
        bits_to_psk8(bits)

def test_16qam_rejects_non_multiple_of_four_bits():
    bits = np.array([0, 1, 0, 1, 1], dtype=np.uint8)

    with pytest.raises(ValueError):
        bits_to_qam16(bits)



@pytest.mark.parametrize("num_bits", [1200, 12000])
def test_average_symbol_energy(num_bits):
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    bpsk = bits_to_bpsk(bits)
    qpsk = bits_to_qpsk(bits)
    psk8 = bits_to_psk8(bits)
    qam16 = bits_to_qam16(bits)

    assert np.isclose(np.mean(np.abs(bpsk) ** 2), 1.0)
    assert np.isclose(np.mean(np.abs(qpsk) ** 2), 1.0)
    assert np.isclose(np.mean(np.abs(psk8) ** 2), 1.0)
    assert np.isclose(
        np.mean(np.abs(qam16) ** 2),
        1.0,
        atol=0.03,
    )

def test_16qam_constellation_average_energy():
    import itertools

    bits = np.array(
        list(itertools.product([0, 1], repeat=4)),
        dtype=np.uint8,
    ).reshape(-1)

    symbols = bits_to_qam16(bits)

    assert np.isclose(np.mean(np.abs(symbols) ** 2), 1.0)


@pytest.mark.parametrize(
    "modulate",
    [bits_to_bpsk, bits_to_qpsk, bits_to_psk8, bits_to_qam16],
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

@pytest.mark.parametrize(
    "demodulate",
    [bpsk_to_bits, qpsk_to_bits, psk8_to_bits, qam16_to_bits],
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
