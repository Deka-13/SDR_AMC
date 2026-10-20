import numpy as np
import matplotlib.pyplot as plt

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
from amc_link.channel import add_awgn


def ber(tx_bits, rx_bits):
    return np.mean(tx_bits != rx_bits)


def run_experiment():
    rng = np.random.default_rng(2026)

    n_bits = 199_992

    # SNR values in dB.
    # Includes -15 dB and ends at 15 dB.
    snr_db_values = np.arange(-15, 16, 2)

    # bits per symbol for each modulation
    modulations = {
        "BPSK": (1, bits_to_bpsk, bpsk_to_bits),
        "QPSK": (2, bits_to_qpsk, qpsk_to_bits),
        "8-PSK": (3, bits_to_psk8, psk8_to_bits),
        "16-QAM": (4, bits_to_qam16, qam16_to_bits),
    }

    results_symbol_snr = {
        name: [] for name in modulations
    }

    results_bit_snr = {
        name: [] for name in modulations
    }

    # Run both experiments.
    for snr_db in snr_db_values:

        for name, (bits_per_symbol, modulator, demodulator) in modulations.items():

            # Generate random bits.
            # For 8-PSK and 16-QAM, the bit count must be
            # divisible by 3 and 4, respectively.
            bits = rng.integers(
                0, 2, n_bits, dtype=np.uint8
            )

            # Convert bits to modulation symbols.
            tx_symbols = modulator(bits)

            # -------------------------------------------------
            # Experiment A: Equal symbol SNR (Es/N0)
            # -------------------------------------------------
            rx_symbols_a = add_awgn(
                tx_symbols, snr_db, rng
            )

            rx_bits_a = demodulator(rx_symbols_a)

            results_symbol_snr[name].append(
                ber(bits, rx_bits_a)
            )

            # -------------------------------------------------
            # Experiment B: Equal bit SNR (Eb/N0)
            #
            # Es/N0 (dB) = Eb/N0 (dB)
            #              + 10*log10(bits per symbol)
            # -------------------------------------------------
            symbol_snr_db = (
                snr_db
                + 10 * np.log10(bits_per_symbol)
            )

            rx_symbols_b = add_awgn(
                tx_symbols, symbol_snr_db, rng
            )

            rx_bits_b = demodulator(rx_symbols_b)

            results_bit_snr[name].append(
                ber(bits, rx_bits_b)
            )

    # ---------------------------------------------------------
    # Plot Experiment A: Equal symbol SNR
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))

    for name, values in results_symbol_snr.items():
        plt.semilogy(
            snr_db_values,
            values,
            "o-",
            label=name,
        )

    plt.xlabel("Average symbol SNR Es/N0 (dB)")
    plt.ylabel("Bit error rate (BER)")
    plt.title("Experiment A: Equal Symbol SNR")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # ---------------------------------------------------------
    # Plot Experiment B: Equal bit SNR
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))

    for name, values in results_bit_snr.items():
        plt.semilogy(
            snr_db_values,
            values,
            "o-",
            label=name,
        )

    plt.xlabel("Bit SNR Eb/N0 (dB)")
    plt.ylabel("Bit error rate (BER)")
    plt.title("Experiment B: Equal Bit SNR")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_experiment()
