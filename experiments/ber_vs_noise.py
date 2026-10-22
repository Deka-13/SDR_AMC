import csv
from pathlib import Path

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
    bits_to_qam64,
    qam64_to_bits,
)
from amc_link.channel import add_awgn


# Save results next to this script, regardless of the working directory.
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def count_bit_errors(tx_bits, rx_bits):
    """Return the number of incorrectly received bits."""
    return int(np.count_nonzero(tx_bits != rx_bits))


def ber(tx_bits, rx_bits):
    """Calculate the bit error rate."""
    return count_bit_errors(tx_bits, rx_bits) / tx_bits.size


def run_experiment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)

    # Divisible by 1, 2, 3, 4, and 6.
    n_bits = 199_992

    # SNR values from -20 dB to +20 dB, in 2 dB steps.
    snr_db_values = np.arange(-20, 21, 2)

    # Modulation name: (bits per symbol, modulator, demodulator).
    modulations = {
        "BPSK": (1, bits_to_bpsk, bpsk_to_bits),
        "QPSK": (2, bits_to_qpsk, qpsk_to_bits),
        "8-PSK": (3, bits_to_psk8, psk8_to_bits),
        "16-QAM": (4, bits_to_qam16, qam16_to_bits),
        "64-QAM": (6, bits_to_qam64, qam64_to_bits),
    }

    results_symbol_snr = {
        name: [] for name in modulations
    }

    results_bit_snr = {
        name: [] for name in modulations
    }

    # One CSV row per modulation, SNR, and experiment.
    csv_rows = []

    for snr_db in snr_db_values:
        for name, (
            bits_per_symbol,
            modulator,
            demodulator,
        ) in modulations.items():

            # Generate random information bits.
            bits = rng.integers(
                0, 2, n_bits, dtype=np.uint8
            )

            # Modulate the bits.
            tx_symbols = modulator(bits)

            # -------------------------------------------------
            # Experiment A: Equal symbol SNR (Es/N0)
            # -------------------------------------------------
            rx_symbols_a = add_awgn(
                tx_symbols, snr_db, rng
            )
            rx_bits_a = demodulator(rx_symbols_a)

            errors_a = count_bit_errors(bits, rx_bits_a)
            ber_a = errors_a / bits.size

            results_symbol_snr[name].append(ber_a)

            csv_rows.append({
                "experiment": "equal_symbol_snr",
                "modulation": name,
                "bits_per_symbol": bits_per_symbol,
                "snr_db": float(snr_db),
                "snr_definition": "Es/N0",
                "esn0_db": float(snr_db),
                "ebn0_db": float(
                    snr_db - 10 * np.log10(bits_per_symbol)
                ),
                "total_bits": int(bits.size),
                "bit_errors": errors_a,
                "ber": ber_a,
            })

            # -------------------------------------------------
            # Experiment B: Equal bit SNR (Eb/N0)
            #
            # Es/N0 (dB) = Eb/N0 (dB)
            #              + 10*log10(bits per symbol)
            # -------------------------------------------------
            symbol_snr_db = (
                snr_db + 10 * np.log10(bits_per_symbol)
            )

            rx_symbols_b = add_awgn(
                tx_symbols, symbol_snr_db, rng
            )
            rx_bits_b = demodulator(rx_symbols_b)

            errors_b = count_bit_errors(bits, rx_bits_b)
            ber_b = errors_b / bits.size

            results_bit_snr[name].append(ber_b)

            csv_rows.append({
                "experiment": "equal_bit_snr",
                "modulation": name,
                "bits_per_symbol": bits_per_symbol,
                "snr_db": float(snr_db),
                "snr_definition": "Eb/N0",
                "esn0_db": float(symbol_snr_db),
                "ebn0_db": float(snr_db),
                "total_bits": int(bits.size),
                "bit_errors": errors_b,
                "ber": ber_b,
            })

    # ---------------------------------------------------------
    # Save CSV results and statistics.
    # ---------------------------------------------------------
    csv_path = RESULTS_DIR / "ber_results.csv"

    fieldnames = [
        "experiment",
        "modulation",
        "bits_per_symbol",
        "snr_db",
        "snr_definition",
        "esn0_db",
        "ebn0_db",
        "total_bits",
        "bit_errors",
        "ber",
    ]

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # ---------------------------------------------------------
    # Plot and save Experiment A: Equal symbol SNR.
    # ---------------------------------------------------------
    fig_a, ax_a = plt.subplots(figsize=(9, 6))

    for name, values in results_symbol_snr.items():
        ax_a.semilogy(
            snr_db_values,
            [value if value > 0 else np.nan for value in values],
            "o-",
            label=name,
        )

    ax_a.set_xlabel("Average symbol SNR Es/N0 (dB)")
    ax_a.set_ylabel("Bit error rate (BER)")
    ax_a.set_title("Experiment A: Equal Symbol SNR")
    ax_a.grid(True, which="both")
    ax_a.legend()
    fig_a.tight_layout()

    plot_a_path = RESULTS_DIR / "ber_equal_symbol_snr.png"
    fig_a.savefig(plot_a_path, dpi=300, bbox_inches="tight")

    # ---------------------------------------------------------
    # Plot and save Experiment B: Equal bit SNR.
    # ---------------------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=(9, 6))

    for name, values in results_bit_snr.items():
        ax_b.semilogy(
            snr_db_values,
            [value if value > 0 else np.nan for value in values],
            "o-",
            label=name,
        )

    ax_b.set_xlabel("Bit SNR Eb/N0 (dB)")
    ax_b.set_ylabel("Bit error rate (BER)")
    ax_b.set_title("Experiment B: Equal Bit SNR")
    ax_b.grid(True, which="both")
    ax_b.legend()
    fig_b.tight_layout()

    plot_b_path = RESULTS_DIR / "ber_equal_bit_snr.png"
    fig_b.savefig(plot_b_path, dpi=300, bbox_inches="tight")

    print(f"Results directory: {RESULTS_DIR}")
    print(f"CSV saved to:      {csv_path}")
    print(f"Symbol-SNR plot:   {plot_a_path}")
    print(f"Bit-SNR plot:      {plot_b_path}")
    print(f"CSV data rows:     {len(csv_rows)}")

    plt.show()


if __name__ == "__main__":
    run_experiment()
