import math
import numpy as np
import pytest

from coding import (
    AMCController,
    LDPCCodec,
    MCSMode,
    _count_4cycles,
    _find_invertible_parity_cols,
    _is_finite_number,
    _is_valid_int,
    _logsumexp,
    awgn_bpsk_llr,
    bsc_llr,
    build_constellation,
    generate_regular_ldpc_matrix,
    gf2_inv,
    gf2_rank,
    qam_soft_llr,
    simulate_amc_block,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_codec() -> LDPCCodec:
    return LDPCCodec()


@pytest.fixture
def regular_h_14_7() -> np.ndarray:
    return generate_regular_ldpc_matrix(
        n=14, k=7, w_c=3, avoid_4cycles=True, require_full_rank=True, seed=42
    )


@pytest.fixture
def amc_controller(default_codec: LDPCCodec) -> AMCController:
    return AMCController([
        MCSMode(name="BPSK_R0.5", mod_order=2, codec=default_codec, min_snr_db=-2.0, target_bler=1e-2),
        MCSMode(name="QPSK_R0.5", mod_order=4, codec=default_codec, min_snr_db=2.0, target_bler=1e-2),
        MCSMode(name="16QAM_R0.5", mod_order=16, codec=default_codec, min_snr_db=8.0, target_bler=1e-2),
    ])


# ---------------------------------------------------------------------------
# Required Table Tests & Core Functional Verification
# ---------------------------------------------------------------------------

def test_gf2_inverse_identity(default_codec: LDPCCodec):
    """GF(2) inverse: P * P^{-1} == I and P^{-1} * P == I over GF(2)."""
    H = default_codec.H
    _, parity_cols, P_inv = _find_invertible_parity_cols(H, default_codec.k)
    P = H[:, parity_cols] & 1
    eye = np.eye(P.shape[0], dtype=int)

    assert np.array_equal((P @ P_inv) % 2, eye)
    assert np.array_equal((P_inv @ P) % 2, eye)
    assert np.array_equal(gf2_inv(P), P_inv)


def test_matrix_rank_equals_m(regular_h_14_7: np.ndarray):
    """Matrix rank: rank(H) == m over GF(2)."""
    m, _ = regular_h_14_7.shape
    assert m == 7
    assert gf2_rank(regular_h_14_7) == m


def test_column_weights_equal_wc(regular_h_14_7: np.ndarray):
    """Column weights: all columns have equal weight w_c."""
    w_c = 3
    col_weights = np.sum(regular_h_14_7, axis=0)
    assert col_weights.shape == (regular_h_14_7.shape[1],)
    assert np.all(col_weights == w_c)


def test_row_weights_equal_wr(regular_h_14_7: np.ndarray):
    """Row weights: all rows have equal weight w_r = (n * w_c) / m."""
    m, n = regular_h_14_7.shape
    expected_wr = (n * 3) // m
    row_weights = np.sum(regular_h_14_7, axis=1)
    assert row_weights.shape == (m,)
    assert np.all(row_weights == expected_wr)


def test_four_cycle_count_is_zero(regular_h_14_7: np.ndarray):
    """4-cycle count: zero 4-cycles in generated regular LDPC matrix."""
    assert _count_4cycles(regular_h_14_7) == 0


def test_encoder_syndrome_zero_and_message_preserved(default_codec: LDPCCodec):
    """Encoder syndrome: H @ c^T == 0 (mod 2) and systematic message u is preserved."""
    rng = np.random.default_rng(2026)
    for length in [1, 3, 7, 14, 21, 35]:
        msg = rng.integers(0, 2, size=length)
        codeword, orig_size = default_codec.encode(msg)
        assert orig_size == length
        assert codeword.size % default_codec.n == 0

        blocks = codeword.reshape(-1, default_codec.n)
        syndromes = (blocks @ default_codec.H.T) % 2
        assert np.all(syndromes == 0)

        recovered_padded = np.concatenate([default_codec.extract_message_bits(b) for b in blocks])
        assert np.array_equal(recovered_padded[:orig_size], msg)


@pytest.mark.parametrize("algorithm", ["sum-product", "min-sum", "normalized-min-sum"])
def test_noiseless_bpsk_zero_ber(default_codec: LDPCCodec, algorithm: str):
    """Noiseless BPSK: zero BER and zero block errors across iterative decoders."""
    rng = np.random.default_rng(101)
    msg = rng.integers(0, 2, size=28)
    cw, orig_size = default_codec.encode(msg)

    syms, bit_map = build_constellation(2)
    sym_map = {int(bit_map[i, 0]): syms[i] for i in range(2)}
    tx_syms = np.array([sym_map[int(b)] for b in cw])

    llrs = qam_soft_llr(tx_syms, mod_order=2, noise_var_per_dim=1e-4)
    decoded, stats = default_codec.decode(
        llrs, orig_size=orig_size, max_iter=25, algorithm=algorithm, reference_msg_bits=msg
    )

    assert np.array_equal(decoded, msg)
    assert stats["msg_error_blocks"] == 0
    assert stats["syndrome_failed_blocks"] == 0
    assert stats["bler"] == 0.0


@pytest.mark.parametrize("algorithm", ["sum-product", "min-sum", "normalized-min-sum"])
def test_noiseless_qpsk_zero_ber(default_codec: LDPCCodec, algorithm: str):
    """Noiseless QPSK: zero BER and zero block errors across iterative decoders."""
    rng = np.random.default_rng(202)
    msg = rng.integers(0, 2, size=28)
    cw, orig_size = default_codec.encode(msg)

    syms, bit_map = build_constellation(4)
    bit_to_idx = {tuple(row.tolist()): idx for idx, row in enumerate(bit_map)}
    tx_syms = np.array([syms[bit_to_idx[tuple(cw[i : i + 2].tolist())]] for i in range(0, len(cw), 2)])

    llrs = qam_soft_llr(tx_syms, mod_order=4, noise_var_per_dim=1e-4)
    decoded, stats = default_codec.decode(
        llrs, orig_size=orig_size, max_iter=25, algorithm=algorithm, reference_msg_bits=msg
    )

    assert np.array_equal(decoded, msg)
    assert stats["msg_error_blocks"] == 0
    assert stats["syndrome_failed_blocks"] == 0
    assert stats["bler"] == 0.0


@pytest.mark.parametrize("algorithm", ["sum-product", "min-sum", "normalized-min-sum"])
def test_noiseless_16qam_zero_ber_with_actual_padding(default_codec: LDPCCodec, algorithm: str):
    """Noiseless 16-QAM: tests non-multiple-of-4 codeword bit padding (14 mod 4 == 2) and zero BER."""
    rng = np.random.default_rng(303)
    msg = rng.integers(0, 2, size=7)  # 14 bits -> 14 % 4 == 2 -> pad = 2
    cw, orig_size = default_codec.encode(msg)
    assert cw.size == 14
    pad = (4 - (cw.size % 4)) % 4
    assert pad == 2

    padded_cw = np.pad(cw, (0, pad), mode="constant")
    syms, bit_map = build_constellation(16)
    bit_to_idx = {tuple(row.tolist()): idx for idx, row in enumerate(bit_map)}
    tx_syms = np.array(
        [syms[bit_to_idx[tuple(padded_cw[i : i + 4].tolist())]] for i in range(0, len(padded_cw), 4)]
    )

    all_llrs = qam_soft_llr(tx_syms, mod_order=16, noise_var_per_dim=1e-4)
    assert all_llrs.size == padded_cw.size
    coded_llrs = all_llrs[: cw.size]

    decoded, stats = default_codec.decode(
        coded_llrs, orig_size=orig_size, max_iter=25, algorithm=algorithm, reference_msg_bits=msg
    )

    assert np.array_equal(decoded, msg)
    assert stats["msg_error_blocks"] == 0
    assert stats["syndrome_failed_blocks"] == 0
    assert stats["bler"] == 0.0


@pytest.mark.parametrize("algorithm", ["sum-product", "min-sum", "normalized-min-sum"])
def test_decoder_convergence_moderate_snr_all_algorithms(default_codec: LDPCCodec, algorithm: str):
    """Decoder convergence: recovers message bits under moderate AWGN across all soft algorithms."""
    rng = np.random.default_rng(404)
    msg = rng.integers(0, 2, size=28)
    cw, orig_size = default_codec.encode(msg)

    tx = np.where(cw == 0, 1.0, -1.0)
    es_n0_linear = 12.0
    noise_sigma = 1.0 / math.sqrt(2.0 * es_n0_linear)
    rx = tx + rng.normal(0.0, noise_sigma, size=tx.shape)
    llrs = awgn_bpsk_llr(rx, es_n0_linear=es_n0_linear)

    decoded, stats = default_codec.decode(
        llrs, orig_size=orig_size, max_iter=30, algorithm=algorithm, reference_msg_bits=msg
    )
    assert np.array_equal(decoded, msg)
    assert stats["syndrome_failed_blocks"] == 0
    assert stats["msg_error_blocks"] == 0
    assert 1 <= stats["avg_iterations"] <= 30


@pytest.mark.parametrize("algorithm", ["sum-product", "min-sum", "normalized-min-sum"])
def test_zero_llr_input_stability_and_stats(default_codec: LDPCCodec, algorithm: str):
    """Zero LLR input: verifies no exceptions, valid output shape/values, and finite statistics."""
    zero_llrs = np.zeros(default_codec.n * 2, dtype=float)
    decoded, stats = default_codec.decode(
        zero_llrs, orig_size=14, max_iter=15, algorithm=algorithm
    )

    assert decoded.shape == (14,)
    assert decoded.dtype.kind in {"i", "u"}
    assert set(np.unique(decoded)).issubset({0, 1})
    assert stats["blocks"] == 2
    assert stats["algorithm"] == algorithm
    assert isinstance(stats["total_iterations"], int)
    assert 1 <= stats["total_iterations"] <= 2 * 15
    assert isinstance(stats["avg_iterations"], float) and math.isfinite(stats["avg_iterations"])
    assert isinstance(stats["bler"], float) and math.isfinite(stats["bler"])
    assert 0.0 <= stats["bler"] <= 1.0


def test_amc_selection_highest_eligible_spectral_efficiency(default_codec: LDPCCodec):
    """AMC selection: selects highest spectral efficiency among SNR-eligible modes."""
    bpsk_mode = MCSMode("BPSK_R0.5", mod_order=2, codec=default_codec, min_snr_db=-2.0)
    qpsk_mode = MCSMode("QPSK_R0.5", mod_order=4, codec=default_codec, min_snr_db=2.0)
    qam16_mode = MCSMode("16QAM_R0.5", mod_order=16, codec=default_codec, min_snr_db=8.0)

    H_high_rate = np.array([
        [1, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 1],
    ], dtype=int)
    codec_high_rate = LDPCCodec(H=H_high_rate)
    qpsk_r075 = MCSMode("QPSK_R0.75", mod_order=4, codec=codec_high_rate, min_snr_db=5.0)

    controller = AMCController([qam16_mode, qpsk_r075, qpsk_mode, bpsk_mode])

    assert controller.select_mode(6.0) == qpsk_r075
    assert controller.select_mode(9.0) == qam16_mode


# ---------------------------------------------------------------------------
# Additional Experiment: 100 Seeds Generation, Reproducibility & Diversity
# ---------------------------------------------------------------------------

@pytest.mark.experiment
def test_hundred_seeds_matrix_generation_consistency():
    """
    Generate matrices across 100 seeds for (n=14, k=7, w_c=3) and verify consistent
    success, regularity, zero 4-cycles, and full GF(2) rank.
    """
    n, k, w_c = 14, 7, 3
    m = n - k
    expected_wr = (n * w_c) // m

    for seed in range(100):
        H = generate_regular_ldpc_matrix(
            n=n,
            k=k,
            w_c=w_c,
            avoid_4cycles=True,
            require_full_rank=True,
            max_attempts=500,
            seed=seed,
        )
        assert H.shape == (m, n)
        assert np.all(np.sum(H, axis=0) == w_c)
        assert np.all(np.sum(H, axis=1) == expected_wr)
        assert _count_4cycles(H) == 0
        assert gf2_rank(H) == m


def test_matrix_generation_reproducibility_and_seed_variation():
    """Verify deterministic output for identical seeds and distinct matrices across seeds (generator check)."""
    H1 = generate_regular_ldpc_matrix(n=14, k=7, w_c=3, avoid_4cycles=True, require_full_rank=True, seed=42)
    H2 = generate_regular_ldpc_matrix(n=14, k=7, w_c=3, avoid_4cycles=True, require_full_rank=True, seed=42)
    H_other = generate_regular_ldpc_matrix(n=14, k=7, w_c=3, avoid_4cycles=True, require_full_rank=True, seed=43)

    assert np.array_equal(H1, H2)
    assert not np.array_equal(H1, H_other)


# ---------------------------------------------------------------------------
# Type & Validation Helpers (3.1 & 3.2)
# ---------------------------------------------------------------------------

def test_is_valid_int_cases():
    assert _is_valid_int(5) is True
    assert _is_valid_int(0) is True
    assert _is_valid_int(-3) is True
    assert _is_valid_int(np.int64(5)) is True
    assert _is_valid_int(True) is False
    assert _is_valid_int(False) is False
    assert _is_valid_int(5.0) is False
    assert _is_valid_int("5") is False
    assert _is_valid_int(None) is False


def test_is_finite_number_cases():
    assert _is_finite_number(5) is True
    assert _is_finite_number(0) is True
    assert _is_finite_number(-12) is True
    assert _is_finite_number(3.14) is True
    assert _is_finite_number(-2.5) is True
    assert _is_finite_number(np.float64(1.25)) is True
    assert _is_finite_number(float("nan")) is False
    assert _is_finite_number(float("inf")) is False
    assert _is_finite_number(float("-inf")) is False
    assert _is_finite_number(True) is False
    assert _is_finite_number(False) is False
    assert _is_finite_number("3.14") is False


# ---------------------------------------------------------------------------
# 2.1 GF(2) Linear Algebra & Matrix Generation Edge Cases
# ---------------------------------------------------------------------------

class TestGF2AndGenerationExtended:
    def test_count_4cycles_known_matrices(self):
        H_one_cycle = np.array([
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=int)
        assert _count_4cycles(H_one_cycle) == 1

        H_no_cycle = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ], dtype=int)
        assert _count_4cycles(H_no_cycle) == 0

    def test_gf2_rank_and_inv_edge_cases(self):
        assert gf2_rank(np.zeros((3, 4), dtype=int)) == 0
        assert gf2_rank(np.zeros((0, 0), dtype=int)) == 0
        assert gf2_rank(np.zeros((0, 3), dtype=int)) == 0
        assert gf2_rank(np.zeros((3, 0), dtype=int)) == 0
        assert gf2_rank(np.array([], dtype=int)) == 0
        assert gf2_rank(np.array([1, 0, 1], dtype=int)) == 0

        H_dep = np.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ], dtype=int)
        assert gf2_rank(H_dep) == 2

        with pytest.raises(ValueError, match="square"):
            gf2_inv(np.ones((2, 3), dtype=int))

        for mat in (np.ones((2, 2), dtype=int), np.array([[1, 0], [0, 0]], dtype=int)):
            with pytest.raises(ValueError, match="singular"):
                gf2_inv(mat)

        A_mod2 = np.array([[3, 2], [2, 3]], dtype=int)
        assert np.array_equal(gf2_inv(A_mod2), np.eye(2, dtype=int))

    def test_find_invertible_parity_cols_errors(self):
        rank_def_h = np.array([
            [1, 0, 1, 0],
            [1, 0, 1, 0],
        ], dtype=int)
        with pytest.raises(ValueError, match="full row rank"):
            _find_invertible_parity_cols(rank_def_h, k=2)

    def test_generate_regular_ldpc_matrix_constraints_and_failures(self):
        with pytest.raises(ValueError):
            generate_regular_ldpc_matrix(n=10, k=10, w_c=2)
        with pytest.raises(ValueError):
            generate_regular_ldpc_matrix(n=14, k=7, w_c=1)
        # Rejected by generator's (n-k) % w_c == 0 rule (7 % 4 != 0); note also that
        # 14 * C(4,2) = 84 check-pair incidences exceeds C(7,2) = 21 check pairs for 4-cycle-free graphs.
        with pytest.raises(ValueError):
            generate_regular_ldpc_matrix(n=14, k=7, w_c=4)
        with pytest.raises(ValueError):
            generate_regular_ldpc_matrix(n=14, k=7, w_c=3, max_attempts=0)

        with pytest.raises(RuntimeError, match="Failed to eliminate all 4-cycles"):
            generate_regular_ldpc_matrix(
                n=8, k=4, w_c=2, avoid_4cycles=True, require_full_rank=False, max_attempts=1, seed=1
            )


# ---------------------------------------------------------------------------
# 2.2 LDPC Encoder & Decoder Extended Tests
# ---------------------------------------------------------------------------

class TestLDPCEncoderDecoderExtended:
    def test_bit_flipping_specific_position(self, default_codec: LDPCCodec):
        """Functional check: test hard-decision bit-flipping on a single-bit error at index 0."""
        msg = np.array([1, 0, 1, 1, 0, 0, 1])
        cw, orig_size = default_codec.encode(msg)

        corrupted = cw.copy()
        corrupted[0] ^= 1
        dec, stats = default_codec.decode_bit_flipping(corrupted, orig_size=orig_size, max_iter=30)

        assert np.array_equal(dec, msg)
        assert stats["failed_blocks"] == 0

    @staticmethod
    def _assert_valid_audit_entry(
        entry: tuple[int, bool, int],
        expected_n: int,
    ) -> None:
        idx, recovered, failed_blocks = entry
        assert isinstance(idx, int) and 0 <= idx < expected_n
        assert isinstance(recovered, bool)
        assert isinstance(failed_blocks, int) and failed_blocks >= 0

    @pytest.mark.experiment
    def test_bit_flipping_all_single_bit_positions_audit(self, default_codec: LDPCCodec):
        """
        Exploratory experiment: record per-position bit-flipping outcomes across indices 0..n-1.
        Does not assert universal single-error correction, as recovery depends on Tanner graph
        structure and the bit-flipping decision rule.
        """
        msg = np.array([1, 0, 1, 1, 0, 0, 1])
        cw, orig_size = default_codec.encode(msg)

        audit = []
        for idx in range(default_codec.n):
            corrupted = cw.copy()
            corrupted[idx] ^= 1
            dec, stats = default_codec.decode_bit_flipping(corrupted, orig_size=orig_size, max_iter=30)
            entry = (idx, bool(np.array_equal(dec, msg)), int(stats["failed_blocks"]))
            self._assert_valid_audit_entry(entry, default_codec.n)
            audit.append(entry)

        assert len(audit) == default_codec.n

    def test_noiseless_all_zero_and_all_one_messages_multi_block(self, default_codec: LDPCCodec):
        """Noiseless BPSK LLR decoding of all-zero and all-one messages across 3 blocks."""
        for bit_val in (0, 1):
            msg = np.full(21, bit_val, dtype=int)
            cw, orig_size = default_codec.encode(msg)
            tx_noiseless = np.where(cw == 0, 1.0, -1.0)
            llrs = awgn_bpsk_llr(tx_noiseless, es_n0_linear=8.0)
            dec, stats = default_codec.decode(llrs, orig_size=orig_size, reference_msg_bits=msg)
            assert np.array_equal(dec, msg)
            assert stats["blocks"] == 3
            assert stats["msg_error_blocks"] == 0

    def test_syndrome_zero_vs_message_error_distinction(self):
        """
        Verify that H @ z^T == 0 (valid syndrome) does not guarantee z == c
        when conflicting LLRs push the decoder toward a different valid codeword.
        """
        H = np.array([
            [1, 1, 1, 0],
            [0, 1, 1, 1],
        ], dtype=int)
        codec = LDPCCodec(H=H)

        msg_target = np.array([0, 0])
        _, orig_size = codec.encode(msg_target)

        msg_alt = np.array([1, 1])
        alt_cw, _ = codec.encode(msg_alt)
        assert np.all(((alt_cw.reshape(1, -1) @ H.T) % 2) == 0)

        conflicting_llrs = np.where(alt_cw == 0, 20.0, -20.0).astype(float)
        dec, stats = codec.decode(
            conflicting_llrs, orig_size=orig_size, max_iter=10, reference_msg_bits=msg_target
        )

        assert stats["syndrome_failed_blocks"] == 0
        assert stats["msg_error_blocks"] == 1
        assert np.array_equal(dec, msg_alt)

    @pytest.mark.experiment
    @pytest.mark.parametrize("snr_db,expect_low_error", [
        (-4.0, False),
        (2.0, True),
        (10.0, True),
    ])
    def test_decoder_snr_sweep_experiment(
        self, default_codec: LDPCCodec, snr_db: float, expect_low_error: bool, capsys
    ):
        """
        Empirical experiment: multi-block decoder BER/BLER across low, moderate, and high SNR.
        Reports measured BER and BLER alongside empirical regression bounds.
        """
        num_blocks = 200
        msg_rng = np.random.default_rng(500)
        messages = [msg_rng.integers(0, 2, size=default_codec.k) for _ in range(num_blocks)]
        msg = np.concatenate(messages)
        cw, orig_size = default_codec.encode(msg)

        noise_rng = np.random.default_rng(505)
        tx = np.where(cw == 0, 1.0, -1.0)
        es_n0 = 10.0 ** (snr_db / 10.0)
        sigma = 1.0 / math.sqrt(2.0 * es_n0)
        rx = tx + noise_rng.normal(0.0, sigma, size=tx.shape)
        llrs = awgn_bpsk_llr(rx, es_n0_linear=es_n0)

        dec, stats = default_codec.decode(
            llrs, orig_size=orig_size, max_iter=25, algorithm="normalized-min-sum", reference_msg_bits=msg
        )
        ber = float(np.mean(dec != msg))
        bler = float(stats["bler"])

        print(
            f"[SNR={snr_db:5.1f} dB | blocks={num_blocks} | algo={stats['algorithm']}] "
            f"measured_BER={ber:.6f}, measured_BLER={bler:.6f}, avg_iters={stats['avg_iterations']:.2f}"
        )
        captured = capsys.readouterr().out
        assert f"SNR={snr_db:5.1f} dB" in captured

        if expect_low_error:
            assert ber < 0.05
            assert bler < 0.15
        else:
            assert ber > 0.05

    def test_early_stopping_on_zero_llr(self, default_codec: LDPCCodec):
        """Zero LLRs yield all-zero hard decisions with zero syndrome on iteration 1 -> stops in 1 iter."""
        zero_llrs = np.zeros(default_codec.n, dtype=float)
        _, stats_1 = default_codec.decode(zero_llrs, orig_size=7, max_iter=1)
        _, stats_5 = default_codec.decode(zero_llrs, orig_size=7, max_iter=5)

        assert stats_1["total_iterations"] == 1
        assert stats_5["total_iterations"] == 1

    def test_max_iter_cap_and_invalid_parameters(self, default_codec: LDPCCodec):
        """Verify max_iter cap on a multi-iteration LLR input and validate bad parameters."""
        # Flipping bit 12 of an all-zero codeword produces check parity mismatches on
        # rows 2, 5, and 6 of default_codec.H; verify empirically that max_iter=1 runs
        # 1 iteration while max_iter=4 executes multiple iterations (2..4).
        msg = np.zeros(7, dtype=int)
        cw, orig_size = default_codec.encode(msg)
        cw_flipped = cw.copy()
        cw_flipped[12] ^= 1
        llrs_flipped = np.where(cw_flipped == 0, 10.0, -10.0).astype(float)

        _, stats_cap_1 = default_codec.decode(llrs_flipped, orig_size=orig_size, max_iter=1)
        _, stats_cap_4 = default_codec.decode(llrs_flipped, orig_size=orig_size, max_iter=4)
        assert stats_cap_1["total_iterations"] == 1
        assert 2 <= stats_cap_4["total_iterations"] <= 4

        with pytest.raises(ValueError, match="algorithm"):
            default_codec.decode(llrs_flipped, orig_size=orig_size, algorithm="unknown-algo")
        with pytest.raises(ValueError, match="max_iter"):
            default_codec.decode(llrs_flipped, orig_size=orig_size, max_iter=0)
        with pytest.raises(ValueError, match="max_iter"):
            default_codec.decode(llrs_flipped, orig_size=orig_size, max_iter=True)
        with pytest.raises(ValueError, match="H must be a 2D binary matrix"):
            LDPCCodec(H=np.array([[0.5, 1.0], [1.0, 0.0]]))


# ---------------------------------------------------------------------------
# 2.3 & 3.3 Channel & Modulation Analytical and BSC Tests
# ---------------------------------------------------------------------------

class TestChannelAndModulationExtended:
    def test_bsc_llr_values_and_invalid_p(self):
        """Test BSC LLR sign/magnitude convention: +ln((1-p)/p) for y=0, -ln((1-p)/p) for y=1."""
        bits = np.array([0, 1, 0, 1])
        p = 0.1
        llrs = bsc_llr(bits, crossover_prob=p, llr_clip=50.0)
        expected_mag = math.log((1.0 - p) / p)
        assert np.allclose(llrs, np.array([expected_mag, -expected_mag, expected_mag, -expected_mag]))

        # Near-zero crossover probability clips to +/- llr_clip
        llrs_near_zero = bsc_llr(bits, crossover_prob=1e-40, llr_clip=25.0)
        assert np.allclose(llrs_near_zero, np.array([25.0, -25.0, 25.0, -25.0]))

        # Coding.py requires strictly 0 < p < 0.5; p=0.5, p=0, p=1, negative, >1, NaN, inf, bool raise ValueError
        for invalid_p in (0.0, 0.5, 1.0, -0.2, 1.5, float("nan"), float("inf"), True, False):
            with pytest.raises(ValueError):
                bsc_llr(bits, crossover_prob=invalid_p)

    def test_awgn_bpsk_llr_exact_scaling_and_clipping(self):
        y = np.array([0.5, -0.5, 1.25])
        llrs = awgn_bpsk_llr(y, es_n0_linear=2.0, llr_clip=50.0)
        assert np.allclose(llrs, np.array([4.0, -4.0, 10.0]))

        clipped = awgn_bpsk_llr(np.array([5.0, -5.0]), es_n0_linear=4.0, llr_clip=10.0)
        assert np.allclose(clipped, np.array([10.0, -10.0]))

    def test_awgn_bpsk_llr_with_real_gaussian_noise_convention(self):
        """Test awgn_bpsk_llr when noise n_i ~ N(0, 1 / (2 * Es/N0)) is added to BPSK symbols."""
        rng = np.random.default_rng(2026)
        es_n0 = 4.0
        var_per_dim = 1.0 / (2.0 * es_n0)
        tx = np.ones(2000, dtype=float)
        rx = tx + rng.normal(0.0, math.sqrt(var_per_dim), size=tx.shape)
        llrs = awgn_bpsk_llr(rx, es_n0_linear=es_n0, llr_clip=100.0)
        assert math.isclose(float(np.mean(llrs)), 4.0 * es_n0, rel_tol=0.05)

    @pytest.mark.parametrize("mod_order", [2, 4, 16])
    def test_qam_soft_llr_analytical_and_sign_consistency(self, mod_order: int):
        syms, bit_map = build_constellation(mod_order)
        noise_var = 0.25
        h = 1.0 + 0.0j

        for idx, sym in enumerate(syms):
            bits = bit_map[idx]
            llrs = qam_soft_llr(np.array([sym]), mod_order=mod_order, noise_var_per_dim=noise_var, channel_gain=h)
            inferred_bits = np.where(llrs > 0.0, 0, 1)
            assert np.array_equal(inferred_bits, bits)

            bps = bit_map.shape[1]
            metrics = -np.abs(sym - syms) ** 2 / (2.0 * noise_var)
            for q in range(bps):
                m0 = bit_map[:, q] == 0
                m1 = bit_map[:, q] == 1
                expected_llr = float(_logsumexp(metrics[m0]) - _logsumexp(metrics[m1]))
                assert math.isclose(float(llrs[q]), expected_llr, rel_tol=1e-9, abs_tol=1e-9)

    @staticmethod
    def _expected_log_map_llrs(
        y: complex,
        syms: np.ndarray,
        bit_map: np.ndarray,
        noise_var_per_dim: float,
        h: complex,
    ) -> np.ndarray:
        metrics = -np.abs(y - h * syms) ** 2 / (2.0 * noise_var_per_dim)
        bps = bit_map.shape[1]
        expected = np.empty(bps, dtype=float)
        for q in range(bps):
            m0 = bit_map[:, q] == 0
            m1 = bit_map[:, q] == 1
            expected[q] = float(_logsumexp(metrics[m0]) - _logsumexp(metrics[m1]))
        return expected

    def test_qam_soft_llr_channel_gain_cases(self):
        syms, bit_map = build_constellation(4)
        # Zero channel gain -> equal metric for all symbols -> all LLRs == 0.0
        llrs_zero_h = qam_soft_llr(np.array([1.0 + 1.0j]), mod_order=4, noise_var_per_dim=0.1, channel_gain=0.0)
        assert np.allclose(llrs_zero_h, np.zeros(2))

        # Non-unit complex channel gain -> verify exact log-MAP magnitudes and signs
        h_complex = -0.6 + 0.8j
        noise_var = 0.05
        rx = h_complex * syms[0]
        llrs_c = qam_soft_llr(rx, mod_order=4, noise_var_per_dim=noise_var, channel_gain=h_complex)
        expected_c = self._expected_log_map_llrs(rx, syms, bit_map, noise_var, h_complex)
        assert np.allclose(llrs_c, expected_c, rtol=1e-9, atol=1e-9)
        assert np.array_equal(np.where(llrs_c > 0.0, 0, 1), bit_map[0])

        # Incompatible channel gain shape
        with pytest.raises(ValueError):
            qam_soft_llr(
                np.array([syms[0], syms[1]]),
                mod_order=4,
                noise_var_per_dim=0.1,
                channel_gain=np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j]),
            )

    def test_constellation_geometry_gray_coding_and_min_distance(self):
        syms16, bits16 = build_constellation(16)
        min_step = min(
            abs(syms16[i] - syms16[j])
            for i in range(len(syms16))
            for j in range(i + 1, len(syms16))
            if abs(syms16[i] - syms16[j]) > 0
        )
        expected_min_dist = 2.0 / math.sqrt(10.0)
        assert math.isclose(min_step, expected_min_dist, rel_tol=1e-12)

        for i in range(len(syms16)):
            for j in range(i + 1, len(syms16)):
                if math.isclose(abs(syms16[i] - syms16[j]), min_step, rel_tol=1e-9):
                    hamming = int(np.sum(bits16[i] != bits16[j]))
                    assert hamming == 1

        for order in (4, 16):
            s, _ = build_constellation(order)
            assert np.allclose(np.mean(s), 0.0 + 0.0j, atol=1e-12)
            assert math.isclose(float(np.mean(np.real(s) ** 2)), 0.5, rel_tol=1e-12)
            assert math.isclose(float(np.mean(np.imag(s) ** 2)), 0.5, rel_tol=1e-12)

    @pytest.mark.experiment
    def test_simulate_amc_block_awgn_high_snr_all_zero_stream(self, default_codec: LDPCCodec):
        """High-SNR AWGN end-to-end recovery of an all-zero multi-block stream via simulate_amc_block."""
        mode = MCSMode("BPSK_R0.5", mod_order=2, codec=default_codec, min_snr_db=-2.0)
        rng = np.random.default_rng(909)
        all_zero_msg = np.zeros(default_codec.k * 50, dtype=int)

        recovered, info = simulate_amc_block(
            all_zero_msg, mode, snr_db=14.0, fading="awgn", max_iter=25, algorithm="sum-product", rng=rng
        )
        assert np.array_equal(recovered, all_zero_msg)
        assert info["bit_errors"] == 0
        assert info["ber"] == 0.0
        assert info["bler"] == 0.0

    @pytest.mark.parametrize("algorithm", ["sum-product", "min-sum", "normalized-min-sum"])
    def test_partial_message_multi_codeword_16qam_padding_recovery(
        self, default_codec: LDPCCodec, algorithm: str
    ):
        """
        End-to-end partial message (not a multiple of k) across multiple LDPC codewords
        combined with 16-QAM bit padding (not a multiple of 4 coded bits):
        encode -> multi-codeword stream -> 16-QAM pad -> modulate -> soft LLR demap ->
        strip modulation pad -> decode to orig_size -> verify exact message recovery.
        """
        rng = np.random.default_rng(616)
        # k = 7, n = 14: 15 message bits -> 3 blocks (21 bits, 6 message-pad bits) ->
        # 3 codewords = 42 coded bits -> 42 % 4 == 2 -> 2 QAM pad bits -> 44 bits (11 16-QAM symbols).
        msg = rng.integers(0, 2, size=15)
        cw, orig_size = default_codec.encode(msg)
        assert orig_size == 15
        assert cw.size == 3 * default_codec.n  # 42 bits across 3 codewords

        bps = 4  # 16-QAM
        qam_pad = (bps - (cw.size % bps)) % bps
        assert qam_pad == 2

        padded_cw = np.pad(cw, (0, qam_pad), mode="constant")
        assert padded_cw.size == 44

        syms, bit_map = build_constellation(16)
        bit_to_idx = {tuple(row.tolist()): idx for idx, row in enumerate(bit_map)}
        sym_indices = [
            bit_to_idx[tuple(padded_cw[i : i + bps].tolist())]
            for i in range(0, padded_cw.size, bps)
        ]
        tx_syms = syms[sym_indices]

        all_llrs = qam_soft_llr(tx_syms, mod_order=16, noise_var_per_dim=1e-4)
        assert all_llrs.size == padded_cw.size

        # Remove only the modulation padding, keeping all 42 coded bits for decoding
        coded_llrs = all_llrs[: cw.size]
        assert coded_llrs.size == cw.size

        decoded, stats = default_codec.decode(
            coded_llrs,
            orig_size=orig_size,
            max_iter=25,
            algorithm=algorithm,
            reference_msg_bits=msg,
        )

        assert decoded.shape == (15,)
        assert np.array_equal(decoded, msg)
        assert stats["blocks"] == 3
        assert stats["msg_error_blocks"] == 0
        assert stats["syndrome_failed_blocks"] == 0
        assert stats["bler"] == 0.0

# ---------------------------------------------------------------------------
# 2.4 AMC Controller Extended Tests
# ---------------------------------------------------------------------------

class TestAMCControllerExtended:
    def test_threshold_boundaries_and_fallback(self, amc_controller: AMCController):
        assert amc_controller.select_mode(2.0).name == "QPSK_R0.5"
        assert amc_controller.select_mode(2.0 - 1e-9).name == "BPSK_R0.5"
        assert amc_controller.select_mode(8.0).name == "16QAM_R0.5"
        assert amc_controller.select_mode(8.0 - 1e-9).name == "QPSK_R0.5"
        assert amc_controller.select_mode(-50.0).name == "BPSK_R0.5"

    def test_equal_spectral_efficiency_tie_breaking(self, default_codec: LDPCCodec):
        m_low_thresh = MCSMode("BPSK_LowThresh", mod_order=2, codec=default_codec, min_snr_db=0.0)
        m_high_thresh = MCSMode("BPSK_HighThresh", mod_order=2, codec=default_codec, min_snr_db=3.0)

        ctrl = AMCController([m_high_thresh, m_low_thresh])
        assert ctrl.select_mode(5.0) == m_low_thresh

    def test_mcs_target_bler_validation(self, default_codec: LDPCCodec):
        mode = MCSMode("QPSK_R0.5", mod_order=4, codec=default_codec, min_snr_db=2.0, target_bler=1e-3)
        assert mode.target_bler == 1e-3
        for bad_bler in (0.0, 1.0, -0.1, 1.2, float("nan"), True):
            with pytest.raises(ValueError, match="target_bler"):
                MCSMode("BadBLER", mod_order=4, codec=default_codec, min_snr_db=2.0, target_bler=bad_bler)


# ---------------------------------------------------------------------------
# 1.7, 2.2 & 2.5 End-to-End Simulation: Deterministic & Statistical Tests
# ---------------------------------------------------------------------------

class TestEndToEndSimulation:
    def test_deterministic_high_snr_awgn_recovers_message(self, amc_controller: AMCController):
        """Level A: deterministic high-SNR AWGN transmission recovers exact message."""
        rng = np.random.default_rng(1234)
        msg = rng.integers(0, 2, size=28)
        snr_db = 16.0
        mcs = amc_controller.select_mode(snr_db)

        recovered, info = simulate_amc_block(
            msg_bits=msg,
            mcs=mcs,
            snr_db=snr_db,
            fading="awgn",
            max_iter=25,
            algorithm="sum-product",
            rng=rng,
        )
        assert np.array_equal(recovered, msg)
        assert info["bit_errors"] == 0
        assert info["ber"] == 0.0
        assert info["bler"] == 0.0

    @pytest.mark.experiment
    def test_paired_snr_common_random_numbers_multi_seed(self, default_codec: LDPCCodec, capsys):
        """
        Paired CRN experiment across multiple seeds: reuses unit-variance noise n
        across low and high SNR via y = x + sigma * n, then reports BER/BLER across seeds.
        """
        num_blocks = 200
        snr_low_db = -2.0
        snr_high_db = 8.0
        sigma_low = math.sqrt(1.0 / (2.0 * (10.0 ** (snr_low_db / 10.0))))
        sigma_high = math.sqrt(1.0 / (2.0 * (10.0 ** (snr_high_db / 10.0))))

        seeds = [7100, 7101, 7102]
        low_bers, high_bers = [], []
        low_blers, high_blers = [], []

        for seed in seeds:
            rng = np.random.default_rng(seed)
            msg = rng.integers(0, 2, size=num_blocks * default_codec.k)
            cw, orig_size = default_codec.encode(msg)
            tx = np.where(cw == 0, 1.0, -1.0)
            # Unit-variance real noise n ~ N(0, 1) scaled by sigma for each SNR
            n = rng.normal(0.0, 1.0, size=tx.shape)

            rx_low = tx + sigma_low * n
            rx_high = tx + sigma_high * n

            llrs_low = awgn_bpsk_llr(rx_low, es_n0_linear=10.0 ** (snr_low_db / 10.0))
            llrs_high = awgn_bpsk_llr(rx_high, es_n0_linear=10.0 ** (snr_high_db / 10.0))

            dec_low, stats_low = default_codec.decode(
                llrs_low, orig_size=orig_size, max_iter=25, algorithm="sum-product", reference_msg_bits=msg
            )
            dec_high, stats_high = default_codec.decode(
                llrs_high, orig_size=orig_size, max_iter=25, algorithm="sum-product", reference_msg_bits=msg
            )

            ber_l = float(np.mean(dec_low != msg))
            ber_h = float(np.mean(dec_high != msg))
            bler_l = float(stats_low["bler"])
            bler_h = float(stats_high["bler"])

            low_bers.append(ber_l)
            high_bers.append(ber_h)
            low_blers.append(bler_l)
            high_blers.append(bler_h)

            print(
                f"[seed={seed}] SNR={snr_low_db:4.1f}dB: BER={ber_l:.5f}, BLER={bler_l:.5f} | "
                f"SNR={snr_high_db:4.1f}dB: BER={ber_h:.5f}, BLER={bler_h:.5f}"
            )

        out = capsys.readouterr().out
        assert len(low_bers) == len(seeds)
        print(
            f"[mean across {len(seeds)} seeds] "
            f"low_BER={np.mean(low_bers):.5f}, high_BER={np.mean(high_bers):.5f} | "
            f"low_BLER={np.mean(low_blers):.5f}, high_BLER={np.mean(high_blers):.5f}"
        )
        assert "mean across 3 seeds" in capsys.readouterr().out or "mean across 3 seeds" in out

    def test_rayleigh_fading_simulation_runs_and_reports_metrics(self, amc_controller: AMCController):
        """
        End-to-end execution test with fading='rayleigh' (inline per-symbol Rayleigh gain in
        simulate_amc_block; standalone fading/noise generation is not exposed separately in coding.py).
        """
        rng = np.random.default_rng(8080)
        msg = rng.integers(0, 2, size=28)
        snr_db = 12.0
        mcs = amc_controller.select_mode(snr_db)

        recovered, info = simulate_amc_block(
            msg_bits=msg,
            mcs=mcs,
            snr_db=snr_db,
            fading="rayleigh",
            max_iter=25,
            algorithm="normalized-min-sum",
            rng=rng,
        )
        assert recovered.shape == msg.shape
        assert info["mcs"] == mcs.name
        assert 0.0 <= info["ber"] <= 1.0
        assert 0.0 <= info["bler"] <= 1.0
