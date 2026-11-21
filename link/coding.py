import math
from dataclasses import dataclass
import numpy as np


def _is_valid_int(val) -> bool:
    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


def _is_finite_number(val) -> bool:
    return isinstance(val, (int, float, np.number)) and not isinstance(val, bool) and math.isfinite(float(val))


def _logsumexp(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(arr, axis=axis, keepdims=True)
    res = m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))
    return np.squeeze(res, axis=axis)


def gf2_rank(mat: np.ndarray) -> int:
    """Compute matrix rank over GF(2) using Gaussian elimination."""
    A = (np.asarray(mat, dtype=int) & 1).copy()
    if A.ndim != 2 or A.size == 0:
        return 0
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if A[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(rows):
            if i != r and A[i, c] == 1:
                A[i] ^= A[r]
        r += 1
        if r == rows:
            break
    return r


def gf2_inv(mat: np.ndarray) -> np.ndarray:
    """Compute inverse of a square binary matrix over GF(2), or raise ValueError if singular."""
    A = (np.asarray(mat, dtype=int) & 1).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for GF(2) inversion.")
    n = A.shape[0]
    aug = np.concatenate([A, np.eye(n, dtype=int)], axis=1)

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is singular over GF(2) and cannot be inverted.")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(n):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]

    return aug[:, n:]


def _find_invertible_parity_cols(H: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given m x n binary matrix H with rank m and information dimension k = n - m,
    partition columns into k systematic info columns and m invertible parity columns P
    such that P is invertible over GF(2).
    Returns (info_cols, parity_cols, P_inv).
    """
    m, n = H.shape
    if gf2_rank(H) != m:
        raise ValueError("Parity-check matrix H is not full row rank over GF(2).")

    aug = np.concatenate([H.copy(), np.eye(m, dtype=int)], axis=1)
    pivot_cols: list[int] = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        pivot = None
        for i in range(r, m):
            if aug[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            aug[[r, pivot]] = aug[[pivot, r]]
        for i in range(m):
            if i != r and aug[i, c] == 1:
                aug[i] ^= aug[r]
        pivot_cols.append(c)
        r += 1

    if len(pivot_cols) != m:
        raise ValueError("Unable to find m linearly independent parity columns over GF(2).")

    parity_cols = np.array(pivot_cols, dtype=int)
    is_parity = np.zeros(n, dtype=bool)
    is_parity[parity_cols] = True
    info_cols = np.flatnonzero(~is_parity)
    if info_cols.size != k:
        raise ValueError("Systematic column partition mismatch.")

    P = H[:, parity_cols] & 1
    P_inv = gf2_inv(P)
    return info_cols, parity_cols, P_inv


def awgn_bpsk_llr(received_symbols: np.ndarray, es_n0_linear: float, llr_clip: float = 50.0) -> np.ndarray:
    """
    Compute a priori LLRs r_i = ln(P(c_i=0)/P(c_i=1)) for BPSK mapping:
    bit 0 -> +1, bit 1 -> -1 over AWGN (r_i = 4 * y_i * (Es / N0)).
    """
    if not _is_finite_number(es_n0_linear) or float(es_n0_linear) <= 0.0:
        raise ValueError("es_n0_linear must be a positive finite number.")
    if not _is_finite_number(llr_clip) or float(llr_clip) <= 0.0:
        raise ValueError("llr_clip must be a positive finite number.")
    y = np.asarray(received_symbols, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("received_symbols must contain only finite values.")
    return np.clip(4.0 * y * float(es_n0_linear), -float(llr_clip), float(llr_clip))


def bsc_llr(received_bits: np.ndarray, crossover_prob: float, llr_clip: float = 50.0) -> np.ndarray:
    """
    Compute a priori LLRs for a Binary Symmetric Channel with crossover probability p.
    Strictly validates binary 0/1 inputs (excluding booleans) and finite parameters.
    """
    if not _is_finite_number(crossover_prob):
        raise ValueError("crossover_prob must be a finite number.")
    p = float(crossover_prob)
    if not (0.0 < p < 0.5):
        raise ValueError("crossover_prob must be strictly between 0 and 0.5.")
    if not _is_finite_number(llr_clip) or float(llr_clip) <= 0.0:
        raise ValueError("llr_clip must be a positive finite number.")

    y = np.asarray(received_bits)
    if y.size == 0:
        raise ValueError("received_bits cannot be empty.")
    if not np.all(np.isin(y, [0, 1])) or np.issubdtype(y.dtype, np.bool_):
        raise ValueError("received_bits must contain strictly binary values (0 or 1), excluding booleans.")

    llr_zero = float(np.clip(np.log((1.0 - p) / p), -float(llr_clip), float(llr_clip)))
    return np.where(y.astype(int) == 0, llr_zero, -llr_zero)


def build_constellation(mod_order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (constellation_symbols, bit_map) with unit average symbol energy E[|s|^2] = 1.
    bit_map has shape (M, log2(M)) with entries in {0, 1}.
    """
    if mod_order == 2:
        syms = np.array([1.0 + 0.0j, -1.0 + 0.0j])
        bits = np.array([[0], [1]], dtype=int)
        return syms, bits

    if mod_order == 4:
        scale = 1.0 / np.sqrt(2.0)
        vals = np.array([1.0, -1.0]) * scale
        syms = []
        bits = []
        for b0, r in enumerate(vals):
            for b1, im in enumerate(vals):
                syms.append(complex(r, im))
                bits.append([b0, b1])
        return np.asarray(syms, dtype=complex), np.asarray(bits, dtype=int)

    if mod_order == 16:
        pam_levels = np.array([3.0, 1.0, -1.0, -3.0]) / np.sqrt(10.0)
        pam_bits = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
        ], dtype=int)
        syms = []
        bits = []
        for i_idx, re in enumerate(pam_levels):
            for q_idx, im in enumerate(pam_levels):
                syms.append(complex(re, im))
                bits.append(np.concatenate([pam_bits[i_idx], pam_bits[q_idx]]))
        return np.asarray(syms, dtype=complex), np.asarray(bits, dtype=int)

    raise ValueError("Unsupported mod_order; use 2 (BPSK), 4 (QPSK), or 16 (16-QAM).")


def qam_soft_llr(
    rx_symbols: np.ndarray,
    mod_order: int,
    noise_var_per_dim: float,
    channel_gain: np.ndarray | complex | float = 1.0,
    llr_clip: float = 50.0,
) -> np.ndarray:
    """
    Exact log-MAP soft LLR demapper for BPSK / QPSK / 16-QAM over AWGN or flat fading.
    """
    if not _is_finite_number(noise_var_per_dim) or float(noise_var_per_dim) <= 0.0:
        raise ValueError("noise_var_per_dim must be a positive finite number.")
    if not _is_finite_number(llr_clip) or float(llr_clip) <= 0.0:
        raise ValueError("llr_clip must be a positive finite number.")

    y = np.asarray(rx_symbols, dtype=complex)
    if not np.all(np.isfinite(y)):
        raise ValueError("rx_symbols must contain only finite values.")
    h = np.broadcast_to(np.asarray(channel_gain, dtype=complex), y.shape)
    if not np.all(np.isfinite(h)):
        raise ValueError("channel_gain must contain only finite values.")

    syms, bit_map = build_constellation(mod_order)
    bps = bit_map.shape[1]

    exp_sym = h[..., None] * syms[None, :]
    metrics = -np.abs(y[..., None] - exp_sym) ** 2 / (2.0 * float(noise_var_per_dim))

    llrs = np.empty(y.shape + (bps,), dtype=float)
    for q in range(bps):
        mask0 = bit_map[:, q] == 0
        mask1 = bit_map[:, q] == 1
        num = _logsumexp(metrics[..., mask0], axis=-1)
        den = _logsumexp(metrics[..., mask1], axis=-1)
        llrs[..., q] = np.clip(num - den, -float(llr_clip), float(llr_clip))

    return llrs.reshape(-1)


def _count_4cycles(H: np.ndarray) -> int:
    """Count number of column pairs sharing >= 2 ones (4-cycles in Tanner graph)."""
    _, n = H.shape
    cycles = 0
    for c1 in range(n - 1):
        col1 = H[:, c1] == 1
        overlaps = (H[:, c1 + 1 :] == 1) & col1[:, None]
        counts = np.sum(overlaps, axis=0)
        cycles += int(np.sum(counts >= 2))
    return cycles


def generate_regular_ldpc_matrix(
    n: int,
    k: int,
    w_c: int = 3,
    avoid_4cycles: bool = True,
    require_full_rank: bool = True,
    max_attempts: int = 500,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a (w_c, w_r)-regular sparse parity-check matrix H of shape (n - k, n)
    using degree-preserving 2x2 edge switches to eliminate 4-cycles while preserving
    exact row and column degrees, and verifying 4-cycle freedom, regularity, and GF(2) rank.
    """
    if not (_is_valid_int(n) and _is_valid_int(k) and _is_valid_int(w_c)):
        raise ValueError("n, k, and w_c must be integers.")
    if not _is_valid_int(max_attempts) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer.")

    m = n - k
    if m <= 0 or k <= 0:
        raise ValueError("Require 0 < k < n.")
    if w_c < 2 or m % w_c != 0:
        raise ValueError("Require w_c >= 2 and (n - k) divisible by w_c.")

    w_r_num = n * w_c
    if w_r_num % m != 0:
        raise ValueError("Require n * w_c divisible by (n - k) for regular row weight.")
    w_r = w_r_num // m
    rows_per_group = m // w_c
    if n % rows_per_group != 0:
        raise ValueError("Require n divisible by (n - k) // w_c.")
    if (n // rows_per_group) < w_r:
        raise ValueError("Insufficient columns per row in banded group.")

    rng = np.random.default_rng(seed)
    base_group = np.zeros((rows_per_group, n), dtype=int)
    for r_idx in range(rows_per_group):
        start_col = (r_idx * w_r) % n
        for offset in range(w_r):
            base_group[r_idx, (start_col + offset) % n] = 1

    groups = [base_group.copy()]
    for _ in range(1, w_c):
        perm = rng.permutation(n)
        groups.append(base_group[:, perm])

    H = np.vstack(groups)

    if avoid_4cycles:
        for _ in range(max_attempts):
            conflicted_cols: list[int] = []
            for c1 in range(n - 1):
                for c2 in range(c1 + 1, n):
                    if int(np.sum((H[:, c1] == 1) & (H[:, c2] == 1))) >= 2:
                        conflicted_cols.extend([c1, c2])
            if not conflicted_cols:
                break

            c_target = int(rng.choice(conflicted_cols))
            ones_in_target = np.flatnonzero(H[:, c_target] == 1)
            zeros_in_target = np.flatnonzero(H[:, c_target] == 0)
            rng.shuffle(ones_in_target)
            rng.shuffle(zeros_in_target)

            switched = False
            for r1 in ones_in_target:
                zero_cols_in_r1 = np.flatnonzero(H[r1, :] == 0)
                zero_cols_in_r1 = zero_cols_in_r1[zero_cols_in_r1 != c_target]
                rng.shuffle(zero_cols_in_r1)
                for c_other in zero_cols_in_r1:
                    for r2 in zeros_in_target:
                        if r2 != r1 and H[r2, c_other] == 1:
                            H[r1, c_target] = 0
                            H[r1, c_other] = 1
                            H[r2, c_other] = 0
                            H[r2, c_target] = 1
                            switched = True
                            break
                    if switched:
                        break
                if switched:
                    break

        remaining_4cycles = _count_4cycles(H)
        if remaining_4cycles > 0:
            raise RuntimeError(
                f"Failed to eliminate all 4-cycles after {max_attempts} attempts "
                f"({remaining_4cycles} column-pair conflicts remain)."
            )

    if not np.all(np.sum(H, axis=0) == w_c):
        raise RuntimeError("Generated matrix violated column weight regularity.")
    if not np.all(np.sum(H, axis=1) == w_r):
        raise RuntimeError("Generated matrix violated row weight regularity.")
    if require_full_rank and gf2_rank(H) != m:
        raise RuntimeError("Generated matrix does not have full row rank over GF(2).")

    return H


class LDPCCodec:
    """
    Systematic linear block / LDPC codec supporting Sum-Product (LLR BP),
    Min-Sum, Normalized Min-Sum, and Gallager Bit-Flipping decoding.
    """

    def __init__(self, H: np.ndarray | None = None, n: int = 14, k: int = 7):
        if H is None:
            if not _is_valid_int(n) or not _is_valid_int(k):
                raise ValueError("Codec dimensions n and k must be integers.")
            if n != 14 or k != 7:
                raise ValueError("Default matrix only supports n=14, k=7; pass H for custom dimensions.")
            self.n = n
            self.k = k
            self.H = np.array([
                [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ], dtype=int)
        else:
            H_arr = np.asarray(H)
            if H_arr.ndim != 2 or not np.all(np.isin(H_arr, [0, 1])) or np.issubdtype(H_arr.dtype, np.bool_):
                raise ValueError("H must be a 2D binary matrix containing strictly 0 and 1.")
            _, n_cols = H_arr.shape
            rank_h = gf2_rank(H_arr)
            if rank_h == 0 or n_cols <= rank_h:
                raise ValueError("Invalid H matrix rank over GF(2).")
            self.H = H_arr.astype(int)
            self.n = n_cols
            self.k = n_cols - rank_h

        self.m = self.H.shape[0]
        self.rank = gf2_rank(self.H)
        self.B = [np.flatnonzero(self.H[j, :]) for j in range(self.m)]
        self.A = [np.flatnonzero(self.H[:, i]) for i in range(self.n)]

        if self.rank == self.m:
            self._info_cols, self._parity_cols, self._P_inv = _find_invertible_parity_cols(self.H, self.k)
            self._A_info = self.H[:, self._info_cols] & 1
            self._assemble_order = np.concatenate([self._info_cols, self._parity_cols])
            self._inv_assemble_order = np.argsort(self._assemble_order)
        else:
            self._info_cols = None
            self._parity_cols = None
            self._P_inv = None
            self._A_info = None
            self._assemble_order = None
            self._inv_assemble_order = None

    def _validate_bits(self, bits: np.ndarray, name: str = "Input") -> np.ndarray:
        arr = np.asarray(bits)
        if arr.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if not np.all(np.isin(arr, [0, 1])) or np.issubdtype(arr.dtype, np.bool_):
            raise ValueError(f"{name} must contain strictly binary values (0 or 1), excluding booleans.")
        return arr.astype(int)

    def _validate_decode_common(self, length: int, orig_size: int, max_iter: int, unit_name: str) -> int:
        if length == 0:
            raise ValueError(f"Received {unit_name} cannot be empty.")
        if length % self.n != 0:
            raise ValueError(f"Received {unit_name} length must be a multiple of codeword length n={self.n}.")
        if not _is_valid_int(orig_size) or int(orig_size) <= 0:
            raise ValueError("orig_size must be a positive integer.")
        if not _is_valid_int(max_iter) or int(max_iter) <= 0:
            raise ValueError("max_iter must be a positive integer.")

        num_blocks = length // self.n
        if int(orig_size) > num_blocks * self.k:
            raise ValueError("orig_size exceeds the decoded message capacity.")
        return num_blocks

    def encode(self, bits: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Systematic encoding for full-rank H over GF(2):
        Computes p = P^{-1} A_info u (mod 2) and places [u, p] at their codeword indices.
        """
        if self._P_inv is None or self._A_info is None or self._inv_assemble_order is None:
            raise RuntimeError("Encoding requires full row rank H over GF(2).")

        bits = self._validate_bits(bits, "Input message")
        orig_size = bits.size

        padding_len = (self.k - (bits.size % self.k)) % self.k
        if padding_len > 0:
            bits = np.pad(bits, (0, padding_len), mode="constant")

        blocks = bits.reshape(-1, self.k)
        parity = ((blocks @ self._A_info.T) % 2) @ self._P_inv.T % 2
        permuted_cw = np.concatenate([blocks, parity], axis=1)
        codewords = permuted_cw[:, self._inv_assemble_order].reshape(-1)
        return codewords, orig_size

    def extract_message_bits(self, codeword_block: np.ndarray) -> np.ndarray:
        """Extract the k systematic message bits from a length-n codeword block."""
        cw = np.asarray(codeword_block, dtype=int)
        if cw.shape != (self.n,):
            raise ValueError(f"Expected codeword block of shape ({self.n},).")
        if self._info_cols is not None:
            return cw[self._info_cols]
        return cw[:self.k]

    def _sum_product_block(
        self,
        r: np.ndarray,
        max_iter: int = 25,
        algorithm: str = "sum-product",
        norm_factor: float = 0.8,
        llr_clip: float = 50.0,
    ) -> tuple[np.ndarray, int, bool]:
        if not _is_valid_int(max_iter) or int(max_iter) <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if not _is_finite_number(llr_clip) or float(llr_clip) <= 0.0:
            raise ValueError("llr_clip must be a positive finite number.")
        if not _is_finite_number(norm_factor) or not (0.0 < float(norm_factor) <= 1.0):
            raise ValueError("norm_factor must be in (0, 1].")
        if algorithm not in ("sum-product", "min-sum", "normalized-min-sum"):
            raise ValueError("algorithm must be 'sum-product', 'min-sum', or 'normalized-min-sum'.")

        r_arr = np.asarray(r, dtype=float)
        if r_arr.shape != (self.n,) or not np.all(np.isfinite(r_arr)):
            raise ValueError(f"r must be a finite 1D array of length n={self.n}.")

        clip_val = float(llr_clip)
        alpha = float(norm_factor)
        r_clipped = np.clip(r_arr, -clip_val, clip_val)

        M = np.zeros((self.m, self.n), dtype=float)
        for i in range(self.n):
            M[self.A[i], i] = r_clipped[i]

        E = np.zeros((self.m, self.n), dtype=float)
        converged = False
        iters = 0

        for it in range(1, int(max_iter) + 1):
            iters = it

            for j in range(self.m):
                cols = self.B[j]
                if cols.size < 2:
                    continue
                m_vals = M[j, cols]

                if algorithm == "sum-product":
                    th = np.clip(np.tanh(0.5 * m_vals), -1.0 + 1e-12, 1.0 - 1e-12)
                    for idx, i in enumerate(cols):
                        prod_ext = float(np.prod(np.delete(th, idx)))
                        prod_ext = float(np.clip(prod_ext, -1.0 + 1e-12, 1.0 - 1e-12))
                        E[j, i] = float(np.clip(2.0 * np.arctanh(prod_ext), -clip_val, clip_val))
                else:
                    signs = np.where(m_vals >= 0.0, 1.0, -1.0)
                    mags = np.abs(m_vals)
                    scale = 1.0 if algorithm == "min-sum" else alpha
                    for idx, i in enumerate(cols):
                        s_ext = float(np.prod(np.delete(signs, idx)))
                        min_ext = float(np.min(np.delete(mags, idx)))
                        E[j, i] = float(np.clip(scale * s_ext * min_ext, -clip_val, clip_val))

            L = r_clipped.copy()
            for i in range(self.n):
                L[i] += float(np.sum(E[self.A[i], i]))

            z = np.where(L <= 0.0, 1, 0)
            if not np.any((self.H @ z) % 2):
                converged = True
                break

            if it < int(max_iter):
                for i in range(self.n):
                    checks = self.A[i]
                    total_ext = float(np.sum(E[checks, i]))
                    for j in checks:
                        M[j, i] = float(np.clip(r_clipped[i] + (total_ext - E[j, i]), -clip_val, clip_val))

        return z, iters, converged

    def decode(
        self,
        received_llr: np.ndarray,
        orig_size: int,
        max_iter: int = 25,
        algorithm: str = "sum-product",
        norm_factor: float = 0.8,
        llr_clip: float = 50.0,
        reference_msg_bits: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Soft-decision iterative LDPC decoding over a stream of finite a priori LLRs.
        """
        llr_arr = np.asarray(received_llr, dtype=float)
        if not np.all(np.isfinite(llr_arr)):
            raise ValueError("received_llr must contain only finite values (no NaN or Inf).")

        num_blocks = self._validate_decode_common(llr_arr.size, orig_size, max_iter, "LLR")

        ref_blocks = None
        if reference_msg_bits is not None:
            ref_arr = self._validate_bits(reference_msg_bits, "reference_msg_bits")
            if ref_arr.size != int(orig_size):
                raise ValueError("reference_msg_bits length must match orig_size.")
            pad_len = (num_blocks * self.k) - ref_arr.size
            padded_ref = np.pad(ref_arr, (0, pad_len), mode="constant") if pad_len > 0 else ref_arr
            ref_blocks = padded_ref.reshape(num_blocks, self.k)

        blocks = llr_arr.reshape(num_blocks, self.n)
        decoded_blocks = []
        syndrome_failed_blocks = 0
        msg_error_blocks = 0
        total_iterations = 0

        for idx, r_block in enumerate(blocks):
            z_block, iters, converged = self._sum_product_block(
                r_block,
                max_iter=max_iter,
                algorithm=algorithm,
                norm_factor=norm_factor,
                llr_clip=llr_clip,
            )
            total_iterations += iters
            if not converged:
                syndrome_failed_blocks += 1
            msg_block = self.extract_message_bits(z_block)
            if ref_blocks is not None and not np.array_equal(msg_block, ref_blocks[idx]):
                msg_error_blocks += 1
            decoded_blocks.append(msg_block)

        decoded_bits = np.concatenate(decoded_blocks)[:int(orig_size)]
        stats = {
            "blocks": num_blocks,
            "failed_blocks": syndrome_failed_blocks,
            "syndrome_failed_blocks": syndrome_failed_blocks,
            "msg_error_blocks": msg_error_blocks if ref_blocks is not None else None,
            "bler": (msg_error_blocks / num_blocks) if ref_blocks is not None else (syndrome_failed_blocks / num_blocks),
            "total_iterations": total_iterations,
            "avg_iterations": total_iterations / num_blocks,
            "algorithm": algorithm,
        }
        return decoded_bits, stats

    def decode_bit_flipping(
        self, received_bits: np.ndarray, orig_size: int, max_iter: int = 20
    ) -> tuple[np.ndarray, dict]:
        """
        Hard-decision Gallager Bit-Flipping decoder.
        """
        bits = self._validate_bits(received_bits, "Received bits")
        num_blocks = self._validate_decode_common(bits.size, orig_size, max_iter, "bits")

        blocks = bits.reshape(num_blocks, self.n).copy()
        decoded_blocks = []
        failed_blocks = 0
        flipped_bits_total = 0

        for block in blocks:
            M = block.copy()
            converged = not np.any((self.H @ M) % 2)

            if not converged:
                for _ in range(int(max_iter)):
                    E = np.zeros((self.m, self.n), dtype=int)
                    for j in range(self.m):
                        cols = self.B[j]
                        if cols.size == 0:
                            continue
                        parity_sum = int(np.sum(M[cols])) % 2
                        for i in cols:
                            E[j, i] = (parity_sum - int(M[i])) % 2

                    next_M = M.copy()
                    for i in range(self.n):
                        checks = self.A[i]
                        if checks.size == 0:
                            continue
                        disagree_count = int(np.sum(E[checks, i] != M[i]))
                        if disagree_count > (checks.size / 2.0):
                            next_M[i] ^= 1
                            flipped_bits_total += 1

                    M = next_M
                    if not np.any((self.H @ M) % 2):
                        converged = True
                        break

            if not converged:
                failed_blocks += 1
            decoded_blocks.append(self.extract_message_bits(M))

        decoded_bits = np.concatenate(decoded_blocks)[:int(orig_size)]
        stats = {
            "blocks": num_blocks,
            "failed_blocks": failed_blocks,
            "syndrome_failed_blocks": failed_blocks,
            "bler": failed_blocks / num_blocks,
            "flipped_bits": flipped_bits_total,
            "algorithm": "bit-flipping",
        }
        return decoded_bits, stats


@dataclass(frozen=True)
class MCSMode:
    name: str
    mod_order: int
    codec: LDPCCodec
    min_snr_db: float
    target_bler: float = 1e-2

    def __post_init__(self):
        if not _is_finite_number(self.min_snr_db):
            raise ValueError("min_snr_db must be a finite number.")
        if not _is_finite_number(self.target_bler) or not (0.0 < float(self.target_bler) < 1.0):
            raise ValueError("target_bler must be strictly between 0 and 1.")

    @property
    def rate(self) -> float:
        return self.codec.k / self.codec.n

    @property
    def spectral_efficiency(self) -> float:
        return float(np.log2(self.mod_order)) * self.rate


class AMCController:
    """
    Selects the highest spectral-efficiency MCS among all modes whose
    SNR threshold is satisfied.
    """

    def __init__(self, modes: list[MCSMode]):
        if not modes:
            raise ValueError("At least one MCSMode is required.")
        self.modes = sorted(modes, key=lambda m: (m.min_snr_db, m.spectral_efficiency))

    def select_mode(self, snr_db: float) -> MCSMode:
        if not _is_finite_number(snr_db):
            raise ValueError("snr_db must be a finite number.")
        snr_val = float(snr_db)
        eligible = [mode for mode in self.modes if snr_val >= mode.min_snr_db]
        if not eligible:
            return self.modes[0]
        return max(eligible, key=lambda mode: mode.spectral_efficiency)


def simulate_amc_block(
    msg_bits: np.ndarray,
    mcs: MCSMode,
    snr_db: float,
    fading: str = "awgn",
    max_iter: int = 25,
    algorithm: str = "sum-product",
    norm_factor: float = 0.8,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """
    End-to-end simulation of one AMC transmission:
    LDPC encode -> QAM map -> AWGN / Rayleigh fading -> soft LLR demap -> LDPC decode.
    snr_db is Es/N0 in dB.
    """
    if not _is_finite_number(snr_db):
        raise ValueError("snr_db must be a finite number.")
    if rng is None:
        rng = np.random.default_rng()

    msg_arr = mcs.codec._validate_bits(msg_bits, "msg_bits")
    codeword, orig_size = mcs.codec.encode(msg_arr)
    cw_blocks = codeword.reshape(-1, mcs.codec.n)
    if np.any((cw_blocks @ mcs.codec.H.T) % 2 != 0):
        raise RuntimeError("Encoded codeword failed parity check H @ c^T == 0.")

    syms, bit_map = build_constellation(mcs.mod_order)
    bps = bit_map.shape[1]

    pad_len = (bps - (codeword.size % bps)) % bps
    tx_bits = np.pad(codeword, (0, pad_len), mode="constant") if pad_len > 0 else codeword
    bit_groups = tx_bits.reshape(-1, bps)

    bit_to_idx = {tuple(row.tolist()): idx for idx, row in enumerate(bit_map)}
    sym_indices = [bit_to_idx[tuple(row.tolist())] for row in bit_groups]
    tx_syms = syms[sym_indices]

    es_n0 = 10.0 ** (float(snr_db) / 10.0)
    noise_var_per_dim = 1.0 / (2.0 * es_n0)

    if fading == "awgn":
        h = np.ones_like(tx_syms, dtype=complex)
    elif fading == "rayleigh":
        h = (
            rng.normal(0.0, 1.0 / math.sqrt(2.0), size=tx_syms.shape)
            + 1j * rng.normal(0.0, 1.0 / math.sqrt(2.0), size=tx_syms.shape)
        )
    else:
        raise ValueError("fading must be 'awgn' or 'rayleigh'.")

    noise = (
        rng.normal(0.0, math.sqrt(noise_var_per_dim), size=tx_syms.shape)
        + 1j * rng.normal(0.0, math.sqrt(noise_var_per_dim), size=tx_syms.shape)
    )
    rx_syms = h * tx_syms + noise

    all_llrs = qam_soft_llr(
        rx_syms,
        mod_order=mcs.mod_order,
        noise_var_per_dim=noise_var_per_dim,
        channel_gain=h,
    )
    coded_llrs = all_llrs[:codeword.size]
    recovered_bits, dec_stats = mcs.codec.decode(
        coded_llrs,
        orig_size=orig_size,
        max_iter=max_iter,
        algorithm=algorithm,
        norm_factor=norm_factor,
        reference_msg_bits=msg_arr,
    )

    bit_errors = int(np.sum(msg_arr != recovered_bits))
    info = {
        "mcs": mcs.name,
        "spectral_eff": mcs.spectral_efficiency,
        "bit_errors": bit_errors,
        "ber": bit_errors / msg_arr.size,
        **dec_stats,
    }
    return recovered_bits, info


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    codec_default = LDPCCodec()

    amc = AMCController([
        MCSMode("BPSK_R0.5", mod_order=2, codec=codec_default, min_snr_db=-2.0),
        MCSMode("QPSK_R0.5", mod_order=4, codec=codec_default, min_snr_db=2.0),
        MCSMode("16QAM_R0.5", mod_order=16, codec=codec_default, min_snr_db=8.0),
    ])

    test_msg = rng.integers(0, 2, size=28)
    for snr in [-1.0, 4.0, 10.0]:
        selected = amc.select_mode(snr)
        _, res = simulate_amc_block(test_msg, selected, snr_db=snr, fading="awgn", rng=rng)
        print(
            f"SNR={snr:5.1f} dB -> MCS={res['mcs']:10s} | SE={res['spectral_eff']:.2f} | "
            f"BER={res['ber']:.4f} | BLER={res['bler']:.4f} | iters={res['avg_iterations']:.1f}"
        )
