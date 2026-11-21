"""Unified IQ ingestion for files, SigMF recordings, in-memory streams, and Pluto SDR.

Supported file & SigMF binary formats (subset of SigMF datatypes):
- ``cf32_le`` / ``cf32`` / ``complex64`` / ``cfile``: interleaved little-endian float32 I, Q.
- ``cf64_le`` / ``cf64`` / ``complex128``: interleaved little-endian float64 I, Q.
- ``ci16_le`` / ``ci16`` / ``int16`` / ``sc16``: interleaved little-endian int16 I, Q scaled by 32768.0.
- ``cs8``: interleaved int8 I, Q scaled by 128.0.
- ``cu8`` / ``uint8``: interleaved uint8 I, Q centered at 127.5 and scaled by 127.5.
- ``npy`` (file/source only, not a SigMF datatype): 1D complex NumPy array.

SigMF capture frequency fallback policy:
- When a capture entry omits ``core:frequency`` (or sets it to ``None``), the
  loader/source falls back to the externally supplied ``center_freq_hz`` (if any),
  rather than assuming an unchanged or known frequency.

Metadata (`IQBatch.meta`) conventions:
- Common keys across sources:
  - ``source_type`` (str): ``"array"``, ``"file"``, ``"sigmf"``, or ``"pluto_sdr"``.
  - ``num_samples`` (int): Number of IQ samples in the batch (when present, must equal ``len(batch.samples)``).
- File / SigMF keys:
  - ``path`` (str): Path to the IQ sample file.
  - ``format`` (str): Normalized format identifier.
  - ``sample_rate`` (float | None): Sample rate in Hz.
  - ``center_freq_hz`` (float | None): Active center frequency in Hz for the batch/recording.
- Additional SigMF keys:
  - ``meta_path`` (str): Path to the ``.sigmf-meta`` file.
  - ``data_path`` (str): Path to the ``.sigmf-data`` file.
  - ``description`` (str): ``core:description`` from ``global``.
  - ``sigmf_has_non_zero_capture_offsets`` (bool): Whether any capture has ``core:sample_start > 0``.
  - ``sigmf_has_annotations`` (bool): Whether non-empty ``annotations`` are present.
  - ``sigmf_capture_boundary_crossed`` (bool): True when a batch crosses a capture/segment with a different frequency.
- Additional Pluto SDR keys:
  - ``uri`` (str): IIO URI used to connect to the Pluto SDR.
  - ``raw_scale`` (float): Fixed normalization divisor applied to raw Pluto I/Q samples.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


class IQCaptureError(RuntimeError):
    """Raised when an IQ source fails to capture valid samples from hardware or stream."""


_VALID_SOURCE_TYPES = frozenset({"array", "file", "sigmf", "pluto_sdr"})

_BINARY_FORMATS = frozenset(
    {
        "cf32_le",
        "cf32",
        "complex64",
        "cfile",
        "cf64_le",
        "cf64",
        "complex128",
        "ci16_le",
        "ci16",
        "int16",
        "sc16",
        "cs8",
        "cu8",
        "uint8",
    }
)


def _validate_finite_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite real number.")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite real number.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite real number.")
    return result


def _validate_positive_finite_real(value: object, name: str) -> float:
    return _validate_finite_real(value, name, positive=True)


def _validate_optional_positive_real(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _validate_positive_finite_real(value, name)


def _validate_optional_finite_real(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _validate_finite_real(value, name, positive=False)


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _validate_integer_hz(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer-valued frequency/rate in Hz.")
    if isinstance(value, (int, np.integer)):
        ival = int(value)
    elif isinstance(value, (float, np.floating)):
        fval = float(value)
        if not math.isfinite(fval) or not fval.is_integer():
            raise ValueError(f"{name} must be an exact integer value in Hz; got {value!r}.")
        ival = int(fval)
    else:
        raise ValueError(f"{name} must be an integer-valued frequency/rate in Hz.")

    if positive and ival <= 0:
        raise ValueError(f"{name} must be a positive integer in Hz.")
    return ival


def _validate_optional_integer_hz(value: object, name: str, *, positive: bool = False) -> int | None:
    if value is None:
        return None
    return _validate_integer_hz(value, name, positive=positive)


def _validate_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    return bool(value)


@dataclass(frozen=True)
class _BinaryIQSpec:
    dtype: np.dtype
    bytes_per_sample: int  # 2 * itemsize (I + Q)
    scale: float = 1.0
    offset: float = 0.0


def _resolve_binary_iq_spec(fmt: str) -> _BinaryIQSpec | None:
    fmt_norm = fmt.lower().strip()
    if fmt_norm in ("cf32_le", "cf32", "complex64", "cfile"):
        return _BinaryIQSpec(dtype=np.dtype("<f4"), bytes_per_sample=8)
    if fmt_norm in ("cf64_le", "cf64", "complex128"):
        return _BinaryIQSpec(dtype=np.dtype("<f8"), bytes_per_sample=16)
    if fmt_norm in ("ci16_le", "ci16", "int16", "sc16"):
        return _BinaryIQSpec(dtype=np.dtype("<i2"), bytes_per_sample=4, scale=32768.0)
    if fmt_norm == "cs8":
        return _BinaryIQSpec(dtype=np.dtype("<i1"), bytes_per_sample=2, scale=128.0)
    if fmt_norm in ("cu8", "uint8"):
        return _BinaryIQSpec(dtype=np.dtype("u1"), bytes_per_sample=2, scale=127.5, offset=-127.5)
    return None


def _resolve_effective_format(path: Path, fmt: str, *, fmt_explicit: bool = False) -> str:
    fmt_norm = fmt.lower().strip()
    is_npy_ext = path.suffix.lower() == ".npy"

    if is_npy_ext:
        if fmt_norm == "npy":
            return "npy"
        if fmt_explicit and fmt_norm in _BINARY_FORMATS:
            raise ValueError(
                f"Format mismatch for '{path}': file has '.npy' extension, "
                f"but binary format '{fmt}' was explicitly requested."
            )
        return "npy"

    if fmt_norm == "npy":
        raise ValueError(
            f"Format mismatch for '{path}': format 'npy' requested, "
            f"but file extension is '{path.suffix or '(none)'}'."
        )
    return fmt_norm


def _load_complex_npy_1d(path: Path, *, mmap_mode: str | None = None) -> np.ndarray:
    try:
        raw = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    except Exception as exc:
        if mmap_mode is not None:
            raise IQCaptureError(f"Failed to memory-map NPY IQ file '{path}': {exc}") from exc
        raise ValueError(f"Failed to load NPY IQ file '{path}': {exc}") from exc

    if not isinstance(raw, np.ndarray):
        raise ValueError(f"NPY IQ file '{path}' must contain a NumPy array.")
    if raw.ndim != 1:
        raise ValueError(f"NPY IQ array in '{path}' must be one-dimensional; got ndim={raw.ndim}.")
    if not np.iscomplexobj(raw):
        raise ValueError(f"NPY IQ array in '{path}' must contain complex samples; got dtype={raw.dtype}.")
    if raw.size == 0:
        raise ValueError(f"NPY IQ file '{path}' contains no samples.")
    return raw


@dataclass(frozen=True)
class _SigMFMetadata:
    datatype: str
    sample_rate: float | None
    captures: list[dict]
    description: str
    has_non_zero_capture_offsets: bool
    has_annotations: bool

    def frequency_at(
        self,
        sample_index: int = 0,
        fallback_freq: float | None = None,
    ) -> float | None:
        active: dict | None = None
        for cap in self.captures:
            start = int(cap.get("core:sample_start", 0))
            if start <= sample_index:
                active = cap
            else:
                break
        if active is None:
            return fallback_freq
        freq = active.get("core:frequency")
        if freq is None:
            return fallback_freq
        return _validate_finite_real(freq, "core:frequency")

    def spans_frequency_change(
        self,
        start_sample: int,
        length: int,
        fallback_freq: float | None = None,
    ) -> bool:
        if length <= 1 or not self.captures:
            return False
        end_sample = start_sample + length - 1
        initial_freq = self.frequency_at(start_sample, fallback_freq=fallback_freq)

        for cap in self.captures:
            cap_start = int(cap.get("core:sample_start", 0))
            if start_sample < cap_start <= end_sample:
                raw_freq = cap.get("core:frequency")
                next_freq = (
                    fallback_freq
                    if raw_freq is None
                    else _validate_finite_real(raw_freq, "core:frequency")
                )
                if next_freq != initial_freq:
                    return True
        return False


def _parse_and_validate_sigmf(doc: object) -> _SigMFMetadata:
    if not isinstance(doc, dict):
        raise ValueError("SigMF metadata JSON root must be a JSON object.")

    global_sec = doc.get("global")
    if not isinstance(global_sec, dict):
        raise ValueError("SigMF 'global' section must be a JSON object.")

    captures = doc.get("captures")
    if not isinstance(captures, list):
        raise ValueError("SigMF 'captures' section must be a JSON list.")
    for idx, cap in enumerate(captures):
        if not isinstance(cap, dict):
            raise ValueError(f"SigMF capture entry at index {idx} must be a JSON object.")

    datatype = global_sec.get("core:datatype", "cf32_le")
    if not isinstance(datatype, str) or not datatype.strip():
        raise ValueError("SigMF 'core:datatype' must be a non-empty string.")
    datatype_norm = datatype.strip()
    if _resolve_binary_iq_spec(datatype_norm) is None:
        raise ValueError(f"Unsupported or non-standard SigMF 'core:datatype': '{datatype_norm}'.")

    sample_rate = _validate_optional_positive_real(global_sec.get("core:sample_rate"), "core:sample_rate")

    has_non_zero_offsets = False
    validated_captures: list[dict] = []
    seen_offsets: set[int] = set()
    for idx, cap in enumerate(captures):
        raw_start = cap.get("core:sample_start", 0)
        if isinstance(raw_start, (bool, np.bool_)) or not isinstance(raw_start, (int, np.integer)):
            raise ValueError(f"SigMF capture[{idx}] 'core:sample_start' must be a non-negative integer.")
        start = int(raw_start)
        if start < 0:
            raise ValueError(f"SigMF capture[{idx}] 'core:sample_start' cannot be negative.")
        if start in seen_offsets:
            raise ValueError(f"SigMF captures must not have duplicate core:sample_start offset {start}.")
        seen_offsets.add(start)
        if start > 0:
            has_non_zero_offsets = True
        if "core:frequency" in cap and cap["core:frequency"] is not None:
            _validate_finite_real(cap["core:frequency"], "core:frequency")
        validated_captures.append(cap)

    validated_captures.sort(key=lambda c: int(c.get("core:sample_start", 0)))

    annotations = doc.get("annotations")
    if annotations is not None and not isinstance(annotations, list):
        raise ValueError("SigMF 'annotations' section must be a JSON list when present.")
    has_annotations = bool(annotations)

    raw_desc = global_sec.get("core:description", "")
    if raw_desc is None:
        description = ""
    elif isinstance(raw_desc, str):
        description = raw_desc
    else:
        raise ValueError("SigMF 'core:description' must be a string when present.")

    return _SigMFMetadata(
        datatype=datatype_norm,
        sample_rate=sample_rate,
        captures=validated_captures,
        description=description,
        has_non_zero_capture_offsets=has_non_zero_offsets,
        has_annotations=has_annotations,
    )


@dataclass
class IQBatch:
    samples: np.ndarray
    sample_rate: float | None = None
    center_freq_hz: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.complex128).ravel()
        if arr.size == 0:
            raise ValueError("IQBatch samples cannot be empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("IQBatch samples must be finite.")
        if not isinstance(self.meta, dict):
            raise ValueError("IQBatch meta must be a dictionary.")

        if "num_samples" in self.meta:
            reported_count = _validate_positive_int(self.meta["num_samples"], "meta.num_samples")
            if reported_count != arr.size:
                raise ValueError(
                    f"meta.num_samples ({reported_count}) must match IQBatch sample count ({arr.size})."
                )

        if "source_type" in self.meta:
            src_type = self.meta["source_type"]
            if not isinstance(src_type, str) or src_type not in _VALID_SOURCE_TYPES:
                raise ValueError(
                    f"meta.source_type must be one of {sorted(_VALID_SOURCE_TYPES)}, got {src_type!r}."
                )

        self.samples = arr
        self.sample_rate = _validate_optional_positive_real(self.sample_rate, "sample_rate")
        self.center_freq_hz = _validate_optional_finite_real(self.center_freq_hz, "center_freq_hz")


def detect_clipping_and_saturation(
    samples: np.ndarray,
    rail_threshold: float = 0.995,
    full_scale: float = 1.0,
    dc_threshold_ratio: float = 0.05,
    high_power_ratio: float = 0.5,
    stuck_run_threshold: int = 8,
) -> list[str]:
    """Check IQ stream for signal quality/level conditions.

    Diagnostics reported:
    - Clipping: Samples reaching or exceeding ``full_scale * rail_threshold`` on I or Q.
    - Amplitude/power warning: Average complex power ``mean(|x|^2)`` exceeding
      ``(full_scale^2) * high_power_ratio`` (note: a legitimate high-power signal
      can trigger this warning; it is not proof of ADC saturation on its own).
    - DC offset: Non-zero complex mean magnitude ``|mean(x)|`` exceeding
      ``full_scale * dc_threshold_ratio``.
    - Constant I/Q: Invariant I or Q component across multiple samples.
    - Stuck samples: Consecutive identical complex samples of length ``>= stuck_run_threshold``.
    """
    _validate_positive_finite_real(full_scale, "full_scale")
    _validate_positive_finite_real(rail_threshold, "rail_threshold")
    if rail_threshold > 1.0:
        raise ValueError("rail_threshold must be positive and at most 1.0.")
    _validate_positive_finite_real(dc_threshold_ratio, "dc_threshold_ratio")
    _validate_positive_finite_real(high_power_ratio, "high_power_ratio")
    stuck_min = _validate_positive_int(stuck_run_threshold, "stuck_run_threshold")

    x = np.asarray(samples, dtype=np.complex128).ravel()
    if x.size == 0:
        return []
    if not np.all(np.isfinite(x)):
        raise ValueError("IQ samples must contain only finite values.")

    warnings: list[str] = []
    re = np.real(x)
    im = np.imag(x)
    abs_re = np.abs(re)
    abs_im = np.abs(im)

    rail_level = full_scale * rail_threshold
    hard_limit = full_scale * 0.9999

    peak_rail = float(max(np.max(abs_re), np.max(abs_im)))
    clipped_frac = float(max(np.mean(abs_re >= rail_level), np.mean(abs_im >= rail_level)))

    if clipped_frac > 0.005:
        warnings.append(
            f"Clipping detected ({clipped_frac * 100:.2f}% of samples >= {rail_level:.4g} "
            f"[{rail_threshold * 100:.2f}% of full_scale={full_scale:g}])."
        )
    elif peak_rail >= hard_limit:
        warnings.append(
            f"Near-full-scale amplitude detected (peak={peak_rail:.4g} vs full_scale={full_scale:g}); "
            "verify source was not hard-limited or saturated."
        )

    dc_val = complex(np.mean(x))
    dc_mag = abs(dc_val)
    dc_limit = full_scale * dc_threshold_ratio
    if dc_mag >= dc_limit:
        warnings.append(
            f"DC offset detected (|mean(x)|={dc_mag:.4g} at {dc_val.real:+.4g}{dc_val.imag:+.4g}j, "
            f"threshold={dc_limit:.4g})."
        )

    if x.size > 1 and np.ptp(re) == 0.0:
        warnings.append(f"Constant I component detected ({re[0]:+.4g} across {x.size} samples).")
    if x.size > 1 and np.ptp(im) == 0.0:
        warnings.append(f"Constant Q component detected ({im[0]:+.4g} across {x.size} samples).")

    if x.size > 1:
        unchanged = x[1:] == x[:-1]
        if np.any(unchanged):
            max_run = 1
            run = 1
            for same in unchanged:
                if same:
                    run += 1
                    if run > max_run:
                        max_run = run
                else:
                    run = 1
            if max_run >= stuck_min:
                repeated_frac = float(np.mean(unchanged)) * 100.0
                warnings.append(
                    f"Stuck samples detected (max identical run={max_run} samples, "
                    f"{repeated_frac:.2f}% identical consecutive transitions)."
                )

    mean_power = float(np.mean(np.abs(x) ** 2))
    power_limit = (full_scale**2) * high_power_ratio
    if mean_power >= power_limit:
        warnings.append(
            f"High average power warning (mean |x|^2={mean_power:.4g} >= {power_limit:.4g} "
            f"[{high_power_ratio * 100:.1f}% of full_scale^2={full_scale**2:g}])."
        )

    return warnings


def load_iq_file(
    path: str | Path,
    fmt: str = "cf32_le",
    sample_rate: float | None = None,
    center_freq_hz: float | None = None,
    max_samples: int | None = None,
    *,
    fmt_explicit: bool = False,
) -> tuple[np.ndarray, dict]:
    """Load IQ samples into memory from raw binary or .npy files."""
    if not isinstance(fmt, str) or not fmt.strip():
        raise ValueError("fmt must be a non-empty string.")
    validated_sr = _validate_optional_positive_real(sample_rate, "sample_rate")
    validated_cf = _validate_optional_finite_real(center_freq_hz, "center_freq_hz")
    validated_max = None if max_samples is None else _validate_positive_int(max_samples, "max_samples")

    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"IQ file not found: {p}")

    fmt_norm = _resolve_effective_format(p, fmt, fmt_explicit=fmt_explicit)

    if fmt_norm == "npy":
        mmap_arr = _load_complex_npy_1d(p, mmap_mode="r")
        sliced = mmap_arr[:validated_max] if validated_max is not None else mmap_arr
        arr = np.asarray(sliced, dtype=np.complex128)
    else:
        spec = _resolve_binary_iq_spec(fmt_norm)
        if spec is None:
            raise ValueError(f"Unsupported IQ format '{fmt}'.")
        file_bytes = p.stat().st_size
        if file_bytes == 0:
            raise ValueError(f"Insufficient samples in {fmt_norm} IQ file.")
        if validated_max is None and file_bytes % spec.bytes_per_sample != 0:
            raise ValueError(
                f"Invalid IQ file size: {file_bytes} bytes is not "
                f"divisible by {spec.bytes_per_sample} bytes per sample."
            )
        count = None if validated_max is None else validated_max * 2
        data = np.fromfile(p, dtype=spec.dtype, count=count if count is not None else -1)
        if data.size < 2:
            raise ValueError(f"Insufficient samples in {fmt_norm} IQ file.")
        if data.size % 2 != 0:
            raise ValueError(
                f"Invalid IQ stream in {fmt_norm} file: odd number of I/Q components ({data.size})."
            )
        iq = data.reshape(-1, 2).astype(np.float64)
        if spec.offset != 0.0:
            iq = iq + spec.offset
        if spec.scale != 1.0:
            iq = iq / spec.scale
        arr = iq[:, 0] + 1j * iq[:, 1]

    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Loaded IQ samples are empty or contain non-finite values.")

    meta = {
        "source_type": "file",
        "path": str(p),
        "format": fmt_norm,
        "num_samples": int(arr.size),
        "sample_rate": validated_sr,
        "center_freq_hz": validated_cf,
    }
    return arr, meta


def load_sigmf(
    meta_path: str | Path,
    data_path: str | Path | None = None,
    max_samples: int | None = None,
    fallback_center_freq_hz: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Load IQ samples and validated metadata from a SigMF (.sigmf-meta + .sigmf-data) recording."""
    validated_max = None if max_samples is None else _validate_positive_int(max_samples, "max_samples")
    validated_fallback_cf = _validate_optional_finite_real(fallback_center_freq_hz, "fallback_center_freq_hz")

    m_path = Path(meta_path)
    if not m_path.exists() or not m_path.is_file():
        raise FileNotFoundError(f"SigMF metadata file not found: {m_path}")

    with m_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)

    parsed = _parse_and_validate_sigmf(doc)
    initial_freq = parsed.frequency_at(0, fallback_freq=validated_fallback_cf)

    if data_path is None:
        if m_path.name.endswith(".sigmf-meta"):
            d_path = m_path.with_name(m_path.name[: -len(".sigmf-meta")] + ".sigmf-data")
        else:
            d_path = m_path.with_suffix(".sigmf-data")
    else:
        d_path = Path(data_path)

    if not d_path.exists() or not d_path.is_file():
        raise FileNotFoundError(f"SigMF data file not found: {d_path}")

    samples, file_meta = load_iq_file(
        d_path,
        fmt=parsed.datatype,
        sample_rate=parsed.sample_rate,
        center_freq_hz=initial_freq,
        max_samples=validated_max,
        fmt_explicit=True,
    )
    meta = {
        **file_meta,
        "source_type": "sigmf",
        "meta_path": str(m_path),
        "data_path": str(d_path),
        "description": parsed.description,
        "sigmf_has_non_zero_capture_offsets": parsed.has_non_zero_capture_offsets,
        "sigmf_has_annotations": parsed.has_annotations,
    }
    return samples, meta


class IQSource(ABC):
    """Abstract base class for IQ capture/replay sources.

    Batch length contract:
    - When ``loop=False``, ``capture_batch(num_samples)`` returns up to
      ``num_samples`` samples and may return a shorter final batch when fewer
      samples remain before end-of-stream. Subsequent calls once exhausted raise
      ``StopIteration``.
    - When ``loop=True`` (for finite/replayable sources), ``capture_batch(num_samples)``
      wraps around and returns ``num_samples`` samples.
    """

    @abstractmethod
    def capture_batch(self, num_samples: int) -> IQBatch:
        raise NotImplementedError

    def stream_batches(self, batch_size: int = 4096, max_batches: int | None = None) -> Iterator[IQBatch]:
        validated_batch_size = _validate_positive_int(batch_size, "batch_size")
        validated_max_batches = None if max_batches is None else _validate_positive_int(max_batches, "max_batches")
        count = 0
        while validated_max_batches is None or count < validated_max_batches:
            try:
                batch = self.capture_batch(validated_batch_size)
            except StopIteration:
                break
            yield batch
            count += 1

    def close(self) -> None:
        """Release any underlying resources. Override in subclasses if needed."""

    def __enter__(self) -> IQSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ArrayIQSource(IQSource):
    """Replay or chunk an existing in-memory complex IQ NumPy array.

    Batch length contract:
    - ``loop=False``: returns up to ``num_samples`` samples; the final batch may
      be shorter if fewer samples remain, and the following call raises ``StopIteration``.
    - ``loop=True``: wraps around the array and always returns ``num_samples`` samples.
    """

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: float | None = None,
        center_freq_hz: float | None = None,
        loop: bool = False,
    ) -> None:
        arr = np.asarray(samples, dtype=np.complex128).ravel()
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("samples must be a non-empty finite array.")
        validated_sr = _validate_optional_positive_real(sample_rate, "sample_rate")
        validated_cf = _validate_optional_finite_real(center_freq_hz, "center_freq_hz")
        validated_loop = _validate_bool(loop, "loop")

        self._samples = arr
        self._cursor = 0
        self._sample_rate = validated_sr
        self._center_freq_hz = validated_cf
        self._loop = validated_loop

    def capture_batch(self, num_samples: int) -> IQBatch:
        """Capture a batch of IQ samples from the array."""
        n = _validate_positive_int(num_samples, "num_samples")
        if self._cursor >= self._samples.size:
            if self._loop:
                self._cursor = 0
            else:
                raise StopIteration("End of IQ array reached.")

        if self._cursor + n <= self._samples.size:
            chunk = self._samples[self._cursor : self._cursor + n]
            self._cursor += n
        else:
            rem = self._samples.size - self._cursor
            if self._loop:
                parts = [self._samples[self._cursor :]]
                need = n - rem
                while need > 0:
                    take = min(need, self._samples.size)
                    parts.append(self._samples[:take])
                    need -= take
                self._cursor = (n - rem) % self._samples.size
                chunk = np.concatenate(parts)
            else:
                chunk = self._samples[self._cursor :]
                self._cursor = self._samples.size

        return IQBatch(
            samples=chunk,
            sample_rate=self._sample_rate,
            center_freq_hz=self._center_freq_hz,
            meta={"source_type": "array", "num_samples": int(chunk.size)},
        )


class FileIQSource(IQSource):
    """Chunked streaming reader for raw IQ or SigMF files without loading full files into RAM.

    Batch length contract:
    - ``loop=False``: returns up to ``num_samples`` samples; the final batch may
      be shorter if fewer samples remain, and the following call raises ``StopIteration``.
    - ``loop=True``: wraps around the recording and always returns ``num_samples`` samples.
    """

    def __init__(
        self,
        path: str | Path,
        fmt: str = "cf32_le",
        sample_rate: float | None = None,
        center_freq_hz: float | None = None,
        is_sigmf: bool = False,
        loop: bool = False,
        *,
        fmt_explicit: bool = False,
    ) -> None:
        if not isinstance(fmt, str) or not fmt.strip():
            raise ValueError("fmt must be a non-empty string.")
        validated_is_sigmf = _validate_bool(is_sigmf, "is_sigmf")
        validated_loop = _validate_bool(loop, "loop")
        validated_sr = _validate_optional_positive_real(sample_rate, "sample_rate")
        validated_cf = _validate_optional_finite_real(center_freq_hz, "center_freq_hz")

        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"IQ file not found: {p}")

        self._sigmf: _SigMFMetadata | None = None

        if validated_is_sigmf or p.suffix == ".sigmf-meta" or p.name.endswith(".sigmf-meta"):
            with p.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
            parsed = _parse_and_validate_sigmf(doc)
            self._sigmf = parsed

            if p.name.endswith(".sigmf-meta"):
                d_path = p.with_name(p.name[: -len(".sigmf-meta")] + ".sigmf-data")
            else:
                d_path = p.with_suffix(".sigmf-data")
            if not d_path.exists() or not d_path.is_file():
                raise FileNotFoundError(f"SigMF data file not found: {d_path}")

            fmt_to_use = parsed.datatype
            resolved_sr = parsed.sample_rate if parsed.sample_rate is not None else validated_sr
            resolved_cf = parsed.frequency_at(0, fallback_freq=validated_cf)
            self._base_meta = {
                "source_type": "sigmf",
                "path": str(d_path),
                "format": fmt_to_use.lower().strip(),
                "sample_rate": resolved_sr,
                "center_freq_hz": resolved_cf,
                "meta_path": str(p),
                "data_path": str(d_path),
                "description": parsed.description,
                "sigmf_has_non_zero_capture_offsets": parsed.has_non_zero_capture_offsets,
                "sigmf_has_annotations": parsed.has_annotations,
            }
            data_file = d_path
            fmt_norm = fmt_to_use.lower().strip()
        else:
            fmt_norm = _resolve_effective_format(p, fmt, fmt_explicit=fmt_explicit)
            resolved_sr = validated_sr
            resolved_cf = validated_cf
            self._base_meta = {
                "source_type": "file",
                "path": str(p),
                "format": fmt_norm,
                "sample_rate": resolved_sr,
                "center_freq_hz": resolved_cf,
            }
            data_file = p

        self._sample_rate = resolved_sr
        self._fallback_center_freq_hz = resolved_cf
        self._loop = validated_loop
        self._cursor_samples = 0
        self._fh = None
        self._mmap: np.ndarray | None = None
        self._spec: _BinaryIQSpec | None = None
        self._closed = False

        if fmt_norm == "npy":
            mmap_arr = _load_complex_npy_1d(data_file, mmap_mode="r")
            self._mmap = mmap_arr
            self._total_samples = int(mmap_arr.size)
        else:
            spec = _resolve_binary_iq_spec(fmt_norm)
            if spec is None:
                raise ValueError(f"Unsupported IQ format '{fmt_norm}'.")
            file_bytes = data_file.stat().st_size
            if file_bytes == 0:
                raise ValueError(f"Insufficient samples in {fmt_norm} IQ file.")
            if file_bytes % spec.bytes_per_sample != 0:
                raise ValueError(
                    f"Invalid IQ file size: {file_bytes} bytes is not "
                    f"divisible by {spec.bytes_per_sample} bytes per sample."
                )
            self._spec = spec
            self._total_samples = int(file_bytes // spec.bytes_per_sample)
            self._fh = data_file.open("rb")

    def _read_raw_range(self, start_sample: int, count_samples: int) -> np.ndarray:
        if count_samples <= 0:
            return np.empty(0, dtype=np.complex128)
        if self._mmap is not None:
            try:
                raw_slice = self._mmap[start_sample : start_sample + count_samples]
                chunk = np.asarray(raw_slice, dtype=np.complex128)
            except Exception as exc:
                raise IQCaptureError(f"Failed to read memory-mapped IQ samples: {exc}") from exc
            if chunk.size != count_samples:
                raise IQCaptureError(
                    f"Unexpected end of NPY IQ stream (requested {count_samples}, got {chunk.size}); "
                    "file may have been truncated or modified."
                )
        else:
            if self._fh is None or self._spec is None:
                raise IQCaptureError("FileIQSource stream is not open.")
            try:
                self._fh.seek(start_sample * self._spec.bytes_per_sample)
                raw = np.fromfile(self._fh, dtype=self._spec.dtype, count=count_samples * 2)
            except Exception as exc:
                raise IQCaptureError(f"Failed to read IQ samples from file: {exc}") from exc

            if raw.size != count_samples * 2:
                raise IQCaptureError(
                    "Unexpected end of IQ file while reading samples. "
                    "The file may have been truncated or modified."
                )
            iq = raw.reshape(-1, 2).astype(np.float64)
            if self._spec.offset != 0.0:
                iq = iq + self._spec.offset
            if self._spec.scale != 1.0:
                iq = iq / self._spec.scale
            chunk = iq[:, 0] + 1j * iq[:, 1]

        if not np.all(np.isfinite(chunk)):
            raise ValueError("Loaded IQ samples contain non-finite values.")
        return chunk

    def _freq_and_crossing_for_segments(
        self,
        segments: list[tuple[int, int]],
    ) -> tuple[float | None, bool]:
        if self._sigmf is None:
            return self._fallback_center_freq_hz, False

        initial_freq: float | None = None
        has_initial = False
        crossed = False
        prev_end_freq: float | None = None

        for seg_start, seg_len in segments:
            if seg_len <= 0:
                continue

            seg_start_freq = self._sigmf.frequency_at(
                seg_start,
                fallback_freq=self._fallback_center_freq_hz,
            )
            if not has_initial:
                initial_freq = seg_start_freq
                has_initial = True
            elif seg_start_freq != prev_end_freq:
                crossed = True

            if self._sigmf.spans_frequency_change(
                seg_start,
                seg_len,
                fallback_freq=self._fallback_center_freq_hz,
            ):
                crossed = True

            prev_end_freq = self._sigmf.frequency_at(
                seg_start + seg_len - 1,
                fallback_freq=self._fallback_center_freq_hz,
            )

        if not has_initial:
            return self._fallback_center_freq_hz, crossed
        return initial_freq, crossed

    def capture_batch(self, num_samples: int) -> IQBatch:
        """Capture a batch of IQ samples from the file stream."""
        if self._closed:
            raise RuntimeError("FileIQSource is closed.")
        n = _validate_positive_int(num_samples, "num_samples")

        if self._cursor_samples >= self._total_samples:
            if not self._loop:
                raise StopIteration("End of IQ stream reached.")
            self._cursor_samples = 0

        start = self._cursor_samples
        segments: list[tuple[int, int]] = []

        if start + n <= self._total_samples:
            chunk = self._read_raw_range(start, n)
            self._cursor_samples = start + n
            segments.append((start, int(chunk.size)))
        else:
            rem = self._total_samples - start
            if not self._loop:
                chunk = self._read_raw_range(start, rem)
                self._cursor_samples = self._total_samples
                segments.append((start, int(chunk.size)))
            else:
                parts = [self._read_raw_range(start, rem)]
                segments.append((start, int(parts[0].size)))
                need = n - rem
                while need > 0:
                    take = min(need, self._total_samples)
                    wrapped_part = self._read_raw_range(0, take)
                    parts.append(wrapped_part)
                    segments.append((0, int(wrapped_part.size)))
                    need -= take
                self._cursor_samples = (n - rem) % self._total_samples
                chunk = np.concatenate(parts)

        freq, crossed = self._freq_and_crossing_for_segments(segments)

        batch_meta = {
            **self._base_meta,
            "center_freq_hz": freq,
            "num_samples": int(chunk.size),
        }
        if self._sigmf is not None:
            batch_meta["sigmf_capture_boundary_crossed"] = crossed

        return IQBatch(
            samples=chunk,
            sample_rate=self._sample_rate,
            center_freq_hz=freq,
            meta=batch_meta,
        )

    def close(self) -> None:
        self._closed = True
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
        self._mmap = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class PlutoSDRSource(IQSource):
    """Live capture wrapper for Analog Devices ADALM-PLUTO via pyadi-iio (`adi.Pluto`).

    Normalization convention:
    - ``_PLUTO_IQ_SCALE = 2048.0`` is a fixed normalization divisor corresponding
      to nominal 12-bit signed I/Q scaling (``2^11``) used to bring raw Pluto
      samples into a consistent unit-scale range across batches and gain changes.
      It does not replace front-end RF/ADC calibration.
    """

    _PLUTO_IQ_SCALE: float = 2048.0
    _SUPPORTED_GAIN_MODES: tuple[str, ...] = ("manual", "slow_attack", "fast_attack", "hybrid")

    def __init__(
        self,
        uri: str = "ip:192.168.2.1",
        sample_rate: int = 2_400_000,
        center_freq_hz: int = 915_000_000,
        rx_bandwidth_hz: int | None = None,
        gain_mode: str = "slow_attack",
        manual_gain_db: float | None = None,
        rx_buffer_size: int = 8192,
    ) -> None:
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError("uri must be a non-empty string.")
        validated_sr_hz = _validate_integer_hz(sample_rate, "sample_rate", positive=True)
        validated_cf_hz = _validate_integer_hz(center_freq_hz, "center_freq_hz", positive=False)
        validated_bw_hz = (
            validated_sr_hz
            if rx_bandwidth_hz is None
            else _validate_integer_hz(rx_bandwidth_hz, "rx_bandwidth_hz", positive=True)
        )
        validated_buf = _validate_positive_int(rx_buffer_size, "rx_buffer_size")
        if not isinstance(gain_mode, str) or gain_mode not in self._SUPPORTED_GAIN_MODES:
            raise ValueError(
                f"gain_mode must be one of {self._SUPPORTED_GAIN_MODES}, got {gain_mode!r}."
            )
        validated_gain = _validate_optional_finite_real(manual_gain_db, "manual_gain_db")
        if gain_mode == "manual" and validated_gain is None:
            raise ValueError("manual_gain_db must be provided when gain_mode='manual'.")

        self._sdr = None
        try:
            import adi  # type: ignore
        except ImportError as exc:
            raise IQCaptureError(
                "pyadi-iio ('adi') is not installed. Install pyadi-iio/libiio to capture from ADALM-PLUTO."
            ) from exc

        sdr = None
        try:
            sdr = adi.Pluto(uri=uri)
            sdr.sample_rate = validated_sr_hz
            sdr.rx_lo = validated_cf_hz
            sdr.rx_rf_bandwidth = validated_bw_hz
            sdr.rx_buffer_size = validated_buf

            if validated_gain is not None and gain_mode == "manual":
                sdr.gain_control_mode_chan0 = "manual"
                sdr.rx_hardwaregain_chan0 = validated_gain
            else:
                sdr.gain_control_mode_chan0 = gain_mode
                if validated_gain is not None:
                    sdr.rx_hardwaregain_chan0 = validated_gain
        except Exception as exc:
            if sdr is not None:
                destroy = getattr(sdr, "rx_destroy_buffer", None)
                if callable(destroy):
                    try:
                        destroy()
                    except Exception:
                        pass
            raise IQCaptureError(f"Failed to initialize Pluto SDR at '{uri}': {exc}") from exc

        self._sdr = sdr
        self._sample_rate = float(validated_sr_hz)
        self._center_freq_hz = float(validated_cf_hz)
        self._uri = uri

    def capture_batch(self, num_samples: int) -> IQBatch:
        if self._sdr is None:
            raise RuntimeError("PlutoSDRSource is closed.")
        n = _validate_positive_int(num_samples, "num_samples")

        try:
            if getattr(self._sdr, "rx_buffer_size", n) != n:
                self._sdr.rx_buffer_size = n
            raw = self._sdr.rx()
        except Exception as exc:
            raise IQCaptureError(f"Pluto SDR capture failed on '{self._uri}': {exc}") from exc

        arr = np.asarray(raw, dtype=np.complex128).ravel()
        if arr.size == 0:
            raise IQCaptureError(f"Pluto SDR ('{self._uri}') returned an empty capture.")
        if not np.all(np.isfinite(arr)):
            raise IQCaptureError(f"Pluto SDR ('{self._uri}') returned non-finite samples.")

        norm_arr = arr / self._PLUTO_IQ_SCALE
        return IQBatch(
            samples=norm_arr,
            sample_rate=self._sample_rate,
            center_freq_hz=self._center_freq_hz,
            meta={
                "source_type": "pluto_sdr",
                "uri": self._uri,
                "raw_scale": self._PLUTO_IQ_SCALE,
                "num_samples": int(norm_arr.size),
            },
        )

    def close(self) -> None:
        sdr = self._sdr
        self._sdr = None
        if sdr is not None:
            destroy = getattr(sdr, "rx_destroy_buffer", None)
            if callable(destroy):
                try:
                    destroy()
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
