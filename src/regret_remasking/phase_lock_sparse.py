"""Fused and bucketed active-row execution for PhaseLock compute locking.

The semantic and representation controllers deliberately do not live here.
This module consumes a representation-safe active-row mask and decides only
how to execute the remaining transformer rows efficiently.

The CUDA fast path fuses three memory-bound boundaries around the dense GEMMs
and scaled-dot-product attention used by LLaDA:

* gather active rows + RMS normalization;
* query/key RoPE + persistent K/V workspace update;
* frozen-row restore + active-row scatter.

Examples with similar active-row counts share a padded bucket.  Bucket widths
are static multiples so the expensive GEMM/attention shapes are reused across
updates instead of recompiling for every distinct lock count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


try:  # Triton is supplied by CUDA PyTorch wheels, but is optional on CPU hosts.
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):  # pragma: no cover - host dependent
    triton = None
    tl = None


@dataclass(frozen=True)
class ActiveRowBucket:
    """One homogeneous execution group produced by the compute controller."""

    sample_indices: tuple[int, ...]
    active_counts: tuple[int, ...]
    padded_active_rows: int
    total_rows: int

    @property
    def actual_active_rows(self) -> int:
        return sum(self.active_counts)

    @property
    def padding_rows(self) -> int:
        return len(self.sample_indices) * self.padded_active_rows - self.actual_active_rows

    @property
    def padding_fraction(self) -> float:
        return self.padding_rows / max(self.actual_active_rows, 1)


def _round_bucket_width(rows: int, multiple: int, total_rows: int) -> int:
    return min(total_rows, int(math.ceil(rows / multiple) * multiple))


def plan_active_row_buckets(
    active_counts: Sequence[int],
    total_rows: int,
    *,
    width_multiple: int = 16,
    max_bucket_size: int = 8,
    max_padding_fraction: float = 0.25,
    launch_penalty_rows: int = 64,
    bucket_cost: Callable[[int, int, int], float] | None = None,
) -> list[ActiveRowBucket]:
    """Partition a heterogeneous batch into similar-active-count buckets.

    A dynamic program minimizes padded active-row work plus a fixed per-bucket
    launch penalty.  Only adjacent samples after sorting by active count may be
    grouped, which is sufficient for the one-dimensional bucketing objective.
    A singleton is always legal so the planner cannot fail closed merely
    because an active count rounds poorly.

    ``launch_penalty_rows`` is a systems calibration in row-equivalent units;
    it is not a safety threshold.  Larger values favor fewer, wider buckets.
    """
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if width_multiple <= 0 or max_bucket_size <= 0:
        raise ValueError("bucket width and size must be positive")
    if max_padding_fraction < 0.0 or launch_penalty_rows < 0:
        raise ValueError("padding fraction and launch penalty must be non-negative")
    if not active_counts:
        return []
    counts = [int(count) for count in active_counts]
    if any(count <= 0 or count > total_rows for count in counts):
        raise ValueError("every active count must be in [1, total_rows]")

    ordered = sorted(enumerate(counts), key=lambda pair: (-pair[1], pair[0]))
    count = len(ordered)
    # Each entry is (cost, number_of_buckets, previous_boundary).
    best: list[tuple[float, int, int] | None] = [None] * (count + 1)
    best[0] = (0.0, 0, -1)
    for end in range(1, count + 1):
        for start in range(max(0, end - max_bucket_size), end):
            members = ordered[start:end]
            width = _round_bucket_width(members[0][1], width_multiple, total_rows)
            actual = sum(item[1] for item in members)
            padded = width * len(members)
            padding_fraction = (padded - actual) / actual
            if len(members) > 1 and padding_fraction > max_padding_fraction:
                continue
            previous = best[start]
            if previous is None:
                continue
            execution_cost = (
                float(bucket_cost(len(members), width, total_rows))
                if bucket_cost is not None
                else float(padded + launch_penalty_rows)
            )
            if not math.isfinite(execution_cost) or execution_cost < 0.0:
                raise ValueError("bucket cost must be finite and non-negative")
            candidate = (
                previous[0] + execution_cost,
                previous[1] + 1,
                start,
            )
            current = best[end]
            if current is None or candidate[:2] < current[:2]:
                best[end] = candidate
    if best[count] is None:  # pragma: no cover - singleton guarantee
        raise RuntimeError("active-row bucket planner found no legal partition")

    groups: list[ActiveRowBucket] = []
    end = count
    while end:
        state = best[end]
        if state is None:  # pragma: no cover - guarded above
            raise RuntimeError("broken bucket backpointer")
        start = state[2]
        members = ordered[start:end]
        groups.append(
            ActiveRowBucket(
                sample_indices=tuple(item[0] for item in members),
                active_counts=tuple(item[1] for item in members),
                padded_active_rows=_round_bucket_width(
                    members[0][1], width_multiple, total_rows
                ),
                total_rows=total_rows,
            )
        )
        end = start
    groups.reverse()
    return groups


@dataclass(frozen=True)
class FusedLatencyEntry:
    sequence_length: int
    batch_size: int
    active_rows: int
    median_ms: float
    dense_median_ms: float


class FusedLatencyTable:
    """Interpolated hardware calibration for Phase 3 only."""

    def __init__(self, entries: Sequence[FusedLatencyEntry]) -> None:
        if not entries:
            raise ValueError("latency table requires at least one entry")
        self.entries = tuple(entries)
        self._curves: dict[tuple[int, int], list[tuple[int, float]]] = {}
        self._dense: dict[tuple[int, int], float] = {}
        for entry in entries:
            if (
                entry.sequence_length <= 0
                or entry.batch_size <= 0
                or entry.active_rows <= 0
                or entry.median_ms <= 0.0
                or entry.dense_median_ms <= 0.0
            ):
                raise ValueError("latency entries must be strictly positive")
            key = (entry.sequence_length, entry.batch_size)
            self._curves.setdefault(key, []).append(
                (entry.active_rows, entry.median_ms)
            )
            existing = self._dense.setdefault(key, entry.dense_median_ms)
            if not math.isclose(existing, entry.dense_median_ms, rel_tol=1e-4):
                raise ValueError("dense latency must be stable within one table curve")
        for key, curve in self._curves.items():
            deduplicated: dict[int, float] = {}
            for active_rows, latency in curve:
                if active_rows in deduplicated:
                    raise ValueError(f"duplicate latency entry for {key + (active_rows,)}")
                deduplicated[active_rows] = latency
            self._curves[key] = sorted(deduplicated.items())

    @classmethod
    def from_rows(cls, rows: Sequence[dict[str, Any]]) -> "FusedLatencyTable":
        return cls(
            [
                FusedLatencyEntry(
                    sequence_length=int(row["sequence_length"]),
                    batch_size=int(row["batch_size"]),
                    active_rows=int(row["active_rows"]),
                    median_ms=float(row["median_ms"]),
                    dense_median_ms=float(row["dense_median_ms"]),
                )
                for row in rows
            ]
        )

    def _curve(self, sequence_length: int, batch_size: int) -> list[tuple[int, float]]:
        key = (sequence_length, batch_size)
        if key not in self._curves:
            raise KeyError(f"missing fused latency curve for {key}")
        return self._curves[key]

    def _sequence_lengths(self, batch_size: int) -> list[int]:
        lengths = sorted(
            sequence_length
            for sequence_length, curve_batch_size in self._curves
            if curve_batch_size == batch_size
        )
        if not lengths:
            raise KeyError(f"missing fused latency curves for batch {batch_size}")
        return lengths

    def _predict_on_curve(
        self, sequence_length: int, batch_size: int, active_rows: float
    ) -> float:
        curve = self._curve(sequence_length, batch_size)
        if active_rows <= curve[0][0]:
            return curve[0][1]
        if active_rows >= curve[-1][0]:
            return curve[-1][1]
        for (left_rows, left_ms), (right_rows, right_ms) in zip(curve, curve[1:]):
            if left_rows <= active_rows <= right_rows:
                fraction = (active_rows - left_rows) / (right_rows - left_rows)
                return left_ms + fraction * (right_ms - left_ms)
        raise RuntimeError("latency interpolation failed")  # pragma: no cover

    def predict_sparse_ms(
        self, batch_size: int, active_rows: int, sequence_length: int
    ) -> float:
        """Linearly interpolate one homogeneous bucket's measured latency."""
        lengths = self._sequence_lengths(batch_size)
        if sequence_length in lengths:
            return self._predict_on_curve(
                sequence_length, batch_size, float(active_rows)
            )
        if sequence_length <= lengths[0]:
            reference = lengths[0]
            mapped_rows = active_rows * reference / sequence_length
            return self._predict_on_curve(reference, batch_size, mapped_rows)
        if sequence_length >= lengths[-1]:
            reference = lengths[-1]
            mapped_rows = active_rows * reference / sequence_length
            # Extrapolate conservatively with the full-reference attention span.
            return self._predict_on_curve(reference, batch_size, mapped_rows) * (
                sequence_length / reference
            )
        upper_index = next(
            index for index, value in enumerate(lengths) if value > sequence_length
        )
        lower = lengths[upper_index - 1]
        upper = lengths[upper_index]
        active_fraction = active_rows / sequence_length
        lower_ms = self._predict_on_curve(
            lower, batch_size, active_fraction * lower
        )
        upper_ms = self._predict_on_curve(
            upper, batch_size, active_fraction * upper
        )
        fraction = (sequence_length - lower) / (upper - lower)
        return lower_ms + fraction * (upper_ms - lower_ms)

    def dense_ms(self, sequence_length: int, batch_size: int) -> float:
        lengths = self._sequence_lengths(batch_size)
        if sequence_length in lengths:
            return self._dense[(sequence_length, batch_size)]
        if sequence_length <= lengths[0]:
            return self._dense[(lengths[0], batch_size)]
        if sequence_length >= lengths[-1]:
            return self._dense[(lengths[-1], batch_size)] * (
                sequence_length / lengths[-1]
            )
        upper_index = next(
            index for index, value in enumerate(lengths) if value > sequence_length
        )
        lower = lengths[upper_index - 1]
        upper = lengths[upper_index]
        fraction = (sequence_length - lower) / (upper - lower)
        return self._dense[(lower, batch_size)] + fraction * (
            self._dense[(upper, batch_size)] - self._dense[(lower, batch_size)]
        )


@dataclass(frozen=True)
class ComputeExecutionPlan:
    """Phase-3 decision; it carries no semantic/reference-safety meaning."""

    backend: str
    buckets: tuple[ActiveRowBucket, ...]
    predicted_latency_ms: float
    dense_latency_ms: float

    @property
    def predicted_gain(self) -> float:
        return 1.0 - self.predicted_latency_ms / self.dense_latency_ms


def choose_fused_compute_plan(
    active_counts: Sequence[int],
    total_rows: int,
    latency_table: FusedLatencyTable,
    *,
    width_multiple: int = 16,
    max_bucket_size: int = 8,
    max_padding_fraction: float = 1.0,
    min_expected_gain: float = 0.03,
) -> ComputeExecutionPlan:
    """Choose dense, global fused, or latency-optimal fused buckets.

    The compute phase consumes only Phase-2-safe active counts.  It cannot
    create semantic or representation locks.  A positive margin is required
    because the calibration excludes online controller and re-bucketing cost.
    """
    if not 0.0 <= min_expected_gain < 1.0:
        raise ValueError("min_expected_gain must be in [0, 1)")
    batch_size = len(active_counts)
    if batch_size <= 0:
        raise ValueError("active_counts cannot be empty")
    counts = [int(value) for value in active_counts]
    dense_ms = latency_table.dense_ms(total_rows, batch_size)

    global_width = _round_bucket_width(max(counts), width_multiple, total_rows)
    global_bucket = ActiveRowBucket(
        sample_indices=tuple(range(batch_size)),
        active_counts=tuple(counts),
        padded_active_rows=global_width,
        total_rows=total_rows,
    )

    def cost(bucket_size: int, width: int, sequence_length: int) -> float:
        return latency_table.predict_sparse_ms(bucket_size, width, sequence_length)

    optimized = plan_active_row_buckets(
        counts,
        total_rows,
        width_multiple=width_multiple,
        max_bucket_size=max_bucket_size,
        max_padding_fraction=max_padding_fraction,
        bucket_cost=cost,
    )
    candidates = [(global_bucket,)]
    optimized_tuple = tuple(optimized)
    if optimized_tuple != candidates[0]:
        candidates.append(optimized_tuple)
    best_buckets = min(
        candidates,
        key=lambda plans: sum(
            cost(len(plan.sample_indices), plan.padded_active_rows, total_rows)
            for plan in plans
        ),
    )
    predicted_ms = sum(
        cost(len(plan.sample_indices), plan.padded_active_rows, total_rows)
        for plan in best_buckets
    )
    predicted_gain = 1.0 - predicted_ms / dense_ms
    if predicted_gain < min_expected_gain:
        return ComputeExecutionPlan("dense", (), dense_ms, dense_ms)
    backend = "fused_global" if len(best_buckets) == 1 else "fused_bucketed"
    return ComputeExecutionPlan(backend, best_buckets, predicted_ms, dense_ms)


@dataclass
class MaterializedActiveRowBucket:
    """Device tensors for one active-row bucket."""

    plan: ActiveRowBucket
    sample_indices: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    active_rank: torch.Tensor


def materialize_active_row_bucket(
    active_mask: torch.Tensor, bucket: ActiveRowBucket
) -> MaterializedActiveRowBucket:
    """Build padded positions and a full-row inverse map for one bucket."""
    if active_mask.ndim != 2 or active_mask.dtype != torch.bool:
        raise ValueError("active_mask must be a [batch, rows] bool tensor")
    if active_mask.shape[1] != bucket.total_rows:
        raise ValueError("active mask and bucket total_rows disagree")
    device = active_mask.device
    sample_indices = torch.tensor(bucket.sample_indices, dtype=torch.long, device=device)
    width = bucket.padded_active_rows
    positions = torch.empty(
        (len(bucket.sample_indices), width), dtype=torch.long, device=device
    )
    valid = torch.zeros_like(positions, dtype=torch.bool)
    active_rank = torch.full(
        (len(bucket.sample_indices), bucket.total_rows),
        -1,
        dtype=torch.long,
        device=device,
    )
    for local_index, (sample_index, expected_count) in enumerate(
        zip(bucket.sample_indices, bucket.active_counts)
    ):
        row_positions = torch.nonzero(
            active_mask[sample_index], as_tuple=False
        ).flatten()
        if int(row_positions.numel()) != expected_count:
            raise ValueError("active mask changed after bucket planning")
        if expected_count == 0:
            # A sample may finish before the other members of its execution
            # bucket. Keep its padded lanes inert and point their gathers at a
            # benign row. ``active_rank`` remains -1 everywhere, so scatter
            # restores every row from the frozen representation and none of
            # the dummy work becomes observable.
            positions[local_index] = 0
            continue
        positions[local_index, :expected_count] = row_positions
        # Repeating the last genuine row makes padded scatter writes benign:
        # every duplicate writes the exact same projected value.
        positions[local_index, expected_count:] = row_positions[-1]
        valid[local_index, :expected_count] = True
        active_rank[local_index, row_positions] = torch.arange(
            expected_count, dtype=torch.long, device=device
        )
    return MaterializedActiveRowBucket(
        plan=bucket,
        sample_indices=sample_indices,
        positions=positions,
        valid=valid,
        active_rank=active_rank,
    )


if triton is not None:

    @triton.jit
    def _gather_rmsnorm_kernel(
        x_ptr,
        positions_ptr,
        weight_ptr,
        raw_ptr,
        normed_ptr,
        stride_xb: tl.constexpr,
        stride_xs: tl.constexpr,
        stride_xd: tl.constexpr,
        active_width: tl.constexpr,
        hidden_dim: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // active_width
        active_row = row - batch * active_width
        position = tl.load(positions_ptr + row)
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_dim
        values = tl.load(
            x_ptr
            + batch * stride_xb
            + position * stride_xs
            + offsets * stride_xd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / hidden_dim
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output_offsets = row * hidden_dim + offsets
        tl.store(raw_ptr + output_offsets, values, mask=mask)
        tl.store(
            normed_ptr + output_offsets,
            values * tl.rsqrt(variance + eps) * weight,
            mask=mask,
        )


    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        hidden_dim: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_dim
        values = tl.load(
            x_ptr + row * hidden_dim + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / hidden_dim
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            output_ptr + row * hidden_dim + offsets,
            values * tl.rsqrt(variance + eps) * weight,
            mask=mask,
        )


    @triton.jit
    def _rope_kv_update_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        positions_ptr,
        sin_ptr,
        cos_ptr,
        q_out_ptr,
        k_out_ptr,
        full_k_ptr,
        full_v_ptr,
        q_stride_b: tl.constexpr,
        q_stride_a: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_a: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_a: tl.constexpr,
        v_stride_d: tl.constexpr,
        positions_stride_b: tl.constexpr,
        positions_stride_a: tl.constexpr,
        rope_stride_s: tl.constexpr,
        rope_stride_d: tl.constexpr,
        packed_stride_b: tl.constexpr,
        packed_stride_h: tl.constexpr,
        packed_stride_a: tl.constexpr,
        packed_stride_d: tl.constexpr,
        full_stride_b: tl.constexpr,
        full_stride_h: tl.constexpr,
        full_stride_s: tl.constexpr,
        full_stride_d: tl.constexpr,
        heads: tl.constexpr,
        active_width: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        batch = row // active_width
        active_row = row - batch * active_width
        position = tl.load(
            positions_ptr
            + batch * positions_stride_b
            + active_row * positions_stride_a
        )
        offsets = tl.arange(0, block_size)
        mask = offsets < head_dim
        partner = tl.where(
            offsets < head_dim // 2,
            offsets + head_dim // 2,
            offsets - head_dim // 2,
        )
        sign = tl.where(offsets < head_dim // 2, -1.0, 1.0)
        q_base = batch * q_stride_b + active_row * q_stride_a + head * head_dim
        k_base = batch * k_stride_b + active_row * k_stride_a + head * head_dim
        v_base = batch * v_stride_b + active_row * v_stride_a + head * head_dim
        q = tl.load(q_ptr + q_base + offsets * q_stride_d, mask=mask, other=0.0).to(
            tl.float32
        )
        q_partner = tl.load(
            q_ptr + q_base + partner * q_stride_d, mask=mask, other=0.0
        ).to(tl.float32)
        k = tl.load(k_ptr + k_base + offsets * k_stride_d, mask=mask, other=0.0).to(
            tl.float32
        )
        k_partner = tl.load(
            k_ptr + k_base + partner * k_stride_d, mask=mask, other=0.0
        ).to(tl.float32)
        rope_offset = position * rope_stride_s + offsets * rope_stride_d
        sin = tl.load(sin_ptr + rope_offset, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + rope_offset, mask=mask, other=0.0).to(tl.float32)
        q_rot = q * cos + sign * q_partner * sin
        k_rot = k * cos + sign * k_partner * sin
        packed_offset = (
            batch * packed_stride_b
            + head * packed_stride_h
            + active_row * packed_stride_a
            + offsets * packed_stride_d
        )
        full_offset = (
            batch * full_stride_b
            + head * full_stride_h
            + position * full_stride_s
            + offsets * full_stride_d
        )
        value = tl.load(
            v_ptr + v_base + offsets * v_stride_d, mask=mask, other=0.0
        )
        tl.store(q_out_ptr + packed_offset, q_rot, mask=mask)
        tl.store(k_out_ptr + packed_offset, k_rot, mask=mask)
        tl.store(full_k_ptr + full_offset, k_rot, mask=mask)
        tl.store(full_v_ptr + full_offset, value, mask=mask)


    @triton.jit
    def _restore_scatter_kernel(
        active_ptr,
        frozen_ptr,
        rank_ptr,
        output_ptr,
        seq_len: tl.constexpr,
        hidden_dim: tl.constexpr,
        active_width: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // seq_len
        sequence_row = row - batch * seq_len
        active_rank = tl.load(rank_ptr + row)
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_dim
        frozen_offset = row * hidden_dim + offsets
        active_offset = (
            (batch * active_width + tl.maximum(active_rank, 0)) * hidden_dim
            + offsets
        )
        frozen = tl.load(frozen_ptr + frozen_offset, mask=mask, other=0.0)
        active = tl.load(active_ptr + active_offset, mask=mask, other=0.0)
        tl.store(
            output_ptr + frozen_offset,
            tl.where(active_rank >= 0, active, frozen),
            mask=mask,
        )


    @triton.jit
    def _kv_update_kernel(
        k_ptr,
        v_ptr,
        positions_ptr,
        full_k_ptr,
        full_v_ptr,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_a: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_a: tl.constexpr,
        v_stride_d: tl.constexpr,
        positions_stride_b: tl.constexpr,
        positions_stride_a: tl.constexpr,
        full_stride_b: tl.constexpr,
        full_stride_h: tl.constexpr,
        full_stride_s: tl.constexpr,
        full_stride_d: tl.constexpr,
        active_width: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        batch = row // active_width
        active_row = row - batch * active_width
        position = tl.load(
            positions_ptr
            + batch * positions_stride_b
            + active_row * positions_stride_a
        )
        offsets = tl.arange(0, block_size)
        mask = offsets < head_dim
        k_offset = (
            batch * k_stride_b
            + head * k_stride_h
            + active_row * k_stride_a
            + offsets * k_stride_d
        )
        v_offset = (
            batch * v_stride_b
            + head * v_stride_h
            + active_row * v_stride_a
            + offsets * v_stride_d
        )
        full_offset = (
            batch * full_stride_b
            + head * full_stride_h
            + position * full_stride_s
            + offsets * full_stride_d
        )
        k = tl.load(k_ptr + k_offset, mask=mask, other=0.0)
        v = tl.load(v_ptr + v_offset, mask=mask, other=0.0)
        tl.store(full_k_ptr + full_offset, k, mask=mask)
        tl.store(full_v_ptr + full_offset, v, mask=mask)


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _can_use_triton(*tensors: torch.Tensor) -> bool:
    return triton is not None and all(tensor.is_cuda for tensor in tensors)


def gather_rmsnorm(
    x: torch.Tensor,
    positions: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    fused: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather rows and return both raw and RMS-normalized packed tensors."""
    if x.ndim != 3 or positions.ndim != 2 or weight.ndim != 1:
        raise ValueError("invalid gather/RMSNorm tensor ranks")
    batch, _seq_len, hidden_dim = x.shape
    if positions.shape[0] != batch or weight.numel() != hidden_dim:
        raise ValueError("gather/RMSNorm shapes disagree")
    width = positions.shape[1]
    if fused and _can_use_triton(x, positions, weight):
        raw = torch.empty(
            (batch, width, hidden_dim), device=x.device, dtype=x.dtype
        )
        normed = torch.empty_like(raw)
        block_size = triton.next_power_of_2(hidden_dim)
        if block_size * x.element_size() > 65536:
            raise RuntimeError("fused RMSNorm row exceeds Triton's 64KB limit")
        _gather_rmsnorm_kernel[(batch * width,)](
            x,
            positions,
            weight,
            raw,
            normed,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            width,
            hidden_dim,
            eps,
            block_size,
            num_warps=8,
        )
        return raw, normed
    gather_index = positions.unsqueeze(-1).expand(-1, -1, hidden_dim)
    raw = torch.gather(x, 1, gather_index)
    working = raw.float()
    variance = working.square().mean(dim=-1, keepdim=True)
    normed = (working * torch.rsqrt(variance + eps)).to(raw.dtype) * weight
    return raw, normed


def rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, *, fused: bool
) -> torch.Tensor:
    """Inference-only RMSNorm with an optional single-kernel CUDA path."""
    hidden_dim = x.shape[-1]
    if weight.ndim != 1 or weight.numel() != hidden_dim:
        raise ValueError("RMSNorm weight shape disagrees with input")
    if fused and _can_use_triton(x, weight):
        output = torch.empty_like(x)
        rows = x.numel() // hidden_dim
        block_size = triton.next_power_of_2(hidden_dim)
        if block_size * x.element_size() > 65536:
            raise RuntimeError("fused RMSNorm row exceeds Triton's 64KB limit")
        _rmsnorm_kernel[(rows,)](
            x,
            weight,
            output,
            hidden_dim,
            eps,
            block_size,
            num_warps=8,
        )
        return output
    working = x.float()
    variance = working.square().mean(dim=-1, keepdim=True)
    return (working * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def rope_and_update_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    pos_sin: torch.Tensor,
    pos_cos: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    *,
    heads: int,
    fused: bool,
    fused_update: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply LLaDA RoPE and update persistent full-reference K/V workspaces."""
    batch, width, q_dim = q.shape
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("fused LLaDA path requires equal Q/K/V shapes")
    if q_dim % heads:
        raise ValueError("projection width must divide evenly into heads")
    head_dim = q_dim // heads
    seq_len = full_k.shape[2]
    expected_workspace_shape = (batch, heads, seq_len, head_dim)
    if full_k.shape != expected_workspace_shape or full_v.shape != expected_workspace_shape:
        raise ValueError("full K/V workspace shape disagrees with projections")
    if fused and (not full_k.is_contiguous() or not full_v.is_contiguous()):
        raise ValueError("fused K/V workspaces must be contiguous [B, H, S, D]")
    if fused and _can_use_triton(
        q, k, v, positions, pos_sin, pos_cos, full_k, full_v
    ):
        q_out = torch.empty(
            (batch, heads, width, head_dim), device=q.device, dtype=q.dtype
        )
        k_out = torch.empty_like(q_out)
        block_size = triton.next_power_of_2(head_dim)
        _rope_kv_update_kernel[(batch * width, heads)](
            q,
            k,
            v,
            positions,
            pos_sin,
            pos_cos,
            q_out,
            k_out,
            full_k,
            full_v,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            positions.stride(0),
            positions.stride(1),
            pos_sin.stride(2),
            pos_sin.stride(3),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            q_out.stride(3),
            full_k.stride(0),
            full_k.stride(1),
            full_k.stride(2),
            full_k.stride(3),
            heads,
            width,
            seq_len,
            head_dim,
            block_size,
            num_warps=4,
        )
        return q_out, k_out

    q_heads = q.view(batch, width, heads, head_dim).transpose(1, 2)
    k_heads = k.view(batch, width, heads, head_dim).transpose(1, 2)
    sin = pos_sin[0, 0].index_select(0, positions.flatten()).view(
        batch, 1, width, head_dim
    )
    cos = pos_cos[0, 0].index_select(0, positions.flatten()).view(
        batch, 1, width, head_dim
    )

    def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor.float().chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    q_out = (q_heads.float() * cos + rotate_half(q_heads) * sin).to(q.dtype)
    k_out = (k_heads.float() * cos + rotate_half(k_heads) * sin).to(k.dtype)
    v_heads = v.view(batch, width, heads, head_dim).transpose(1, 2)
    if fused_update and _can_use_triton(
        k_out, v_heads, positions, full_k, full_v
    ):
        block_size = triton.next_power_of_2(head_dim)
        _kv_update_kernel[(batch * width, heads)](
            k_out,
            v_heads,
            positions,
            full_k,
            full_v,
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            k_out.stride(3),
            v_heads.stride(0),
            v_heads.stride(1),
            v_heads.stride(2),
            v_heads.stride(3),
            positions.stride(0),
            positions.stride(1),
            full_k.stride(0),
            full_k.stride(1),
            full_k.stride(2),
            full_k.stride(3),
            width,
            head_dim,
            block_size,
            num_warps=4,
        )
    else:
        scatter_index = positions[:, None, :, None].expand(
            batch, heads, width, head_dim
        )
        full_k.scatter_(2, scatter_index, k_out)
        full_v.scatter_(2, scatter_index, v_heads)
    return q_out, k_out


def restore_and_scatter(
    active: torch.Tensor,
    frozen: torch.Tensor,
    active_rank: torch.Tensor,
    *,
    fused: bool,
) -> torch.Tensor:
    """Restore frozen rows and scatter active packed rows in one pass."""
    batch, seq_len, hidden_dim = frozen.shape
    if active.shape[0] != batch or active.shape[2] != hidden_dim:
        raise ValueError("active and frozen tensors disagree")
    if active_rank.shape != (batch, seq_len):
        raise ValueError("active_rank shape disagrees with frozen tensor")
    active_width = active.shape[1]
    if fused and _can_use_triton(active, frozen, active_rank):
        output = torch.empty_like(frozen)
        block_size = triton.next_power_of_2(hidden_dim)
        _restore_scatter_kernel[(batch * seq_len,)](
            active,
            frozen,
            active_rank,
            output,
            seq_len,
            hidden_dim,
            active_width,
            block_size,
            num_warps=8,
        )
        return output
    output = frozen.clone()
    for batch_index in range(batch):
        positions = torch.nonzero(
            active_rank[batch_index] >= 0, as_tuple=False
        ).flatten()
        ranks = active_rank[batch_index, positions]
        output[batch_index, positions] = active[batch_index, ranks]
    return output


@dataclass
class _BucketWorkspace:
    materialized: MaterializedActiveRowBucket
    frozen_outputs: torch.Tensor
    full_k: torch.Tensor
    full_v: torch.Tensor


class BucketedPackedBlock:
    """Execute one LLaDA block over heterogeneous active rows.

    This correctness-first integration targets inference on LLaDA's Llama
    block (RMSNorm, equal Q/K/V heads, no dropout).  It is used by the backend
    benchmark before promotion into the online decoder.
    """

    def __init__(
        self,
        block: torch.nn.Module,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        strategy: str = "bucketed",
        fused: bool = True,
        fusion_mode: str | None = None,
        width_multiple: int = 16,
        max_bucket_size: int = 8,
        max_padding_fraction: float = 0.25,
        launch_penalty_rows: int = 64,
        plans: Sequence[ActiveRowBucket] | None = None,
    ) -> None:
        if x.ndim != 3 or active_mask.shape != x.shape[:2]:
            raise ValueError("x and active_mask shapes disagree")
        if active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be boolean")
        if strategy not in {"per_sample", "global", "bucketed"}:
            raise ValueError(f"unknown bucket strategy: {strategy}")
        self.block = block
        self.x = x
        self.active_mask = active_mask
        if fusion_mode is None:
            fusion_mode = "full" if fused else "torch"
        if fusion_mode not in {"torch", "exact", "memory", "full"}:
            raise ValueError("fusion_mode must be torch, exact, memory, or full")
        self.fusion_mode = fusion_mode
        self.fuse_norms = fusion_mode == "full"
        self.fuse_rope = fusion_mode in {"memory", "full"}
        self.fuse_kv_update = fusion_mode in {"exact", "memory", "full"}
        self.fuse_scatter = fusion_mode in {"exact", "memory", "full"}
        self.batch, self.seq_len, self.hidden_dim = x.shape
        self.heads = int(block.config.n_heads)
        if int(block.config.effective_n_kv_heads) != self.heads:
            raise ValueError("fused backend currently requires equal Q and K/V heads")
        if getattr(block, "q_norm", None) is not None or getattr(block, "k_norm", None) is not None:
            raise ValueError("fused backend does not yet support attention Q/K normalization")
        if not all(hasattr(block, name) for name in ("q_proj", "k_proj", "v_proj", "up_proj")):
            raise ValueError("fused backend currently requires the LLaDA Llama block")
        if getattr(block.dropout, "p", 0.0) != 0.0 or block.training:
            raise ValueError("fused backend is inference-only and requires dropout disabled")
        for norm in (block.attn_norm, block.ff_norm):
            if norm.__class__.__name__ != "RMSLayerNorm" or norm.weight is None or norm.bias is not None:
                raise ValueError("fused backend requires affine, bias-free RMSNorm")

        active_counts = [int(value) for value in active_mask.sum(dim=1).tolist()]
        if plans is not None:
            plans = list(plans)
            planned_indices = [
                index for plan in plans for index in plan.sample_indices
            ]
            if sorted(planned_indices) != list(range(self.batch)):
                raise ValueError("explicit plans must contain every sample exactly once")
            for plan in plans:
                if plan.total_rows != self.seq_len:
                    raise ValueError("explicit plan sequence length disagrees with x")
                expected = tuple(active_counts[index] for index in plan.sample_indices)
                if plan.active_counts != expected:
                    raise ValueError("explicit plan active counts disagree with active_mask")
        elif strategy == "per_sample":
            plans = [
                ActiveRowBucket(
                    sample_indices=(index,),
                    active_counts=(count,),
                    padded_active_rows=_round_bucket_width(
                        count, width_multiple, self.seq_len
                    ),
                    total_rows=self.seq_len,
                )
                for index, count in enumerate(active_counts)
            ]
        elif strategy == "global":
            width = _round_bucket_width(max(active_counts), width_multiple, self.seq_len)
            plans = [
                ActiveRowBucket(
                    sample_indices=tuple(range(self.batch)),
                    active_counts=tuple(active_counts),
                    padded_active_rows=width,
                    total_rows=self.seq_len,
                )
            ]
        else:
            plans = plan_active_row_buckets(
                active_counts,
                self.seq_len,
                width_multiple=width_multiple,
                max_bucket_size=max_bucket_size,
                max_padding_fraction=max_padding_fraction,
                launch_penalty_rows=launch_penalty_rows,
            )
        self.plans = plans

        with torch.no_grad():
            dense_outputs, _ = block(x)
            full_k, full_v = self._full_kv(x)
        self.workspaces: list[_BucketWorkspace] = []
        for plan in plans:
            materialized = materialize_active_row_bucket(active_mask, plan)
            sample_indices = materialized.sample_indices
            self.workspaces.append(
                _BucketWorkspace(
                    materialized=materialized,
                    frozen_outputs=dense_outputs.index_select(
                        0, sample_indices
                    ).detach(),
                    full_k=full_k.index_select(0, sample_indices).detach().clone(),
                    full_v=full_v.index_select(0, sample_indices).detach().clone(),
                )
            )

    def _full_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.block
        normalized = block.attn_norm(x)
        k = block.k_proj(normalized)
        v = block.v_proj(normalized)
        head_dim = self.hidden_dim // self.heads
        k = k.view(self.batch, self.seq_len, self.heads, head_dim).transpose(1, 2)
        v = v.view(self.batch, self.seq_len, self.heads, head_dim).transpose(1, 2)
        rotary = block.rotary_emb
        positions = torch.arange(self.seq_len, device=x.device).expand(self.batch, -1)
        pos_sin, pos_cos = rotary.get_rotary_embedding(self.seq_len, x.device)
        # The Triton updater addresses workspaces as contiguous [B, H, S, D].
        # ``k`` and ``v`` above are transposed views, so empty_like would
        # preserve incompatible non-contiguous strides.
        workspace_shape = (self.batch, self.heads, self.seq_len, head_dim)
        workspace_k = torch.empty(
            workspace_shape, device=x.device, dtype=x.dtype
        )
        workspace_v = torch.empty_like(workspace_k)
        _q_dummy = torch.zeros(
            (self.batch, self.seq_len, self.hidden_dim),
            device=x.device,
            dtype=x.dtype,
        )
        _q, k = rope_and_update_kv(
            _q_dummy,
            k.transpose(1, 2).reshape(self.batch, self.seq_len, self.hidden_dim),
            v.transpose(1, 2).reshape(self.batch, self.seq_len, self.hidden_dim),
            positions,
            pos_sin,
            pos_cos,
            workspace_k,
            workspace_v,
            heads=self.heads,
            fused=self.fuse_rope,
            fused_update=self.fuse_kv_update,
        )
        del _q, k, _q_dummy
        return workspace_k, workspace_v

    def _run_bucket(self, workspace: _BucketWorkspace) -> torch.Tensor:
        block = self.block
        materialized = workspace.materialized
        x = self.x.index_select(0, materialized.sample_indices)
        x_active, x_normed = gather_rmsnorm(
            x,
            materialized.positions,
            block.attn_norm.weight,
            float(block.attn_norm.eps),
            fused=self.fuse_norms,
        )
        q = block.q_proj(x_normed)
        k = block.k_proj(x_normed)
        v = block.v_proj(x_normed)
        pos_sin, pos_cos = block.rotary_emb.get_rotary_embedding(
            self.seq_len, x.device
        )
        q, _k = rope_and_update_kv(
            q,
            k,
            v,
            materialized.positions,
            pos_sin,
            pos_cos,
            workspace.full_k,
            workspace.full_v,
            heads=self.heads,
            fused=self.fuse_rope,
            fused_update=self.fuse_kv_update,
        )
        del _k
        attention = block._scaled_dot_product_attention(
            q, workspace.full_k, workspace.full_v, attn_mask=None
        )
        width = materialized.plan.padded_active_rows
        attention = attention.transpose(1, 2).contiguous().view(
            x.shape[0], width, self.hidden_dim
        )
        active_output = x_active + block.attn_out(attention)
        residual = active_output
        normalized = rmsnorm(
            active_output,
            block.ff_norm.weight,
            float(block.ff_norm.eps),
            fused=self.fuse_norms,
        )
        ff = block.act(block.ff_proj(normalized)) * block.up_proj(normalized)
        active_output = residual + block.ff_out(ff)
        return restore_and_scatter(
            active_output,
            workspace.frozen_outputs,
            materialized.active_rank,
            fused=self.fuse_scatter,
        )

    def __call__(self) -> torch.Tensor:
        if len(self.workspaces) == 1 and self.workspaces[0].materialized.plan.sample_indices == tuple(
            range(self.batch)
        ):
            return self._run_bucket(self.workspaces[0])
        output = torch.empty_like(self.x)
        for workspace in self.workspaces:
            bucket_output = self._run_bucket(workspace)
            output.index_copy_(
                0, workspace.materialized.sample_indices, bucket_output
            )
        return output

    def plan_summary(self) -> dict[str, Any]:
        actual = sum(plan.actual_active_rows for plan in self.plans)
        padded = sum(
            len(plan.sample_indices) * plan.padded_active_rows for plan in self.plans
        )
        return {
            "bucket_count": len(self.plans),
            "bucket_widths": [plan.padded_active_rows for plan in self.plans],
            "bucket_sizes": [len(plan.sample_indices) for plan in self.plans],
            "actual_active_rows": actual,
            "executed_active_rows": padded,
            "padding_rows": padded - actual,
            "padding_fraction": (padded - actual) / max(actual, 1),
        }


@dataclass
class _RuntimeBucketWorkspace:
    materialized: MaterializedActiveRowBucket
    full_k: dict[int, torch.Tensor]
    full_v: dict[int, torch.Tensor]


class BucketedReferenceRuntime:
    """Batched representation freeze plus independently gated compute lock.

    Dense hooks implement Phase 2.  ``activate_compute`` makes the one-way
    Phase-3 transition and installs the exact-fusion bucketed wrappers.  The
    caller supplies the compute plan; this runtime never infers that a token or
    representation is safe to lock.
    """

    def __init__(
        self,
        blocks: Sequence[torch.nn.Module],
        *,
        batch_size: int,
        total_len: int,
        fusion_mode: str = "exact",
    ) -> None:
        if not blocks:
            raise ValueError("at least one transformer block is required")
        if batch_size <= 0 or total_len <= 0:
            raise ValueError("batch_size and total_len must be positive")
        if fusion_mode not in {"exact", "memory", "full"}:
            raise ValueError("runtime fusion_mode must be exact, memory, or full")
        self.blocks = list(blocks)
        self.batch_size = batch_size
        self.total_len = total_len
        self.fusion_mode = fusion_mode
        self.device = next(self.blocks[0].parameters()).device
        self.locked = torch.zeros(
            (batch_size, total_len), dtype=torch.bool, device=self.device
        )
        self.locked_positions: list[set[int]] = [set() for _ in range(batch_size)]
        self.active_counts = [total_len] * batch_size
        self.frozen_inputs: dict[int, torch.Tensor] = {}
        self.frozen_outputs: dict[int, torch.Tensor] = {}
        self.frozen_k: dict[int, torch.Tensor] = {}
        self.frozen_v: dict[int, torch.Tensor] = {}
        self.last_inputs: dict[int, torch.Tensor] = {}
        self.last_outputs: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.original_forwards: dict[torch.nn.Module, Any] = {}
        self.compute_active = False
        self.runtime_buckets: list[_RuntimeBucketWorkspace] = []

    @property
    def locked_count(self) -> int:
        return sum(len(positions) for positions in self.locked_positions)

    def install_dense(self) -> None:
        if self.handles or self.original_forwards:
            raise RuntimeError("runtime is already installed")
        for layer, block in enumerate(self.blocks):
            self.handles.append(
                block.register_forward_pre_hook(
                    self._make_pre_hook(layer), with_kwargs=True
                )
            )
            self.handles.append(
                block.register_forward_hook(self._make_post_hook(layer))
            )

    def _make_pre_hook(self, layer: int):
        def hook(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ):
            x = args[0]
            self.last_inputs[layer] = x.detach()
            if self.locked_count == 0:
                return None
            x[self.locked] = self.frozen_inputs[layer][self.locked]
            return None

        return hook

    def _make_post_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any):
            x, cache = output
            self.last_outputs[layer] = x.detach()
            if self.locked_count:
                x[self.locked] = self.frozen_outputs[layer][self.locked]
            return x, cache

        return hook

    def _install_sparse_wrappers(self) -> None:
        for layer, block in enumerate(self.blocks):
            original = block.forward
            self.original_forwards[block] = original

            def wrapped(
                x: torch.Tensor,
                attention_bias: torch.Tensor | None = None,
                layer_past: Any = None,
                use_cache: bool = False,
                *,
                _block: torch.nn.Module = block,
                _layer: int = layer,
            ) -> Any:
                return self._sparse_forward(
                    _block,
                    _layer,
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )

            block.forward = wrapped  # type: ignore[method-assign]

    def _full_kv(
        self, block: torch.nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = block.attn_norm(x)
        k = block.k_proj(normalized)
        v = block.v_proj(normalized)
        heads = int(block.config.n_heads)
        if int(block.config.effective_n_kv_heads) != heads:
            raise ValueError("bucketed runtime requires equal query and K/V heads")
        head_dim = x.shape[-1] // heads
        shape = (self.batch_size, heads, self.total_len, head_dim)
        full_k = torch.empty(shape, device=x.device, dtype=x.dtype)
        full_v = torch.empty_like(full_k)
        q_dummy = torch.zeros_like(k)
        positions = torch.arange(
            self.total_len, device=x.device, dtype=torch.long
        ).expand(self.batch_size, -1)
        pos_sin, pos_cos = block.rotary_emb.get_rotary_embedding(
            self.total_len, x.device
        )
        rope_and_update_kv(
            q_dummy,
            k,
            v,
            positions,
            pos_sin,
            pos_cos,
            full_k,
            full_v,
            heads=heads,
            fused=False,
            fused_update=False,
        )
        return full_k, full_v

    def activate_compute(self, plans: Sequence[ActiveRowBucket]) -> None:
        if self.compute_active:
            raise RuntimeError("compute lock is already active")
        if self.locked_count == 0:
            raise RuntimeError("compute lock requires representation-locked rows")
        if any(layer not in self.last_inputs for layer in range(len(self.blocks))):
            raise RuntimeError("compute activation requires a completed dense capture")
        full_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, block in enumerate(self.blocks):
            full_k, full_v = self._full_kv(block, self.last_inputs[layer])
            full_states[layer] = (full_k, full_v)
            self.frozen_k[layer] = torch.zeros_like(full_k)
            self.frozen_v[layer] = torch.zeros_like(full_v)
            for batch_index, positions in enumerate(self.locked_positions):
                if not positions:
                    continue
                position_tensor = torch.tensor(
                    sorted(positions), dtype=torch.long, device=self.device
                )
                self.frozen_k[layer][batch_index, :, position_tensor] = full_k[
                    batch_index, :, position_tensor
                ]
                self.frozen_v[layer][batch_index, :, position_tensor] = full_v[
                    batch_index, :, position_tensor
                ]
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.compute_active = True
        self._set_compute_plan(plans, full_states=full_states)
        self._install_sparse_wrappers()

    def set_compute_plan(self, plans: Sequence[ActiveRowBucket]) -> None:
        if not self.compute_active:
            raise RuntimeError("compute plan can change only after activation")
        self._set_compute_plan(plans, full_states=None)

    def _set_compute_plan(
        self,
        plans: Sequence[ActiveRowBucket],
        *,
        full_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> None:
        plans = list(plans)
        active_mask = ~self.locked
        materialized = [
            materialize_active_row_bucket(active_mask, plan) for plan in plans
        ]
        signatures = [
            (item.plan.sample_indices, item.plan.padded_active_rows)
            for item in materialized
        ]
        old_signatures = [
            (item.materialized.plan.sample_indices, item.materialized.plan.padded_active_rows)
            for item in self.runtime_buckets
        ]
        if full_states is None and signatures == old_signatures:
            for state, item in zip(self.runtime_buckets, materialized):
                state.materialized = item
            return

        runtime_buckets: list[_RuntimeBucketWorkspace] = []
        for item in materialized:
            sample_indices = item.sample_indices
            bucket_k: dict[int, torch.Tensor] = {}
            bucket_v: dict[int, torch.Tensor] = {}
            for layer in range(len(self.blocks)):
                if full_states is None:
                    source_k = self.frozen_k[layer]
                    source_v = self.frozen_v[layer]
                else:
                    source_k, source_v = full_states[layer]
                bucket_k[layer] = source_k.index_select(0, sample_indices).clone()
                bucket_v[layer] = source_v.index_select(0, sample_indices).clone()
            runtime_buckets.append(
                _RuntimeBucketWorkspace(item, bucket_k, bucket_v)
            )
        self.runtime_buckets = runtime_buckets

    def commit(self, positions: Sequence[tuple[int, int]]) -> None:
        if not positions:
            return
        if len(set(positions)) != len(positions):
            raise ValueError("duplicate representation-lock event")
        for batch_index, position in positions:
            if not (0 <= batch_index < self.batch_size):
                raise ValueError("representation-lock batch index is out of range")
            if not (0 <= position < self.total_len):
                raise ValueError("representation-lock position is out of range")
            if position in self.locked_positions[batch_index]:
                raise ValueError("position is already representation-locked")
        batch_tensor = torch.tensor(
            [item[0] for item in positions], dtype=torch.long, device=self.device
        )
        position_tensor = torch.tensor(
            [item[1] for item in positions], dtype=torch.long, device=self.device
        )
        for layer in range(len(self.blocks)):
            if layer not in self.last_inputs or layer not in self.last_outputs:
                raise RuntimeError(f"missing hidden capture for layer {layer}")
            if layer not in self.frozen_inputs:
                self.frozen_inputs[layer] = torch.zeros_like(self.last_inputs[layer])
                self.frozen_outputs[layer] = torch.zeros_like(self.last_outputs[layer])
            self.frozen_inputs[layer][batch_tensor, position_tensor] = self.last_inputs[
                layer
            ][batch_tensor, position_tensor]
            self.frozen_outputs[layer][batch_tensor, position_tensor] = self.last_outputs[
                layer
            ][batch_tensor, position_tensor]
            if self.compute_active:
                for batch_index, position in positions:
                    state, local_batch = self._locate_runtime_sample(batch_index)
                    self.frozen_k[layer][batch_index, :, position] = state.full_k[
                        layer
                    ][local_batch, :, position]
                    self.frozen_v[layer][batch_index, :, position] = state.full_v[
                        layer
                    ][local_batch, :, position]
        for batch_index, position in positions:
            self.locked[batch_index, position] = True
            self.locked_positions[batch_index].add(position)
            self.active_counts[batch_index] -= 1

    def _locate_runtime_sample(
        self, sample_index: int
    ) -> tuple[_RuntimeBucketWorkspace, int]:
        for state in self.runtime_buckets:
            try:
                local = state.materialized.plan.sample_indices.index(sample_index)
            except ValueError:
                continue
            return state, local
        raise RuntimeError(f"sample {sample_index} is missing from compute plan")

    @staticmethod
    def _bucket_attention_bias(
        attention_bias: torch.Tensor | None,
        sample_indices: torch.Tensor,
        positions: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor | None:
        if attention_bias is None:
            return None
        if attention_bias.shape[0] == 1:
            bias = attention_bias.expand(sample_indices.numel(), *attention_bias.shape[1:])
        else:
            bias = attention_bias.index_select(0, sample_indices)
        bias = bias[..., :seq_len]
        if bias.shape[-2] == 1:
            return bias
        gather_index = positions[:, None, :, None].expand(
            positions.shape[0], bias.shape[1], positions.shape[1], seq_len
        )
        return torch.gather(bias, 2, gather_index)

    def _sparse_forward(
        self,
        block: torch.nn.Module,
        layer: int,
        x: torch.Tensor,
        *,
        attention_bias: torch.Tensor | None,
        layer_past: Any,
        use_cache: bool,
    ) -> tuple[torch.Tensor, None]:
        if layer_past is not None or use_cache:
            raise RuntimeError("bucketed compute lock does not support a past cache")
        if x.shape[:2] != (self.batch_size, self.total_len):
            raise ValueError("runtime batch/sequence shape changed after activation")
        self.last_inputs[layer] = x.detach()
        output = torch.empty_like(x)
        heads = int(block.config.n_heads)
        hidden_dim = x.shape[-1]
        for state in self.runtime_buckets:
            item = state.materialized
            sample_x = x.index_select(0, item.sample_indices)
            x_active, normalized = gather_rmsnorm(
                sample_x,
                item.positions,
                block.attn_norm.weight,
                float(block.attn_norm.eps),
                fused=self.fusion_mode == "full",
            )
            q = block.q_proj(normalized)
            k = block.k_proj(normalized)
            v = block.v_proj(normalized)
            pos_sin, pos_cos = block.rotary_emb.get_rotary_embedding(
                self.total_len, x.device
            )
            q, _k = rope_and_update_kv(
                q,
                k,
                v,
                item.positions,
                pos_sin,
                pos_cos,
                state.full_k[layer],
                state.full_v[layer],
                heads=heads,
                fused=self.fusion_mode in {"memory", "full"},
                fused_update=True,
            )
            del _k
            row_bias = self._bucket_attention_bias(
                attention_bias,
                item.sample_indices,
                item.positions,
                self.total_len,
            )
            attention = block._scaled_dot_product_attention(
                q,
                state.full_k[layer],
                state.full_v[layer],
                attn_mask=row_bias,
            )
            width = item.plan.padded_active_rows
            attention = attention.transpose(1, 2).contiguous().view(
                item.sample_indices.numel(), width, hidden_dim
            )
            active_output = x_active + block.attn_out(attention)
            residual = active_output
            normalized = rmsnorm(
                active_output,
                block.ff_norm.weight,
                float(block.ff_norm.eps),
                fused=self.fusion_mode == "full",
            )
            ff = block.act(block.ff_proj(normalized)) * block.up_proj(normalized)
            active_output = residual + block.ff_out(ff)
            frozen = self.frozen_outputs[layer].index_select(
                0, item.sample_indices
            )
            bucket_output = restore_and_scatter(
                active_output,
                frozen,
                item.active_rank,
                fused=True,
            )
            output.index_copy_(0, item.sample_indices, bucket_output)
        self.last_outputs[layer] = output.detach()
        return output, None

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for block, original in self.original_forwards.items():
            block.forward = original  # type: ignore[method-assign]
        self.original_forwards.clear()
