from __future__ import annotations

from types import SimpleNamespace
import copy

import pytest
import torch
import torch.nn.functional as F

from regret_remasking.phase_lock_sparse import (
    ActiveRowBucket,
    BucketedPackedBlock,
    BucketedReferenceRuntime,
    FusedLatencyTable,
    choose_fused_compute_plan,
    gather_rmsnorm,
    materialize_active_row_bucket,
    plan_active_row_buckets,
    restore_and_scatter,
)


def test_bucket_planner_groups_similar_counts_and_splits_outlier() -> None:
    plans = plan_active_row_buckets(
        [80, 78, 75, 40],
        total_rows=96,
        width_multiple=8,
        max_bucket_size=4,
        max_padding_fraction=0.15,
        launch_penalty_rows=64,
    )
    assert [plan.sample_indices for plan in plans] == [(0, 1, 2), (3,)]
    assert [plan.padded_active_rows for plan in plans] == [80, 40]
    assert sum(plan.actual_active_rows for plan in plans) == 273


def test_bucket_planner_never_drops_or_duplicates_samples() -> None:
    counts = [57, 21, 54, 55, 20, 56, 19]
    plans = plan_active_row_buckets(
        counts,
        total_rows=64,
        width_multiple=8,
        max_bucket_size=3,
        max_padding_fraction=0.2,
    )
    indices = [index for plan in plans for index in plan.sample_indices]
    assert sorted(indices) == list(range(len(counts)))
    for plan in plans:
        assert plan.padding_fraction <= 0.2 or len(plan.sample_indices) == 1


def _latency_table() -> FusedLatencyTable:
    rows = []
    for batch_size in range(1, 5):
        for active_rows in (32, 64, 96):
            rows.append(
                {
                    "sequence_length": 96,
                    "batch_size": batch_size,
                    "active_rows": active_rows,
                    "median_ms": 0.10
                    + 0.001 * batch_size * active_rows,
                    "dense_median_ms": 1.0,
                }
            )
    return FusedLatencyTable.from_rows(rows)


def test_latency_table_interpolates_and_compute_plan_abstains() -> None:
    table = _latency_table()
    assert table.predict_sparse_ms(2, 48, 96) == pytest.approx(0.196)
    assert table.predict_sparse_ms(2, 96, 192) == pytest.approx(0.392)
    assert table.dense_ms(192, 2) == pytest.approx(2.0)
    plan = choose_fused_compute_plan(
        [96, 96, 96, 96],
        96,
        table,
        width_multiple=32,
        min_expected_gain=0.55,
    )
    assert plan.backend == "dense"
    assert plan.predicted_gain == 0.0


def test_latency_calibrated_compute_plan_can_choose_multiple_buckets() -> None:
    entries = []
    for batch_size in range(1, 5):
        for active_rows in (32, 64, 96):
            latency = 0.2 + 0.002 * batch_size * active_rows
            if batch_size == 4 and active_rows == 96:
                latency = 2.0
            entries.append(
                {
                    "sequence_length": 96,
                    "batch_size": batch_size,
                    "active_rows": active_rows,
                    "median_ms": latency,
                    "dense_median_ms": 3.0,
                }
            )
    plan = choose_fused_compute_plan(
        [96, 96, 32, 32],
        96,
        FusedLatencyTable.from_rows(entries),
        width_multiple=32,
        max_padding_fraction=1.0,
        min_expected_gain=0.0,
    )
    assert plan.backend == "fused_bucketed"
    assert [bucket.active_counts for bucket in plan.buckets] == [(96, 96), (32, 32)]


def test_materialized_bucket_repeats_last_real_position_for_padding() -> None:
    active = torch.tensor(
        [
            [True, False, True, False, True, False, False, False],
            [False, True, True, True, False, True, False, False],
        ]
    )
    plan = plan_active_row_buckets(
        [3, 4],
        total_rows=8,
        width_multiple=4,
        max_padding_fraction=1.0,
    )[0]
    bucket = materialize_active_row_bucket(active, plan)
    for local, sample in enumerate(plan.sample_indices):
        count = plan.active_counts[local]
        real = torch.nonzero(active[sample], as_tuple=False).flatten()
        torch.testing.assert_close(bucket.positions[local, :count], real)
        assert bool((bucket.positions[local, count:] == real[-1]).all())
        torch.testing.assert_close(
            bucket.active_rank[local, real], torch.arange(count)
        )


def test_materialized_bucket_accepts_finished_sample() -> None:
    active = torch.tensor(
        [
            [False, False, False, False],
            [False, True, False, True],
        ]
    )
    plan = ActiveRowBucket(
        sample_indices=(0, 1),
        active_counts=(0, 2),
        padded_active_rows=2,
        total_rows=4,
    )

    bucket = materialize_active_row_bucket(active, plan)

    assert bucket.positions.tolist() == [[0, 0], [1, 3]]
    assert bucket.valid.tolist() == [[False, False], [True, True]]
    assert bucket.active_rank.tolist() == [
        [-1, -1, -1, -1],
        [-1, 0, -1, 1],
    ]


def test_cpu_gather_rmsnorm_and_restore_scatter() -> None:
    x = torch.arange(2 * 5 * 4, dtype=torch.float32).view(2, 5, 4) + 1.0
    positions = torch.tensor([[0, 2, 4], [1, 3, 3]])
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0])
    raw, normalized = gather_rmsnorm(
        x, positions, weight, 1e-5, fused=False
    )
    expected_raw = torch.gather(x, 1, positions.unsqueeze(-1).expand(-1, -1, 4))
    expected_normalized = (
        expected_raw
        * torch.rsqrt(expected_raw.square().mean(dim=-1, keepdim=True) + 1e-5)
        * weight
    )
    torch.testing.assert_close(raw, expected_raw)
    torch.testing.assert_close(normalized, expected_normalized)

    frozen = torch.full((2, 5, 4), -1.0)
    active_rank = torch.tensor([[0, -1, 1, -1, 2], [-1, 0, -1, 1, -1]])
    restored = restore_and_scatter(raw, frozen, active_rank, fused=False)
    torch.testing.assert_close(restored[0, [0, 2, 4]], raw[0])
    torch.testing.assert_close(restored[1, [1, 3]], raw[1, :2])
    assert bool((restored[0, [1, 3]] == -1).all())


class RMSLayerNorm(torch.nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.bias = None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        working = x.float()
        normalized = working * torch.rsqrt(
            working.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight


class _Rotary(torch.nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (
            10000
            ** (
                torch.arange(0, self.head_dim, 2, device=device).float()
                / self.head_dim
            )
        )
        frequencies = torch.outer(torch.arange(seq_len, device=device).float(), inv)
        positions = torch.cat((frequencies, frequencies), dim=-1)
        return positions.sin()[None, None], positions.cos()[None, None]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sin, cos = self.get_rotary_embedding(q.shape[-2], q.device)
        return (
            (q.float() * cos + self._rotate_half(q.float()) * sin).to(q.dtype),
            (k.float() * cos + self._rotate_half(k.float()) * sin).to(k.dtype),
        )


class _ToyLLaDABlock(torch.nn.Module):
    def __init__(self, width: int = 8, heads: int = 2) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.config = SimpleNamespace(
            n_heads=heads,
            effective_n_kv_heads=heads,
        )
        self.attn_norm = RMSLayerNorm(width)
        self.ff_norm = RMSLayerNorm(width)
        self.q_norm = None
        self.k_norm = None
        self.q_proj = torch.nn.Linear(width, width, bias=False)
        self.k_proj = torch.nn.Linear(width, width, bias=False)
        self.v_proj = torch.nn.Linear(width, width, bias=False)
        self.attn_out = torch.nn.Linear(width, width, bias=False)
        self.ff_proj = torch.nn.Linear(width, width * 2, bias=False)
        self.up_proj = torch.nn.Linear(width, width * 2, bias=False)
        self.ff_out = torch.nn.Linear(width * 2, width, bias=False)
        self.dropout = torch.nn.Dropout(0.0)
        self.act = torch.nn.SiLU()
        self.rotary_emb = _Rotary(width // heads)
        self.eval()

    @staticmethod
    def _scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        normalized = self.attn_norm(x)
        batch, rows, width = x.shape
        heads = self.config.n_heads
        head_dim = width // heads
        q = self.q_proj(normalized).view(batch, rows, heads, head_dim).transpose(1, 2)
        k = self.k_proj(normalized).view(batch, rows, heads, head_dim).transpose(1, 2)
        v = self.v_proj(normalized).view(batch, rows, heads, head_dim).transpose(1, 2)
        q, k = self.rotary_emb(q, k)
        attention = self._scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(1, 2).contiguous().view(batch, rows, width)
        x = x + self.attn_out(attention)
        normalized = self.ff_norm(x)
        x = x + self.ff_out(
            self.act(self.ff_proj(normalized)) * self.up_proj(normalized)
        )
        return x, None


class _ToyStack(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_ToyLLaDABlock(), _ToyLLaDABlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x, _ = block(x)
        return x


@pytest.mark.parametrize("strategy", ["per_sample", "global", "bucketed"])
def test_bucketed_packed_block_matches_dense_on_cpu(strategy: str) -> None:
    block = _ToyLLaDABlock()
    x = torch.randn(4, 12, 8)
    active = torch.zeros(4, 12, dtype=torch.bool)
    active[0, :11] = True
    active[1, :9] = True
    active[2, :8] = True
    active[3, :5] = True
    expected, _ = block(x)
    packed = BucketedPackedBlock(
        block,
        x,
        active,
        strategy=strategy,
        fused=False,
        width_multiple=2,
        max_padding_fraction=0.25,
    )()
    torch.testing.assert_close(packed, expected, rtol=2e-5, atol=2e-5)


def test_bucketed_runtime_matches_dense_reference_freeze_on_cpu() -> None:
    dense_model = _ToyStack()
    sparse_model = copy.deepcopy(dense_model)
    dense_runtime = BucketedReferenceRuntime(
        dense_model.blocks, batch_size=2, total_len=10
    )
    sparse_runtime = BucketedReferenceRuntime(
        sparse_model.blocks, batch_size=2, total_len=10
    )
    dense_runtime.install_dense()
    sparse_runtime.install_dense()
    first = torch.randn(2, 10, 8)
    dense_model(first.clone())
    sparse_model(first.clone())
    initial_locks = [(0, 2), (0, 7), (1, 3)]
    dense_runtime.commit(initial_locks)
    sparse_runtime.commit(initial_locks)
    second = torch.randn(2, 10, 8)
    dense_model(second.clone())
    sparse_model(second.clone())

    counts = sparse_runtime.active_counts
    plans = plan_active_row_buckets(
        counts,
        total_rows=10,
        width_multiple=2,
        max_padding_fraction=1.0,
    )
    sparse_runtime.activate_compute(plans)
    third = torch.randn(2, 10, 8)
    expected = dense_model(third.clone())
    actual = sparse_model(third.clone())
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)

    later_locks = [(0, 5), (1, 8)]
    dense_runtime.commit(later_locks)
    sparse_runtime.commit(later_locks)
    plans = plan_active_row_buckets(
        sparse_runtime.active_counts,
        total_rows=10,
        width_multiple=2,
        max_padding_fraction=1.0,
    )
    sparse_runtime.set_compute_plan(plans)
    fourth = torch.randn(2, 10, 8)
    expected = dense_model(fourth.clone())
    actual = sparse_model(fourth.clone())
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    dense_runtime.remove()
    sparse_runtime.remove()
