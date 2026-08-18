"""NorMuon ``split_sizes``: each row block must receive the learning-rate
adjustment its own shape would get as a separate parameter, on every path.

The per-block correction is ``adjust(block_shape) / adjust(full_shape)``
(``compute_split_lr_scales``), because the megabatch update derives one
``adjusted_lr`` from the whole fused shape. Applying that factor *before*
NorMuon's normalization does not survive it: the normalization divides U by
sqrt(V) -- and V is an EMA of the scaled U, so the factor cancels -- then
rescales to the Frobenius norm of its input, which reintroduces the factor only
when that rescale is per block. On the FSDP2 sharded path the rescale is
shard-local, so a shard spanning a block boundary blends the blocks' scales
into a single factor and both blocks get the wrong learning rate.

The assertion here is a ratio between two runs that differ *only* in
``adjust_lr``. With ``adjust_lr=None`` the update is ``lr * N``; with an
adjustment it must be ``adjust(block_shape) * N`` for the same N, so per row
block

    ||W0 - W_adjusted|| / ||W0 - W_none||  ==  adjust(lr, block_shape) / lr

exactly. That isolates the learning rate from the shard-local normalization
approximation (which is common to both runs and cancels), so it holds on the
sharded path too, where a full parity-with-separate-parameters test would not.
The grads are supplied, not backpropagated, so the diverging weights of the two
runs never feed back into the updates.

Runs on 1, 2 and 4 GPUs via NCCL. The 48-row matrix with blocks (32, 8, 8)
straddles: at world_size=2 the second shard carries rows of all three blocks,
at world_size=4 two shards straddle a boundary. world_size=1 is a control -- a
mesh dimension of size 1 is not counted as sharded (``_get_shard_info`` requires
``mesh.size(i) > 1``), so it exercises the unsharded per-block normalization
path, which was already correct.

The same identity is asserted for Muon at world_size=2. Muon keeps the scales
inside Newton-Schulz -- correct for Muon, which has nothing after them that
could cancel them -- so the two optimizers now apply the same ``split_scales``
at different points, and the Muon case pins the placement that is safe there.

``TestVarianceRescaleInvariance`` covers the other half of the change -- ``V``
now tracks the unscaled update, so a pre-#113 checkpoint's buffer is off by
``split_scales**2`` per block on resume -- and pins where that cancels and where
it does not.
"""

import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor

from dion.megabatch_base import (
    adjust_lr_rms_norm,
    adjust_lr_spectral_norm,
    compute_split_lr_scales,
    local_split_row_scales,
)
from dion.muon import Muon
from dion.normuon import NorMuon, normuon_normalization_stacked
from dion.polar_express import polar_express


CUDA = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Unequal blocks, and cols small enough that rms_norm's max(fan_out, fan_in)
# does not saturate at fan_in for every block -- otherwise all three scales
# coincide and the test cannot see a blend.
SPLIT_SIZES = (32, 8, 8)
SHAPE = (48, 16)
LR = 0.1
N_STEPS = 3


def _ns(x, epsilon=1e-7):
    return polar_express(x, epsilon=epsilon)


def _adjust_fn(adjust_lr):
    return {
        "spectral_norm": adjust_lr_spectral_norm,
        "rms_norm": adjust_lr_rms_norm,
    }[adjust_lr]


OPTIMIZERS = {"NorMuon": NorMuon, "Muon": Muon}


def _worker(rank, world_size, port, adjust_lr, out_path, opt_name="NorMuon"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    mesh = DeviceMesh("cuda", list(range(world_size)))
    dev = torch.device(f"cuda:{rank}")

    rows, cols = SHAPE
    g = torch.Generator(device=dev).manual_seed(1234)
    W0 = torch.randn(rows, cols, generator=g, device=dev)
    p = torch.nn.Parameter(distribute_tensor(W0, mesh, [Shard(0)]))

    opt = OPTIMIZERS[opt_name](
        [dict(params=[p], split_sizes=SPLIT_SIZES)],
        distributed_mesh=mesh,
        lr=LR,
        weight_decay=0.0,
        adjust_lr=adjust_lr,
        newton_schulz_func=_ns,
    )
    for step in range(N_STEPS):
        gg = torch.Generator(device=dev).manual_seed(100 + step)
        G = torch.randn(rows, cols, generator=gg, device=dev)
        p.grad = distribute_tensor(G, mesh, [Shard(0)])
        opt.step()

    full = p.detach().full_tensor()
    if rank == 0:
        torch.save({"W": full.cpu(), "W0": W0.cpu()}, out_path)
    dist.destroy_process_group()


def _run(world_size, port, adjust_lr, tmp_path, opt_name="NorMuon"):
    out = str(tmp_path / f"out_{port}.pt")
    mp.spawn(
        _worker,
        args=(world_size, port, adjust_lr, out, opt_name),
        nprocs=world_size,
        join=True,
    )
    return torch.load(out)


def measured_block_factors(
    world_size, adjust_lr, tmp_path, port_base, opt_name="NorMuon"
):
    """Per-block ||W0 - W_adjusted|| / ||W0 - W_none||, i.e. the effective
    learning rate each block actually received, in units of ``lr``."""
    adjusted = _run(world_size, port_base, adjust_lr, tmp_path, opt_name)
    baseline = _run(world_size, port_base + 1, None, tmp_path, opt_name)
    d_adj = (adjusted["W0"] - adjusted["W"]).split(list(SPLIT_SIZES), dim=0)
    d_none = (baseline["W0"] - baseline["W"]).split(list(SPLIT_SIZES), dim=0)
    return [(a.norm() / b.norm()).item() for a, b in zip(d_adj, d_none)]


def expected_block_factors(adjust_lr):
    adjust_fn = _adjust_fn(adjust_lr)
    cols = SHAPE[1]
    return [adjust_fn(LR, (rows, cols), flatten=False) / LR for rows in SPLIT_SIZES]


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("adjust_lr", ["spectral_norm", "rms_norm"])
def test_split_blocks_get_their_own_lr_adjustment(world_size, adjust_lr, tmp_path):
    if CUDA < world_size:
        pytest.skip(f"needs >= {world_size} CUDA devices")
    port_base = 29600 + world_size * 10 + (0 if adjust_lr == "spectral_norm" else 4)
    measured = measured_block_factors(world_size, adjust_lr, tmp_path, port_base)
    expected = expected_block_factors(adjust_lr)
    torch.testing.assert_close(
        torch.tensor(measured), torch.tensor(expected), rtol=2e-3, atol=1e-4
    )


@pytest.mark.parametrize("adjust_lr", ["spectral_norm", "rms_norm"])
def test_muon_split_blocks_get_their_own_lr_adjustment(adjust_lr, tmp_path):
    """Muon has no normalization after the scales, so applying them inside
    Newton-Schulz already gave each block its own learning rate on the sharded
    path. Asserted here, with the same ratio identity, so that a future refactor
    cannot move Muon's scales behind a step that cancels them the way NorMuon's
    normalization did -- the two optimizers now apply the same ``split_scales``
    at different points, and only one of the two placements is safe for each.

    world_size=2 only: the placement is what is under test, not the shard
    geometry, which the NorMuon cases above already sweep.
    """
    if CUDA < 2:
        pytest.skip("needs >= 2 CUDA devices")
    port_base = 29680 + (0 if adjust_lr == "spectral_norm" else 4)
    measured = measured_block_factors(2, adjust_lr, tmp_path, port_base, "Muon")
    expected = expected_block_factors(adjust_lr)
    torch.testing.assert_close(
        torch.tensor(measured), torch.tensor(expected), rtol=2e-3, atol=1e-4
    )


class TestLocalSplitRowScales:
    def test_whole_matrix(self):
        assert local_split_row_scales((32, 8, 8), (0.8, 0.5, 0.4), 0, 48) == [
            (0, 32, 0.8),
            (32, 40, 0.5),
            (40, 48, 0.4),
        ]

    def test_shard_straddles_two_boundaries(self):
        # world_size=2 shard 1: global rows [24, 48) covers the tail of block 0
        # and all of blocks 1 and 2, in local coordinates [0, 24).
        assert local_split_row_scales((32, 8, 8), (0.8, 0.5, 0.4), 24, 24) == [
            (0, 8, 0.8),
            (8, 16, 0.5),
            (16, 24, 0.4),
        ]

    def test_shard_inside_one_block(self):
        assert local_split_row_scales((32, 8, 8), (0.8, 0.5, 0.4), 12, 12) == [
            (0, 12, 0.8)
        ]

    def test_unit_scales_skipped(self):
        assert local_split_row_scales((32, 16), (1.0, 0.5), 0, 48) == [(32, 48, 0.5)]

    def test_no_scales(self):
        assert local_split_row_scales((32, 16), None, 0, 48) == []


class TestVarianceRescaleInvariance:
    """The checkpoint-resume claim in the CHANGELOG, executed.

    A pre-#113 checkpoint's ``V`` is the current convention's times
    ``split_scales**2`` per block, because ``V`` used to track the *scaled*
    update. That difference cancels between the division by ``sqrt(V)`` and the
    norm-preserving rescale exactly when the rescale runs per block: within a
    block the factor is constant, so it divides out of ``z / ||z||``. Under FSDP
    the rescale is shard-local, so a shard spanning a block boundary sees a
    non-constant factor and cannot undo it -- the same asymmetry that made the
    old scale placement correct unsharded and wrong sharded.
    """

    # muon_beta2 = 1 freezes the EMA, isolating the division and the rescale
    # from the buffer update. float64 keeps the +1e-8 in the denominator, not
    # rounding, as the only source of residual.
    BETA2 = torch.tensor(1.0, dtype=torch.float64)

    def _setup(self):
        g = torch.Generator().manual_seed(7)
        U = torch.randn(2, *SHAPE, generator=g, dtype=torch.float64)
        V = torch.rand(2, SHAPE[0], 1, generator=g, dtype=torch.float64) + 0.5
        split_scales = compute_split_lr_scales(SPLIT_SIZES, SHAPE, "spectral_norm")
        scales = torch.cat(
            [
                torch.full((n, 1), s, dtype=torch.float64)
                for n, s in zip(SPLIT_SIZES, split_scales)
            ]
        )
        return U, V, scales

    def _normalize(self, U, V, split_sizes):
        out, _ = normuon_normalization_stacked(
            U, V, self.BETA2, split_sizes=split_sizes
        )
        return out

    def test_per_block_rescale_cancels_under_per_block_normalization(self):
        U, V, scales = self._setup()
        base = self._normalize(U, V, SPLIT_SIZES)
        resumed = self._normalize(U, V * scales**2, SPLIT_SIZES)
        torch.testing.assert_close(resumed, base, rtol=1e-6, atol=1e-8)

    def test_uniform_rescale_cancels_under_whole_matrix_normalization(self):
        U, V, _ = self._setup()
        base = self._normalize(U, V, None)
        resumed = self._normalize(U, V * 0.37**2, None)
        torch.testing.assert_close(resumed, base, rtol=1e-6, atol=1e-8)

    def test_per_block_rescale_does_not_cancel_under_shard_local_normalization(self):
        # What a rank whose shard straddles a block boundary actually computes:
        # one rescale over rows of several blocks. The stale V then leaves a
        # per-block error, which is why the resume note is a transient and not
        # an invariance.
        U, V, scales = self._setup()
        base = self._normalize(U, V, None)
        resumed = self._normalize(U, V * scales**2, None)
        ratios = [
            (r.norm() / b.norm()).item()
            for r, b in zip(
                resumed.split(list(SPLIT_SIZES), dim=-2),
                base.split(list(SPLIT_SIZES), dim=-2),
            )
        ]
        assert max(abs(r - 1.0) for r in ratios) > 0.05, ratios
