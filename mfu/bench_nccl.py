"""All-gather / reduce-scatter bandwidth at FSDP-like message sizes, in the training container.

Usage: MFU_GPUS=2,3,4,5,6,7 mfu/docker.sh -- torchrun --standalone --nproc-per-node 6 mfu/bench_nccl.py --group-size 6
Prints achieved bus bandwidth per size for the whole world and for sub-groups of --group-size ranks.
Compare runs with/without NCCL_PROTO / NCCL_ALGO overrides. Note: --group-size 3 (two concurrent 3-rank
groups) hung in this container; use a group size equal to the world size.

Measured on 6x H100 (NCCL 2.27.5): 1 GiB bf16 all-gather 204 GB/s busbw, 2 GiB fp32 reduce-scatter 336 GB/s.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist


def bench(fn, iters: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=6)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    groups = {}
    for start in range(0, world, args.group_size):
        ranks = list(range(start, start + args.group_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            groups["sub"] = (g, len(ranks))
    groups["world"] = (dist.group.WORLD, world)
    for name, (g, n) in groups.items():
        for mb in [4, 64, 256, 1024]:
            numel = mb * 2**20 // 2  # bf16 elements of the gathered output
            numel -= numel % n
            out = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
            inp = torch.empty(numel // n, dtype=torch.bfloat16, device="cuda")
            t_ag = bench(lambda: dist.all_gather_into_tensor(out, inp, group=g))
            rs_in = torch.empty(numel, dtype=torch.float32, device="cuda")
            rs_out = torch.empty(numel // n, dtype=torch.float32, device="cuda")
            t_rs = bench(lambda: dist.reduce_scatter_tensor(rs_out, rs_in, group=g))
            busbw_ag = out.numel() * 2 * (n - 1) / n / t_ag / 1e9
            busbw_rs = rs_in.numel() * 4 * (n - 1) / n / t_rs / 1e9
            if rank == 0:
                print(f"{name:5s} n={n} {mb:5d} MB bf16 AG: {t_ag * 1e3:7.2f} ms busbw {busbw_ag:6.1f} GB/s | "
                      f"fp32 RS ({2 * mb} MB): {t_rs * 1e3:7.2f} ms busbw {busbw_rs:6.1f} GB/s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
