# Course-lecture techniques mapped to this workload

Condensed from `lectures/lec001..lec008` (course repo). Used to pick interventions; ★ = most relevant to the
measured Nemotron-Nano-V3 profile (packed config: DeepEP ~22% on the compute stream, FSDP AG/RS ~16% with <4%
overlap, ~18% GPU idle from host overhead, GEMMs ~25%, optimizer ~6%, ~8k tokens/GPU per micro-batch).

| # | technique | lecture | key heuristic / number from the lecture | status here |
|---|---|---|---|---|
| 1 ★ | Overlap MoE dispatch/combine with compute (async dispatch, dedicated comm SMs, shared-expert side stream, MoK megakernel) | L4, L5, L7 | all-to-all ~ 2·B·D·b·k per layer, harder to hide than all-reduce; MoK +13.6% BF16 vs HybridEP; shared-expert side stream ~neutral for Kimi K3 | testing `dispatcher_async_dispatch` |
| 2 ★ | FSDP prefetch / wrap granularity | L5, Playbook | all-gather unit l+1 during compute of l; hidden only if T_AG ≲ T_compute; too-small wraps pay α per message | implemented explicit MoE prefetch (`enable_fsdp2_prefetch`) |
| 3 ★ | Kill launch overhead: CUDA graphs, torch.compile, fusion, remove host syncs, Python GC | L2, L3, L7 | RMSNorm @8k tokens 83.5 µs wall vs 23.8 µs kernel; "CUDA-graph the launch path first"; compile islands +11.7%; GC every 10 steps +8.4%; bounded-capacity dispatch (no host sync) +8.5% | GC done (−9% mean step); graphs/compile next |
| 4 ★ | More tokens per GPU per step | L7, L2/L3 | local batch 2: +32.5% (MFU 8.3%→12.3%) in the L7 table | done: 2 packs/GPU +22% |
| 5 ★ | MoE load balance / expert GEMM efficiency | L4, L5, L6 | slowest expert sets step time; utilization ≈ mean/max tokens per rank; bias controller γ≈0.001 | to measure (real vs fake-balanced routing) |
| 6 | FP8 GEMMs (TE recipes) | L8, L7 | H100 FP8 peak 1,979 vs 989 TF; ceiling limited by GEMM share (~25% here) | later |
| 7 | Low-precision gradient reduction | L7, L8 | BF16 grad reduce halves bytes (+1.0%); risk: Nemotron-3 Ultra diverged with BF16 output-layer grad accumulation | candidate (experts' fp32 RS is large on 6 GPUs) |
| 8 | Fuse glue ops (residual, norms, act+loss) | L2, L3, L7 | glue is bandwidth-bound (AI ~1/6); fused CE avoids logits round-trips | fused linear-CE done |
| 9 | α–β collective model | L5 | NVLink α ≲ 1 µs, β ≈ 1.1 ps/B, crossover ≈ 0.9 MB; batch small messages | explains RING_LL small FSDP messages |
| 10 | Keep hot collectives intra-node | L5, L7 | EP/TP/CP inside the NVLink domain | single node already |
| 11 | Selective activation checkpointing | L5, L8, L6 | selective ≈ 70% activation savings for ~2.7% compute; recompute norms/activations, not GEMMs | candidate to enable bigger micro-batches |
| 12 | Memory planning | L5, L7 | ~16 B/param mixed-precision AdamW; here all-bf16 ≈ 8 B/param = 253 GB total | used to size 6-GPU runs |
| 13-16 | TP/SP, PP schedules, CP, FlashAttention | L5, L6 | only needed at larger scale / long context | not needed (3.2B active, short sequences) |
| 17 | Sequence packing | L8 | no padding waste + fixed shapes (also enables CUDA graphs) | done: 8.2× |
| 18 | Grad accumulation / batch size | L5, L8 | cuts sync frequency; stay below critical batch | used |
| 19 | MFU / goodput accounting | L3, L8 | MFU = 6·N_active·tokens/(T·G·peak), N_active ≈ 3.2B for Nano | used (useful MFU) |
| 20 | Profiling discipline | L3, L7 | convert kernel time to achieved FLOP/s & B/s; one change at a time vs noise floor | followed |
