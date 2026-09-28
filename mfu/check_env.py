"""Smoke-test the GPU stack inside the MFU-study image: run each native kernel the Nano-V3 recipe uses once.

Usage: mfu/docker.sh -- python mfu/check_env.py
"""

import importlib.metadata as md

import torch


def check(name, fn):
    try:
        out = fn()
        print(f"[ok]   {name}{': ' + out if out else ''}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def te_linear():
    import transformer_engine.pytorch as te

    lin = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lin(x).sum().backward()
    return f"transformer-engine {md.version('transformer-engine')}"


def te_attention():
    import transformer_engine.pytorch as te

    attn = te.DotProductAttention(num_attention_heads=8, kv_channels=128, attn_mask_type="causal", qkv_format="bshd")
    q = torch.randn(2, 256, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attn(q, q, q).sum().backward()
    return ""


def mamba_ssd():
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    b, seqlen, h, p, g, n = 2, 256, 8, 64, 2, 128
    x = torch.randn(b, seqlen, h, p, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dt = torch.rand(b, seqlen, h, device="cuda", dtype=torch.float32)
    A = -torch.rand(h, device="cuda", dtype=torch.float32)
    B = torch.randn(b, seqlen, g, n, device="cuda", dtype=torch.bfloat16)
    C = torch.randn(b, seqlen, g, n, device="cuda", dtype=torch.bfloat16)
    mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=128).sum().backward()
    return f"mamba-ssm {md.version('mamba-ssm')}"


def causal_conv():
    from causal_conv1d import causal_conv1d_fn

    x = torch.randn(2, 512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(512, 4, device="cuda", dtype=torch.bfloat16)
    causal_conv1d_fn(x, w, None, activation="silu").sum().backward()
    return f"causal-conv1d {md.version('causal-conv1d')}"


def grouped_mm():
    offs = torch.tensor([64, 128, 256], device="cuda", dtype=torch.int32)
    a = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(3, 128, 64, device="cuda", dtype=torch.bfloat16)
    torch._grouped_mm(a, b, offs=offs)
    return ""


def deep_ep_import():
    import deep_ep  # noqa: F401

    return "import only (Buffer needs a process group; exercised by the training run)"


if __name__ == "__main__":
    print(f"torch {torch.__version__} cuda {torch.version.cuda} | GPUs: {torch.cuda.device_count()} "
          f"x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    results = [
        check("cuda available", lambda: str(torch.cuda.is_available()) if torch.cuda.is_available() else 1 / 0),
        check("te.Linear fwd/bwd", te_linear),
        check("te.DotProductAttention fwd/bwd", te_attention),
        check("mamba_chunk_scan_combined fwd/bwd", mamba_ssd),
        check("causal_conv1d fwd/bwd", causal_conv),
        check("torch._grouped_mm", grouped_mm),
        check("deep_ep", deep_ep_import),
    ]
    raise SystemExit(0 if all(results) else 1)
