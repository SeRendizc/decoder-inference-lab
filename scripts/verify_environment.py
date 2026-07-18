from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def run_command(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def verify_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 无法访问 CUDA")

    device = torch.device("cuda")
    torch.manual_seed(20260718)
    torch.cuda.manual_seed_all(20260718)

    x = torch.randn(256, 256, device=device, requires_grad=True)
    started_at = time.perf_counter()
    loss = (x @ x.T).square().mean()
    loss.backward()
    synchronize()
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    if x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA backward 没有产生有限梯度")

    return {
        "forward_backward_ok": True,
        "loss": loss.item(),
        "elapsed_ms": elapsed_ms,
    }


def verify_sdpa() -> dict[str, Any]:
    device = torch.device("cuda")
    query = torch.randn(2, 4, 64, 64, device=device, requires_grad=True)
    key = torch.randn(2, 4, 64, 64, device=device, requires_grad=True)
    value = torch.randn(2, 4, 64, 64, device=device, requires_grad=True)

    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
    )
    output.square().mean().backward()
    synchronize()

    if query.grad is None or not torch.isfinite(query.grad).all():
        raise RuntimeError("SDPA backward 没有产生有限梯度")

    return {
        "forward_backward_ok": True,
        "output_shape": list(output.shape),
        "dtype": str(output.dtype),
    }


def verify_compile() -> dict[str, Any]:
    device = torch.device("cuda")
    weight = torch.randn(128, 128, device=device)

    def eager(value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value @ weight)

    compiled = torch.compile(eager)
    value = torch.randn(32, 128, device=device)
    expected = eager(value)
    actual = compiled(value)
    synchronize()

    max_abs_error = (expected - actual).abs().max().item()
    if not torch.allclose(expected, actual, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"torch.compile 结果不一致：max_abs_error={max_abs_error}")

    return {
        "correctness_ok": True,
        "max_abs_error": max_abs_error,
    }


def collect_environment(check_compile: bool) -> dict[str, Any]:
    cuda_result = verify_cuda()
    sdpa_result = verify_sdpa()
    compile_result = verify_compile() if check_compile else {"skipped": True}
    properties = torch.cuda.get_device_properties(0)

    return {
        "status": "pass",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "pytorch": {
            "version": torch.__version__,
            "module_path": torch.__file__,
            "built_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "supported_arches": torch.cuda.get_arch_list(),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "driver_version": run_command(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
        },
        "checks": {
            "cuda_forward_backward": cuda_result,
            "sdpa": sdpa_result,
            "torch_compile": compile_result,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-compile", action="store_true")
    args = parser.parse_args()

    result = collect_environment(check_compile=args.check_compile)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()
