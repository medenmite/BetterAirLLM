"""Check whether the GPT-OSS MXFP4 Triton kernel dependency is usable."""

from __future__ import annotations

import importlib.metadata
import json
import sys


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    report = {
        "python": sys.executable,
        "packages": {
            "kernels": _package_version("kernels"),
            "triton": _package_version("triton"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "gpt_oss_kernel_available": False,
        "error": None,
    }

    try:
        from transformers.integrations.hub_kernels import get_kernel

        kernel = get_kernel("kernels-community/gpt-oss-triton-kernels")
        required = [
            "matmul_ogs",
            "routing",
            "swiglu",
            "tensor",
            "tensor_details",
            "numerics_details",
        ]
        missing = [name for name in required if not hasattr(kernel, name)]
        if missing:
            raise RuntimeError(f"kernel loaded but missing attribute(s): {missing}")
        report["gpt_oss_kernel_available"] = True
    except BaseException as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2))
    return 0 if report["gpt_oss_kernel_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
