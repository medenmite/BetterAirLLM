"""Benchmark the BetterAirLLM OpenAI-compatible chat endpoint.

This script measures non-streaming latency and streaming time-to-first-token
without importing the local model runtime. It is intentionally dependency-light
and uses the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def _post_json(url: str, payload: dict, timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-airllm"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def run_non_streaming(args) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    with _post_json(f"{args.base_url.rstrip('/')}/chat/completions", payload, args.timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    return {
        "mode": "non_streaming",
        "status": response.status,
        "elapsed_seconds": elapsed,
        "response_chars": len(content),
        "usage": usage,
        "tokens_per_second": completion_tokens / elapsed if isinstance(completion_tokens, int) and completion_tokens > 0 and elapsed > 0 else None,
    }


def run_streaming(args) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
    }
    started = time.perf_counter()
    first_token_at = None
    chunks = 0
    chars = 0
    with _post_json(f"{args.base_url.rstrip('/')}/chat/completions", payload, args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or ""
            if content and first_token_at is None:
                first_token_at = time.perf_counter()
            if content:
                chunks += 1
                chars += len(content)
    elapsed = time.perf_counter() - started
    return {
        "mode": "streaming",
        "status": response.status,
        "elapsed_seconds": elapsed,
        "time_to_first_token_seconds": (first_token_at - started) if first_token_at else None,
        "chunks": chunks,
        "response_chars": chars,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain BetterAirLLM in one short paragraph.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--mode", choices=("both", "streaming", "non_streaming"), default="both")
    args = parser.parse_args()

    reports = []
    try:
        if args.mode in {"both", "non_streaming"}:
            reports.append(run_non_streaming(args))
        if args.mode in {"both", "streaming"}:
            reports.append(run_streaming(args))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc

    print(json.dumps({"base_url": args.base_url, "model": args.model, "reports": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
