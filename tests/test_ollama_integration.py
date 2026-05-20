import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from queue import Queue

import ollama_registry
from ollama_registry import (
    clear_ollama_discovery_cache,
    discover_ollama_models,
    get_ollama_discovery_cache_status,
    proxy_ollama_chat_completion,
    stream_ollama_chat_completion,
)
from server_config import ModelEntry, ServerConfig, load_config, load_default_models

try:
    import server
except ModuleNotFoundError as exc:
    server = None
    SERVER_IMPORT_ERROR = exc
else:
    SERVER_IMPORT_ERROR = None


class OllamaIntegrationTests(unittest.TestCase):
    def setUp(self):
        clear_ollama_discovery_cache()
        if server is not None:
            server.model_manager = None
            server.loaded_model = None
            server.loaded_model_id = None
            server.last_generation_stats.clear()

    def test_model_entry_defaults_preserve_airllm_backend(self):
        entry = ModelEntry(id="mistral", repo_id="mistralai/Mistral-7B-Instruct-v0.1")

        self.assertEqual(entry.source, "hf")
        self.assertEqual(entry.backend, "airllm")
        self.assertIsNone(entry.format)
        self.assertIsNone(entry.ollama_model)

    def test_load_config_accepts_new_ollama_entry(self):
        old_models = os.environ.get("AIRLLM_MODELS")
        os.environ["AIRLLM_MODELS"] = json.dumps(
            [
                {
                    "id": "ollama/qwen3:8b",
                    "repo_id": "qwen3:8b",
                    "source": "ollama",
                    "backend": "ollama",
                    "format": "gguf",
                    "ollama_model": "qwen3:8b",
                    "metadata": {"family": "qwen3"},
                }
            ]
        )
        try:
            config = load_config()
        finally:
            if old_models is None:
                os.environ.pop("AIRLLM_MODELS", None)
            else:
                os.environ["AIRLLM_MODELS"] = old_models

        self.assertEqual(config.models[0].id, "ollama/qwen3:8b")
        self.assertEqual(config.models[0].backend, "ollama")
        self.assertEqual(config.models[0].format, "gguf")
        self.assertEqual(config.models[0].metadata["family"], "qwen3")

    def test_default_json_registry_loads_current_defaults(self):
        old_models = os.environ.get("AIRLLM_MODELS")
        old_model = os.environ.get("AIRLLM_MODEL")
        os.environ.pop("AIRLLM_MODELS", None)
        os.environ.pop("AIRLLM_MODEL", None)
        try:
            config = load_config()
        finally:
            if old_models is not None:
                os.environ["AIRLLM_MODELS"] = old_models
            if old_model is not None:
                os.environ["AIRLLM_MODEL"] = old_model

        self.assertEqual(
            [model.id for model in config.models],
            ["qwen3.6-35b-a3b", "qwen3-30b-a3b", "mistral-7b-instruct", "llama2-7b-chat"],
        )
        self.assertEqual(config.models[0].repo_id, "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(config.models[0].family, "qwen3_5_moe")
        self.assertEqual(config.models[0].metadata["family"], "qwen3_5_moe")

    def test_airllm_model_prepends_custom_model(self):
        old_model = os.environ.get("AIRLLM_MODEL")
        old_model_id = os.environ.get("AIRLLM_MODEL_ID")
        old_models = os.environ.get("AIRLLM_MODELS")
        os.environ.pop("AIRLLM_MODELS", None)
        os.environ["AIRLLM_MODEL"] = "local/custom-model"
        os.environ["AIRLLM_MODEL_ID"] = "custom"
        try:
            config = load_config()
        finally:
            if old_model is None:
                os.environ.pop("AIRLLM_MODEL", None)
            else:
                os.environ["AIRLLM_MODEL"] = old_model
            if old_model_id is None:
                os.environ.pop("AIRLLM_MODEL_ID", None)
            else:
                os.environ["AIRLLM_MODEL_ID"] = old_model_id
            if old_models is not None:
                os.environ["AIRLLM_MODELS"] = old_models

        self.assertEqual(config.models[0].id, "custom")
        self.assertEqual(config.models[0].repo_id, "local/custom-model")
        self.assertEqual(config.models[1].id, "qwen3.6-35b-a3b")

    def test_malformed_registry_raises_clear_error(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write('{"not": "a-list"}')
            path = handle.name
        try:
            with self.assertRaisesRegex(TypeError, "model registry must be a JSON array"):
                load_default_models(path)
        finally:
            os.unlink(path)

    def test_discover_ollama_models_from_tags_and_show(self):
        original_request_json = ollama_registry._request_json

        def fake_request_json(method, url, payload, timeout_seconds):
            if url.endswith("/api/tags"):
                return {
                        "models": [
                            {
                                "name": "qwen3:8b",
                                "model": "qwen3:8b",
                                "details": {
                                    "format": "gguf",
                                    "family": "qwen3",
                                    "parameter_size": "8B",
                                    "quantization_level": "Q4_K_M",
                                },
                            }
                        ]
                    }
            if url.endswith("/api/show"):
                return {
                    "details": {"format": "gguf", "family": "qwen3"},
                    "model_info": {"qwen3.context_length": 32768},
                }
            self.fail(f"unexpected URL: {url}")

        ollama_registry._request_json = fake_request_json
        try:
            entries = asyncio.run(discover_ollama_models("http://ollama.test"))
        finally:
            ollama_registry._request_json = original_request_json

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].id, "ollama/qwen3:8b")
        self.assertEqual(entries[0].backend, "ollama")
        self.assertEqual(entries[0].format, "gguf")
        self.assertEqual(entries[0].max_seq_len, 32768)
        self.assertEqual(entries[0].metadata["family"], "qwen3")
        self.assertEqual(entries[0].metadata["parameter_size"], "8B")
        self.assertEqual(entries[0].metadata["context_length"], 32768)
        self.assertEqual(entries[0].metadata["model_info"]["qwen3.context_length"], 32768)

    def test_discovery_cache_returns_cached_models_within_ttl(self):
        original_request_json = ollama_registry._request_json
        calls = {"tags": 0}

        def fake_request_json(method, url, payload, timeout_seconds):
            if url.endswith("/api/tags"):
                calls["tags"] += 1
                return {"models": [{"name": f"model-{calls['tags']}", "details": {"format": "gguf"}}]}
            if url.endswith("/api/show"):
                return {"details": {"format": "gguf"}}
            self.fail(f"unexpected URL: {url}")

        ollama_registry._request_json = fake_request_json
        try:
            first = asyncio.run(discover_ollama_models("http://ollama.test", ttl_seconds=10))
            second = asyncio.run(discover_ollama_models("http://ollama.test", ttl_seconds=10))
        finally:
            ollama_registry._request_json = original_request_json

        self.assertEqual(calls["tags"], 1)
        self.assertEqual(first[0].ollama_model, "model-1")
        self.assertEqual(second[0].ollama_model, "model-1")

    def test_discovery_cache_refreshes_after_ttl(self):
        original_request_json = ollama_registry._request_json
        calls = {"tags": 0}

        def fake_request_json(method, url, payload, timeout_seconds):
            if url.endswith("/api/tags"):
                calls["tags"] += 1
                return {"models": [{"name": f"model-{calls['tags']}", "details": {"format": "gguf"}}]}
            if url.endswith("/api/show"):
                return {"details": {"format": "gguf"}}
            self.fail(f"unexpected URL: {url}")

        ollama_registry._request_json = fake_request_json
        try:
            first = asyncio.run(discover_ollama_models("http://ollama.test", ttl_seconds=0.001))
            asyncio.run(asyncio.sleep(0.01))
            second = asyncio.run(discover_ollama_models("http://ollama.test", ttl_seconds=0.001))
        finally:
            ollama_registry._request_json = original_request_json

        self.assertEqual(calls["tags"], 2)
        self.assertEqual(first[0].ollama_model, "model-1")
        self.assertEqual(second[0].ollama_model, "model-2")

    def test_discovery_uses_stale_cache_on_refresh_failure(self):
        original_request_json = ollama_registry._request_json
        fail = {"enabled": False}

        def fake_request_json(method, url, payload, timeout_seconds):
            if fail["enabled"]:
                raise OSError("daemon down")
            if url.endswith("/api/tags"):
                return {"models": [{"name": "cached-model", "details": {"format": "gguf"}}]}
            if url.endswith("/api/show"):
                return {"details": {"format": "gguf"}}
            self.fail(f"unexpected URL: {url}")

        ollama_registry._request_json = fake_request_json
        try:
            asyncio.run(discover_ollama_models("http://ollama.test", ttl_seconds=10))
            fail["enabled"] = True
            stale = asyncio.run(discover_ollama_models("http://ollama.test", ttl_seconds=10, force_refresh=True))
            status = get_ollama_discovery_cache_status("http://ollama.test")
        finally:
            ollama_registry._request_json = original_request_json

        self.assertEqual(stale[0].ollama_model, "cached-model")
        self.assertEqual(status.discovered_model_count, 1)
        self.assertTrue(status.last_refresh_used_stale)
        self.assertIn("daemon down", status.last_discovery_error)

    def test_proxy_ollama_chat_completion_non_streaming_payload(self):
        original_request_json = ollama_registry._request_json
        seen_payload = {}

        def fake_request_json(method, url, payload, timeout_seconds):
            seen_payload.update(payload)
            return {
                "id": "chatcmpl-upstream",
                "object": "chat.completion",
                "model": "qwen3:8b",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }

        ollama_registry._request_json = fake_request_json
        try:
            result = asyncio.run(
                proxy_ollama_chat_completion(
                    "http://ollama.test",
                    {
                        "model": "qwen3:8b",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 4,
                        "stream": False,
                    },
                )
            )
        finally:
            ollama_registry._request_json = original_request_json

        self.assertEqual(seen_payload["model"], "qwen3:8b")
        self.assertEqual(seen_payload["max_tokens"], 4)
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")

    def test_stream_ollama_chat_completion_forwards_chunks(self):
        original_urlopen = ollama_registry.urlrequest.urlopen

        class FakeResponse:
            def __init__(self):
                self.chunks = [b"data: one\n\n", b"data: two\n\n", b""]

            def read(self, size):
                return self.chunks.pop(0)

            def close(self):
                pass

        def fake_urlopen(request, timeout):
            return FakeResponse()

        async def collect():
            chunks = []
            async for chunk in stream_ollama_chat_completion(
                "http://ollama.test",
                {"model": "qwen3:8b", "messages": [], "stream": True},
            ):
                chunks.append(chunk)
            return chunks

        ollama_registry.urlrequest.urlopen = fake_urlopen
        try:
            chunks = asyncio.run(collect())
        finally:
            ollama_registry.urlrequest.urlopen = original_urlopen

        self.assertEqual(chunks, [b"data: one\n\n", b"data: two\n\n"])

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_list_models_includes_discovered_ollama(self):
        original_discover = server.discover_ollama_models

        async def fake_discover(*args, **kwargs):
            return [
                ModelEntry(
                    id="ollama/qwen3:8b",
                    repo_id="qwen3:8b",
                    owned_by="ollama",
                    source="ollama",
                    backend="ollama",
                    format="gguf",
                    ollama_model="qwen3:8b",
                )
            ]

        server.discover_ollama_models = fake_discover
        try:
            server.config = ServerConfig(
                models=[ModelEntry(id="mistral", repo_id="mistralai/Mistral-7B-Instruct-v0.1")]
            )
            result = asyncio.run(server.list_models())
        finally:
            server.discover_ollama_models = original_discover

        model_ids = {item["id"] for item in result["data"]}
        self.assertIn("mistral", model_ids)
        self.assertIn("ollama/qwen3:8b", model_ids)
        ollama_item = next(item for item in result["data"] if item["id"] == "ollama/qwen3:8b")
        self.assertEqual(ollama_item["backend"], "ollama")
        self.assertEqual(ollama_item["format"], "gguf")

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_capabilities_endpoint_shape(self):
        server.config = ServerConfig(
            models=[ModelEntry(id="mistral", repo_id="mistralai/Mistral-7B-Instruct-v0.1")],
            discover_ollama=False,
        )

        result = asyncio.run(server.capabilities())

        backend_ids = {item["id"] for item in result["backends"]}
        family_ids = {item["id"] for item in result["hf_architecture_families"]}
        self.assertIn("airllm", backend_ids)
        self.assertIn("ollama", backend_ids)
        self.assertIn("qwen2_qwen2_5", family_ids)
        self.assertIn("runtime", result)
        self.assertIn("hardware", result)
        self.assertIn("gpt_oss_mxfp4_reference", {item["id"] for item in result["experimental_features"]})
        self.assertTrue(any("GGUF" in item for item in result["known_limitations"]))

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_preflight_for_airllm_model_reports_registry_and_warnings(self):
        server.config = ServerConfig(
            device="cpu",
            models=[ModelEntry(id="llama", repo_id="meta-llama/Llama-2-7b-chat-hf")],
            discover_ollama=False,
        )

        result = asyncio.run(server.model_preflight("llama"))

        self.assertEqual(result["model_id"], "llama")
        self.assertEqual(result["backend"], "airllm")
        self.assertEqual(result["registry_entry"]["repo_id"], "meta-llama/Llama-2-7b-chat-hf")
        self.assertIn("support", result)
        self.assertIn("paths", result)
        self.assertIn("may require HF_TOKEN", " ".join(result["warnings"]))

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_preflight_for_airllm_gguf_blocks(self):
        server.config = ServerConfig(
            models=[ModelEntry(id="bad-gguf", repo_id="C:/models/model.gguf", backend="airllm", format="gguf")],
            discover_ollama=False,
        )

        result = asyncio.run(server.model_preflight("bad-gguf"))

        self.assertEqual(result["status"], "blocked")
        self.assertIn("GGUF cannot run", result["blockers"][0])

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_preflight_for_configured_ollama_model(self):
        server.config = ServerConfig(
            models=[
                ModelEntry(
                    id="ollama/qwen3:8b",
                    repo_id="qwen3:8b",
                    source="ollama",
                    backend="ollama",
                    format="gguf",
                    ollama_model="qwen3:8b",
                    metadata={"family": "qwen3", "context_length": 32768},
                )
            ],
            discover_ollama=False,
        )

        result = asyncio.run(server.model_preflight("ollama/qwen3:8b"))

        self.assertEqual(result["backend"], "ollama")
        self.assertEqual(result["ollama"]["raw_model"], "qwen3:8b")
        self.assertEqual(result["ollama"]["context_length"], 32768)

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_preflight_for_missing_model_returns_404_payload(self):
        server.config = ServerConfig(models=[], discover_ollama=False)

        result = asyncio.run(server.model_preflight("missing-model"))

        self.assertEqual(result.status_code, 404)
        self.assertIn("missing-model", result.body.decode("utf-8"))

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_ollama_chat_endpoint_uses_proxy_payload(self):
        original_proxy = server.proxy_ollama_chat_completion
        seen_payload = {}

        async def fake_proxy(base_url, payload, **kwargs):
            seen_payload.update(payload)
            return {
                "id": "chatcmpl-upstream",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [{"message": {"role": "assistant", "content": "proxied"}}],
            }

        server.proxy_ollama_chat_completion = fake_proxy
        try:
            server.config = ServerConfig(
                models=[
                    ModelEntry(
                        id="ollama/qwen3:8b",
                        repo_id="qwen3:8b",
                        source="ollama",
                        backend="ollama",
                        format="gguf",
                        ollama_model="qwen3:8b",
                    )
                ],
                discover_ollama=False,
            )
            result = asyncio.run(
                server.chat_completions(
                    server.ChatCompletionRequest(
                        model="ollama/qwen3:8b",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=8,
                        stream=False,
                        tools=[{"type": "function", "function": {"name": "lookup"}}],
                        reasoning_effort="low",
                    )
                )
            )
        finally:
            server.proxy_ollama_chat_completion = original_proxy

        self.assertEqual(seen_payload["model"], "qwen3:8b")
        self.assertEqual(seen_payload["max_tokens"], 8)
        self.assertEqual(seen_payload["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(seen_payload["reasoning_effort"], "low")
        self.assertEqual(result["choices"][0]["message"]["content"], "proxied")

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_airllm_backend_rejects_gguf_config(self):
        server.config = ServerConfig(
            models=[
                ModelEntry(
                    id="bad-gguf",
                    repo_id="C:/models/model.gguf",
                    backend="airllm",
                    format="gguf",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "GGUF cannot currently run through BetterAirLLM"):
            server.get_or_load_model("bad-gguf")

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_missing_ollama_model_gets_targeted_detail(self):
        server.config = ServerConfig(
            models=[
                ModelEntry(
                    id="ollama/qwen3:8b",
                    repo_id="qwen3:8b",
                    source="ollama",
                    backend="ollama",
                    format="gguf",
                    ollama_model="qwen3:8b",
                )
            ],
            discover_ollama=False,
        )

        detail = asyncio.run(server._ollama_model_not_found_detail("ollama/not-installed:latest"))

        self.assertIn("Ollama model 'not-installed:latest' is not installed or not discoverable", detail)
        self.assertIn("qwen3:8b", detail)

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_runtime_endpoint_reports_manager_and_generation_stats(self):
        server.config = ServerConfig(models=[], discover_ollama=False, device="cpu")
        server.last_generation_stats.update({"model_id": "mistral", "tokens_per_second": 12.5})

        result = asyncio.run(server.runtime_status())

        self.assertEqual(result["last_generation"]["model_id"], "mistral")
        self.assertIn("model_manager", result)
        self.assertEqual(result["config"]["stream_mode"], "auto")
        self.assertIn("hardware", result)

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_model_manager_reuses_and_evicts_lru(self):
        original_auto_model = server.AutoModel
        loaded = []

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(repo_id, **kwargs):
                loaded.append(repo_id)
                return SimpleNamespace(repo_id=repo_id, kwargs=kwargs)

        server.AutoModel = FakeAutoModel
        try:
            server.config = ServerConfig(
                models=[
                    ModelEntry(id="one", repo_id="repo/one"),
                    ModelEntry(id="two", repo_id="repo/two"),
                ],
                discover_ollama=False,
                device="cpu",
                max_loaded_models=1,
            )
            first, _ = server.get_or_load_model("one")
            reused, _ = server.get_or_load_model("one")
            second, _ = server.get_or_load_model("two")
            payload = server._get_model_manager().runtime_payload()
        finally:
            server.AutoModel = original_auto_model

        self.assertIs(first, reused)
        self.assertEqual(second.repo_id, "repo/two")
        self.assertEqual(loaded, ["repo/one", "repo/two"])
        self.assertEqual(payload["reuses"], 1)
        self.assertEqual(payload["evictions"], 1)
        self.assertEqual(payload["loaded_model_ids"], ["two"])

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_gpt_oss_kwargs_prefer_hf_triton_when_available(self):
        original_status = server._hf_triton_status
        server._hf_triton_status = lambda: {"available": True, "reason": "available", "cuda_available": True}
        try:
            server.config = ServerConfig(
                models=[],
                discover_ollama=False,
                device="cpu",
                mxfp4_execution="hf_triton",
                hf_triton_module_cache_mb=128,
            )
            kwargs = server._build_model_load_kwargs(ModelEntry(id="gpt-oss", repo_id="openai/gpt-oss-20b"))
        finally:
            server._hf_triton_status = original_status

        self.assertEqual(kwargs["mxfp4_execution"], "hf_triton")
        self.assertEqual(kwargs["hf_triton_module_cache_mb"], 128)

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_stream_response_emits_real_sse_chunks(self):
        original_streamer = server.TextIteratorStreamer

        class FakeStreamer:
            def __init__(self, *args, **kwargs):
                self.queue = Queue()

            def __iter__(self):
                while True:
                    item = self.queue.get(timeout=5)
                    if item is None:
                        return
                    yield item

            def on_finalized_text(self, text, stream_end=False):
                if text:
                    self.queue.put(text)
                if stream_end:
                    self.queue.put(None)

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                if kwargs.get("return_tensors") == "pt":
                    import torch

                    return {"input_ids": torch.tensor([[1, 2]])}
                return {"input_ids": [1]}

        class FakeModel:
            tokenizer = FakeTokenizer()

            def reset_runtime_stats(self):
                pass

            def runtime_stats(self):
                return {"cpu_load_bytes": 0}

            def generate(self, input_ids, **kwargs):
                streamer = kwargs["streamer"]
                streamer.on_finalized_text("hello", stream_end=False)
                streamer.on_finalized_text(" world", stream_end=True)

        async def collect():
            return [
                item
                async for item in server.stream_response(
                    FakeModel(),
                    ModelEntry(id="fake", repo_id="fake"),
                    "prompt",
                    server.ChatCompletionRequest(model="fake", messages=[{"role": "user", "content": "hi"}], stream=True),
                    "chatcmpl-test",
                    123,
                )
            ]

        server.TextIteratorStreamer = FakeStreamer
        try:
            server.config = ServerConfig(models=[ModelEntry(id="fake", repo_id="fake")], discover_ollama=False, device="cpu")
            chunks = asyncio.run(collect())
        finally:
            server.TextIteratorStreamer = original_streamer

        joined = "".join(chunks)
        self.assertIn('"content": "hello"', joined)
        self.assertIn('"content": " world"', joined)
        self.assertTrue(chunks[-1].strip().endswith("[DONE]"))

    @unittest.skipIf(server is None, f"server dependencies unavailable: {SERVER_IMPORT_ERROR}")
    def test_stream_response_falls_back_to_compat_before_first_token(self):
        original_stream = server.run_inference_stream
        original_inference = server.run_inference

        def fake_stream(*args, **kwargs):
            queue = Queue()
            queue.put(("error", RuntimeError("stream unavailable")))
            return queue

        server.run_inference_stream = fake_stream
        server.run_inference = lambda *args, **kwargs: "fallback ok"
        try:
            server.config = ServerConfig(models=[ModelEntry(id="fake", repo_id="fake")], discover_ollama=False, device="cpu")
            request = server.ChatCompletionRequest(model="fake", messages=[{"role": "user", "content": "hi"}], stream=True)

            async def collect():
                return [
                    item
                    async for item in server.stream_response(SimpleNamespace(), ModelEntry(id="fake", repo_id="fake"), "prompt", request, "chatcmpl-test", 123)
                ]

            chunks = asyncio.run(collect())
        finally:
            server.run_inference_stream = original_stream
            server.run_inference = original_inference

        joined = "".join(chunks)
        self.assertIn('"content": "fallback"', joined)
        self.assertIn('"content": " ok"', joined)
        self.assertTrue(chunks[-1].strip().endswith("[DONE]"))


if __name__ == "__main__":
    unittest.main()
