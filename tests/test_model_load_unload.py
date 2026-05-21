import asyncio
import unittest
from types import SimpleNamespace

try:
    import server
    from server_config import ModelEntry, ServerConfig
except ImportError:
    server = None


class ModelLoadUnloadEndpointTests(unittest.TestCase):
    def setUp(self):
        if server is not None:
            server.model_manager = None
            server.loaded_model = None
            server.loaded_model_id = None
            server.config = ServerConfig(
                device="cpu",
                models=[
                    ModelEntry(id="hf-model-1", repo_id="org/hf-model-1", backend="betterairllm"),
                    ModelEntry(id="hf-model-2", repo_id="org/hf-model-2", backend="betterairllm"),
                    ModelEntry(
                        id="ollama-model",
                        repo_id="ollama-model",
                        backend="ollama",
                        source="ollama",
                        format="gguf",
                        ollama_model="ollama-model",
                    ),
                ],
                discover_ollama=False,
                max_loaded_models=1,
            )

    @unittest.skipIf(server is None, "server module not available")
    def test_load_and_unload_hf_model(self):
        original_auto_model = server.AutoModel
        loaded_repos = []

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(repo_id, **kwargs):
                loaded_repos.append(repo_id)
                return SimpleNamespace(repo_id=repo_id)

        server.AutoModel = FakeAutoModel
        try:

            load_response = asyncio.run(server.load_model_endpoint("hf-model-1"))
            self.assertEqual(load_response["status"], "ok")
            self.assertEqual(load_response["model_id"], "hf-model-1")
            self.assertEqual(loaded_repos, ["org/hf-model-1"])
            self.assertEqual(server.loaded_model_id, "hf-model-1")


            unload_response = asyncio.run(server.unload_model_endpoint("hf-model-1"))
            self.assertEqual(unload_response["status"], "ok")
            self.assertIsNone(server.loaded_model_id)


            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(server.unload_model_endpoint("hf-model-1"))
            self.assertEqual(ctx.exception.status_code, 404)

        finally:
            server.AutoModel = original_auto_model

    @unittest.skipIf(server is None, "server module not available")
    def test_load_ollama_model_returns_noop_message(self):

        load_response = asyncio.run(server.load_model_endpoint("ollama-model"))
        self.assertEqual(load_response["status"], "ok")
        self.assertIn("managed by the Ollama daemon", load_response["message"])

    @unittest.skipIf(server is None, "server module not available")
    def test_load_missing_model_raises_404(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(server.load_model_endpoint("non-existent-model"))
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
