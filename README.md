<p align="center">
  <img src="assets/BetterAirLLM_new.png" alt="BetterAirLLM" width="420">
</p>

[**Quickstart**](#quickstart) |
[**Server API**](#openai-compatible-server) |
[**Desktop UI**](#desktop-ui) |
[**Model support**](#model-support-matrix) |
[**MoE status**](#betterairllm-moe-status) |
[**Ollama**](#using-ollama-models-through-betterairllm) |
[**FAQ**](#faq)

**BetterAirLLM** is a local large-model inference stack built on AirLLM-style layer streaming. This repo now includes an OpenAI-compatible FastAPI server, a model registry, Ollama GGUF proxy support, experimental MoE selective expert paths, GPT-OSS/Qwen smoke and benchmark tools, and a Next.js desktop chat shell.

The core memory goal is unchanged: keep only the active pieces of a model resident while streaming weights from disk, so larger Hugging Face checkpoints can run on smaller GPUs than full-model loading would normally allow. The new server path exposes that runtime through `/v1/models`, `/v1/chat/completions`, `/v1/capabilities`, `/v1/runtime`, and per-model preflight checks.

## What's New In This Repo

* OpenAI-compatible backend in `server.py` with streaming and non-streaming chat completions.
* Multi-model registry in `model_registry.json`, with `AIRLLM_MODELS` and `AIRLLM_MODEL` overrides.
* Dynamic Ollama discovery and proxying, exposing local GGUF models as `ollama/<model-name>`.
* Experimental `BetterAirLLMMoE` runtime with strict selective expert adapters for verified layouts.
* Qwen3.5/Qwen3.6 MoE routing support and GPT-OSS MXFP4 staged reference/direct-slice tooling.
* Runtime/preflight diagnostics for CUDA, registry metadata, Ollama health, split paths, and feature gates.
* Next.js desktop chat shell in `desktop/`.
* Tests for architecture dispatch, registry loading, Ollama discovery/proxy behavior, streaming, and MoE adapters.

## Quickstart

### 1. Install dependencies

Create an environment, install the server dependencies, and keep the local `air_llm` package on the Python path by running from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-server.txt
```

For development and tests, also install:

```powershell
pip install -r requirements-dev.txt
```

### 2. Start the OpenAI-compatible server

```powershell
python server.py
```

The server listens on `http://localhost:8000` by default and loads models lazily. Configure a one-off Hugging Face or local checkpoint with:

```powershell
$env:AIRLLM_MODEL = "Qwen/Qwen3-30B-A3B"
$env:AIRLLM_MODEL_ID = "qwen3-30b-a3b"
python server.py
```

### 3. Inspect readiness

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:8000/v1/models | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:8000/v1/capabilities | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:8000/v1/runtime | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:8000/v1/models/qwen3.6-35b-a3b/preflight | ConvertTo-Json -Depth 8
```

### 4. Send a chat request

```powershell
$body = @{
  model = "mistral-7b-instruct"
  messages = @(@{ role = "user"; content = "Explain BetterAirLLM in one short paragraph." })
  max_tokens = 64
  stream = $false
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri http://localhost:8000/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```

## OpenAI-Compatible Server

The server accepts the common OpenAI chat-completions request shape and supports both BetterAirLLM-backed Hugging Face checkpoints and proxied Ollama models.

Important endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Server, CUDA, runtime, Ollama, and cache health. |
| `GET /v1/models` | OpenAI-style model list from the registry plus discovered Ollama models. |
| `GET /v1/capabilities` | Supported backends, model families, runtime feature gates, and experimental features. |
| `GET /v1/runtime` | Loaded model manager state and last-generation metrics. |
| `GET /v1/models/{model_id}/preflight` | Model-specific readiness, blockers, warnings, and split-path hints. |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions with streaming or non-streaming output. |

Common server environment variables:

| Variable | Default | Notes |
| --- | --- | --- |
| `AIRLLM_HOST` | `0.0.0.0` | Bind address. |
| `AIRLLM_PORT` | `8000` | API port. |
| `AIRLLM_DEVICE` | `cuda:0` | Device for BetterAirLLM models; use `cpu` only for small/debug runs. |
| `AIRLLM_DENSE_DEVICE` | unset | Optional device for streamed dense layers. |
| `AIRLLM_DTYPE` | `float16` | Runtime dtype. |
| `AIRLLM_MODELS` | unset | JSON array that replaces `model_registry.json`. |
| `AIRLLM_MODEL` | unset | Prepends one extra HF repo or local checkpoint for quick experiments. |
| `AIRLLM_MODEL_ID` | derived | API id for `AIRLLM_MODEL`. |
| `AIRLLM_MAX_LOADED_MODELS` | `1` | LRU cache size for loaded BetterAirLLM models. |
| `AIRLLM_STREAM_MODE` | `auto` | `auto`, `true`, or `compat`. |
| `AIRLLM_ENABLE_RUNTIME_METRICS` | `true` | Enables runtime stat collection when supported. |
| `HF_TOKEN` | unset | Hugging Face token for gated models. |

## Desktop UI

The `desktop/` app is a Next.js chat shell for the BetterAirLLM experience. It is currently a frontend workspace and still uses local mock responses in `desktop/src/app/page.tsx`; wire it to `http://localhost:8000/v1/chat/completions` before treating it as a production chat client.

```powershell
cd desktop
npm install
npm run dev
```

Open `http://localhost:3000`.

## BetterAirLLM MoE Status

BetterAirLLM adds an experimental strict fused-MoE streaming path. The implementation is deliberately conservative: it enables selective fused runtime only for layouts with adapter correctness tests, and it refuses silent dense fused fallback when `moe_strict_streaming=True`.

Supported:

* Generic MoE architecture dispatch to `BetterAirLLMMoE`.
* Qwen3.5/Qwen3.6 MoE layout detection and selective routed expert adapter tests.
* GPT-OSS-shaped fake fused selective runtime correctness tests.
* Real GPT-OSS layout probing from `config.json` and `model.safetensors.index.json` without downloading full checkpoint shards.
* Strict fused MoE adapter manifests that remove full fused expert tensors from dense shards when the verified `gpt_oss` adapter metadata path is active.
* Detection of the public GPT-OSS MXFP4 packed expert layout (`*_blocks`, `*_scales`, and bias tensors).
* GPT-OSS MXFP4 selected-expert reference dequant/matmul for fake packed tensors and a real `openai/gpt-oss-20b` one-expert smoke path.

Experimental:

* Real `openai/gpt-oss-20b` generation.
* Real `openai/gpt-oss-120b` generation.
* Long context on low-RAM machines.
* Packed MXFP4 selective expert execution at generation time. The current path is a slow correctness reference unless the local environment has a working HF Triton kernel path.

Current speed conclusion:

The next real speed task is fused MXFP4 selected-expert execution. Packed cache helps only modestly, grouped routing is already active, dense cache cannot hit enough, and MXFP4 dequant remains the core cost unless fused with matmul or replaced by a kernel path.

Not supported yet:

* Claiming fast GPT-OSS 120B laptop inference.
* Claiming Mixtral, generic Qwen-MoE, or DeepSeek fused selective runtime without model-specific adapters and correctness tests.

Probe real GPT-OSS metadata without full model download:

```bash
python scripts/probe_moe_layout.py openai/gpt-oss-20b --no-full-download --verify-manifest
python scripts/probe_moe_layout.py openai/gpt-oss-120b --no-full-download --verify-manifest
```

Dry-run estimated GB/token before any real GPT-OSS loading:

```bash
python scripts/benchmark_gb_per_token.py openai/gpt-oss-20b --runtime auto --dry-run-layout
```

Run the gated one-expert MXFP4 smoke test for GPT-OSS 20B:

```bash
python scripts/probe_moe_layout.py openai/gpt-oss-20b --download-one-expert --expert-id 0 --layer-id 0 --confirm-download
python scripts/benchmark_gb_per_token.py openai/gpt-oss-20b --real-gpt-oss-one-expert-smoke --expert-id 0 --layer-id 0 --confirm-download
```

Run the staged GPT-OSS 20B one-token smoke tool:

```bash
python scripts/smoke_gpt_oss_20b_one_token.py openai/gpt-oss-20b --max-seq-len 128 --max-new-tokens 1 --moe-strict-streaming --moe-expert-cache-mb 1024 --moe-cpu-expert-cache-mb 4096 --os-reserved-ram-mb 8192 --preflight-only
python scripts/smoke_gpt_oss_20b_one_token.py openai/gpt-oss-20b --max-seq-len 128 --max-new-tokens 1 --moe-strict-streaming --moe-expert-cache-mb 1024 --moe-cpu-expert-cache-mb 4096 --os-reserved-ram-mb 8192 --one-layer-smoke --confirm-real-run
python scripts/smoke_gpt_oss_20b_one_token.py openai/gpt-oss-20b --max-seq-len 128 --max-new-tokens 1 --moe-strict-streaming --moe-expert-cache-mb 1024 --moe-cpu-expert-cache-mb 4096 --os-reserved-ram-mb 8192 --one-token-real --lazy-build --expert-materialization-mode direct_slice --confirm-real-run
```

The first full one-token run may need to build `splitted_model.moe`, which can take a long time because it writes dense layer shards plus per-expert shards. The smoke script refuses that long build unless `--allow-split-build` or `--lazy-build` is also passed. Lazy build creates and verifies only the next layer needed by decode, persists `progress.json` after each shard group, and resumes instead of restarting after an interruption.

`direct_slice` mode is the preferred GPT-OSS runtime smoke path. It keeps dense/router shards on disk but reads only the selected expert slice from the original packed MXFP4 tensors. The smoke script uses a single raw token by default so the selected expert count stays close to `num_layers * top_k`; pass `--use-harmony-prompt` when you want chat-template formatting instead of a low-level runtime benchmark.

Benchmark or resume split construction:

```bash
python scripts/benchmark_split_build.py openai/gpt-oss-20b --resume-split-build --moe-strict-streaming
python scripts/benchmark_split_build.py openai/gpt-oss-20b --layer-name model.layers.0. --resume-split-build --moe-strict-streaming
```

Real GPT-OSS benchmark runs are gated and require explicit confirmation:

```bash
python scripts/benchmark_gb_per_token.py openai/gpt-oss-20b --runtime auto --max-new-tokens 1 --max-seq-len 128 --moe-strict-streaming --moe-expert-cache-mb 1024 --moe-cpu-expert-cache-mb 4096 --os-reserved-ram-mb 8192 --confirm-real-run
```

GPT-OSS models should be used with OpenAI's harmony chat format for quality evaluation. Raw text prompts in `benchmark_gb_per_token.py` are low-level runtime benchmarks only, not model quality benchmarks.

<a href="https://github.com/lyogavin/airllm/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/lyogavin/airllm?style=social)</a>
[![Downloads](https://static.pepy.tech/personalized-badge/airllm?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/airllm)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/wechat-Anima-brightgreen?logo=wechat)](https://static.aicompose.cn/static/wecom_barcode.png?t=1671918938)
[![Discord](https://img.shields.io/discord/1175437549783760896?logo=discord&color=7289da
)](https://discord.gg/2xffU5sn)
[![PyPI - BetterAirLLM](https://img.shields.io/pypi/format/airllm?logo=pypi&color=3571a3)
](https://pypi.org/project/airllm/)
[![Website](https://img.shields.io/website?up_message=blog&url=https%3A%2F%2Fmedium.com%2F%40lyo.gavin&logo=medium&color=black)](https://medium.com/@lyo.gavin)
[![Website](https://img.shields.io/badge/Gavin_Li-Blog-blue)](https://gavinliblog.com)
[![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dgavinli%26type%3Dpatrons&style=flat)](https://patreon.com/gavinli)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/lyogavin?logo=GitHub&color=lightgray)](https://github.com/sponsors/lyogavin)

## AI Agents Recommendation:

* [Best AI Game Sprite Generator](https://godmodeai.co)

* [Best AI Facial Expression Editor](https://crazyfaceai.com)

## Updates
[2024/08/20] v2.11.0: Support Qwen2.5

[2024/08/18] v2.10.1 Support CPU inference. Support non sharded models. Thanks @NavodPeiris for the great work! 

[2024/07/30] Support Llama3.1 **405B** ([example notebook](https://colab.research.google.com/github/lyogavin/airllm/blob/main/air_llm/examples/run_llama3.1_405B.ipynb)). Support **8bit/4bit quantization**.

[2024/04/20] BetterAirLLM supports Llama3 natively already. Run Llama3 70B on 4GB single GPU.

[2023/12/25] v2.8.2: Support MacOS running 70B large language models.

[2023/12/20] v2.7: Support BetterAirLLMMixtral.

[2023/12/20] v2.6: Added AutoModel, automatically detect model type, no need to provide model class to initialize model.

[2023/12/18] v2.5: added prefetching to overlap the model loading and compute. 10% speed improvement.

[2023/12/03] added support of **ChatGLM**, **QWen**, **Baichuan**, **Mistral**, **InternLM**!

[2023/12/02] added support for safetensors. Now support all top 10 models in open llm leaderboard.

[2023/12/01] airllm 2.0. Support compressions: **3x run time speed up!**

[2023/11/20] airllm Initial version!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lyogavin/airllm&type=Timeline)](https://star-history.com/#lyogavin/airllm&Timeline)

## Python API Quickstart

The lower-level AirLLM-style Python API is still available when you want to bypass the OpenAI-compatible server and call the local runtime directly.

### 1. Install package

First, install the airllm pip package.

```bash
pip install airllm
```

### 2. Inference

Then, initialize BetterAirLLMLlama2, pass in the huggingface repo ID of the model being used, or the local path, and inference can be performed similar to a regular transformer model.

(*You can also specify the path to save the splitted layered model through **layer_shards_saving_path** when init BetterAirLLMLlama2.*

```python
from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

# or use model's local path...
#model = AutoModel.from_pretrained("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
        'What is the capital of United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)

```
 
 
Note: During inference, the original model will first be decomposed and saved layer-wise. Please ensure there is sufficient disk space in the huggingface cache directory.
 

## Model Compression - 3x Inference Speed Up!

We just added model compression based on block-wise quantization-based model compression. Which can further **speed up the inference speed** for up to **3x** , with **almost ignorable accuracy loss!** (see more performance evaluation and why we use block-wise quantization in [this paper](https://arxiv.org/abs/2212.09720))

![speed_improvement](https://github.com/lyogavin/airllm/blob/main/assets/airllm2_time_improvement.png?v=2&raw=true)

#### How to enable model compression speed up:

* Step 1. make sure you have [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) installed by `pip install -U bitsandbytes `
* Step 2. make sure airllm verion later than 2.0.0: `pip install -U airllm` 
* Step 3. when initialize the model, passing the argument compression ('4bit' or '8bit'):

```python
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct",
                     compression='4bit' # specify '8bit' for 8-bit block-wise quantization 
                    )
```

#### What are the differences between model compression and quantization?

Quantization normally needs to quantize both weights and activations to really speed things up. Which makes it harder to maintain accuracy and avoid the impact of outliers in all kinds of inputs.

While in our case the bottleneck is mainly at the disk loading, we only need to make the model loading size smaller. So, we get to only quantize the weights' part, which is easier to ensure the accuracy.

## Configurations
 
When initialize the model, we support the following configurations:

* **compression**: supported options: 4bit, 8bit for 4-bit or 8-bit block-wise quantization, or by default None for no compression
* **profiling_mode**: supported options: True to output time consumptions or by default False
* **layer_shards_saving_path**: optionally another path to save the splitted model
* **hf_token**: huggingface token can be provided here if downloading gated models like: *meta-llama/Llama-2-7b-hf*
* **prefetching**: prefetching to overlap the model loading and compute. By default, turned on. For now, only BetterAirLLMLlama2 supports this.
* **delete_original**: if you don't have too much disk space, you can set delete_original to true to delete the original downloaded hugging face model, only keep the transformed one to save half of the disk space. 

## Model Support Matrix

BetterAirLLM now separates the models configured by this server from the broader model families that the underlying `AutoModel` dispatcher can load.

Default configured BetterAirLLM models are loaded from `model_registry.json`:

| API model id | Repository | Family | Status | Notes |
| --- | --- | --- | --- | --- |
| `qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | `qwen3_5_moe` | Experimental | Selective routed expert streaming path. |
| `qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B` | `qwen_moe` | Experimental | Architecture dispatch is covered; fast selective runtime is layout-gated. |
| `mistral-7b-instruct` | `mistralai/Mistral-7B-Instruct-v0.1` | `mistral` | Supported | Standard layer streaming. |
| `llama2-7b-chat` | `meta-llama/Llama-2-7b-chat-hf` | `llama` | Supported | Requires a Hugging Face token with model access. |

The default registry can be replaced with `AIRLLM_MODELS` as a JSON array. `AIRLLM_MODEL` still prepends one extra model for quick local experiments.

Supported Hugging Face architecture families include MoE, Qwen2/Qwen2.5, QWen, Baichuan, ChatGLM, InternLM, Mistral, and Llama. Unknown architectures fall back to the Llama2 runtime as best effort. GPT-OSS MXFP4 and Qwen3.5/Qwen3.6 MoE selective expert runtime are experimental/staged paths. Fast fused MXFP4 selected-expert kernels are not implemented in this productization pass and remain future performance work.

GGUF models are not loaded directly by the BetterAirLLM backend. They are supported through the Ollama proxy backend and appear as `ollama/<model-name>` when discovered.

Inspect support and readiness through the API:

```powershell
Invoke-RestMethod http://localhost:8000/v1/capabilities | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:8000/v1/models/qwen3.6-35b-a3b/preflight | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:8000/v1/models/ollama%2Fllama3.2%3A3b/preflight | ConvertTo-Json -Depth 8
```

## Using Ollama Models Through BetterAirLLM

BetterAirLLM can expose models already downloaded by Ollama through the same OpenAI-compatible server. Hugging Face and local safetensors checkpoints still use BetterAirLLM. Ollama GGUF models are listed as `ollama/<model-name>` and proxied to the local Ollama OpenAI-compatible API.

Start Ollama and confirm downloaded models:

```powershell
ollama list
Invoke-RestMethod http://localhost:11434/api/tags
```

Start BetterAirLLM:

```powershell
python server.py
```

List all available BetterAirLLM and Ollama models:

```powershell
Invoke-RestMethod http://localhost:8000/v1/models | ConvertTo-Json -Depth 8
```

Call an Ollama model through BetterAirLLM:

```powershell
$body = @{
  model = "ollama/llama3.2:3b"
  messages = @(
    @{ role = "user"; content = "Say hello from Ollama through BetterAirLLM." }
  )
  max_tokens = 64
  stream = $false
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri http://localhost:8000/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```

Test streaming:

```powershell
$body = @{
  model = "ollama/llama3.2:3b"
  messages = @(@{ role = "user"; content = "Stream one short sentence." })
  max_tokens = 64
  stream = $true
} | ConvertTo-Json -Depth 10

Invoke-WebRequest `
  -Uri http://localhost:8000/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Useful environment variables:

```powershell
$env:AIRLLM_DISCOVER_OLLAMA = "true"
$env:AIRLLM_OLLAMA_BASE_URL = "http://localhost:11434"
$env:AIRLLM_OLLAMA_TIMEOUT_SECONDS = "5"
$env:AIRLLM_OLLAMA_DISCOVERY_TTL_SECONDS = "10"
```

Troubleshooting:

* If `/v1/models` does not show `ollama/...` models, make sure Ollama is running and `Invoke-RestMethod http://localhost:11434/api/tags` works.
* If BetterAirLLM reports that Ollama blobs exist but the daemon is unavailable, start or restart Ollama.
* If a request fails for `ollama/<name>`, check that `<name>` exactly matches `ollama list`.
* GGUF models are not loaded through BetterAirLLM directly. They use the Ollama backend. Use Hugging Face or local safetensors checkpoints for the BetterAirLLM backend.

## MacOS

Just install airllm and run the code the same as on linux. See more in [Quick Start](#quickstart).

* make sure you installed [mlx](https://github.com/ml-explore/mlx?tab=readme-ov-file#installation) and torch
* you probably need to install python native see more [here](https://stackoverflow.com/a/65432861/21230266)
* only [Apple silicon](https://support.apple.com/en-us/HT211814) is supported

Example [python notebook] (https://github.com/lyogavin/airllm/blob/main/air_llm/examples/run_on_macos.ipynb)


## Example Python Notebook

Example colabs here:

<a target="_blank" href="https://colab.research.google.com/github/lyogavin/airllm/blob/main/air_llm/examples/run_all_types_of_models.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### example of other models (ChatGLM, QWen, Baichuan, Mistral, etc):

<details>


* ChatGLM:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=True)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache= True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```

* QWen:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen-7B")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


* Baichuan, InternLM, Mistral, etc:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("baichuan-inc/Baichuan2-7B-Base")
#model = AutoModel.from_pretrained("internlm/internlm-20b")
#model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


</details>


#### To request other model support: [here](https://docs.google.com/forms/d/e/1FAIpQLSe0Io9ANMT964Zi-OQOq1TJmnvP-G3_ZgQDhP7SatN0IEdbOg/viewform?usp=sf_link)



## Acknowledgement

A lot of the code are based on SimJeg's great work in the Kaggle exam competition. Big shoutout to SimJeg:

[GitHub account @SimJeg](https://github.com/SimJeg), 
[the code on Kaggle](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag), 
[the associated discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446414).


## FAQ

### 1. MetadataIncompleteBuffer

safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer

If you run into this error, most possible cause is you run out of disk space. The process of splitting model is very disk-consuming. See [this](https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/12). You may need to extend your disk space, clear huggingface [.cache](https://huggingface.co/docs/datasets/cache) and rerun. 

### 2. ValueError: max() arg is an empty sequence

Most likely you are loading QWen or ChatGLM model with Llama2 class. Try the following:

For QWen model: 

```python
from airllm import AutoModel #<----- instead of BetterAirLLMLlama2
AutoModel.from_pretrained(...)
```

For ChatGLM model: 

```python
from airllm import AutoModel #<----- instead of BetterAirLLMLlama2
AutoModel.from_pretrained(...)
```

### 3. 401 Client Error....Repo model ... is gated.

Some models are gated models, needs huggingface api token. You can provide hf_token:

```python
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", #hf_token='HF_API_TOKEN')
```

### 4. ValueError: Asking to pad but the tokenizer does not have a padding token.

Some model's tokenizer doesn't have padding token, so you can set a padding token or simply turn the padding config off:

 ```python
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False  #<-----------   turn off padding 
)
```

## Citing BetterAirLLM

If you find
BetterAirLLM useful in your research and wish to cite it, please use the following
BibTex entry:

```
@software{airllm2023,
  author = {Gavin Li},
  title = {BetterAirLLM: scaling large language models on low-end commodity computers},
  url = {https://github.com/lyogavin/airllm/},
  version = {0.0},
  year = {2023},
}
```


## Contribution 

Welcomed contributions, ideas and discussions!

If you find it useful, please ⭐ or buy me a coffee! 🙏

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)
