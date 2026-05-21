![betterairllm_logo](https://github.com/medenmite/BetterAirLLM/blob/main/assets/betterairllm_logo_sm.png?v=3&raw=true)

[**Quickstart**](#quickstart) | 
[**Configurations**](#class-initialization-parameters) | 
[**MacOS**](#macos) | 
[**Example Notebooks**](#example-python-notebook) | 
[**Troubleshooting & FAQ**](#troubleshooting--faq)

**BetterAirLLM** optimizes inference memory usage, allowing 70B large language models to run inference on a single 4GB GPU card. No quantization, distillation, pruning, or other model compression techniques that would result in degraded model performance are needed.

This subdirectory contains the core `betterairllm` Python library. If you want to run the OpenAI-compatible REST server instead, please refer to the root [README.md](../README.md).

---

## Updates

[2026/05/20-21] BetterAirLLM now supports GPT-OSS(20B & 120B), Qwen MoE and Ollama-downloaded models.
[2024/04/20] AirLLM supports Llama3 natively already. Run Llama3 70B on 4GB single GPU.
[2023/12/25] v2.8.2: Support MacOS running 70B large language models.
[2023/12/20] v2.7: Support BetterAirLLMMixtral. 
[2023/12/20] v2.6: Added AutoModel, automatically detect model type, no need to provide model class to initialize model.
[2023/12/18] v2.5: added prefetching to overlap the model loading and compute. 10% speed improvement.
[2023/12/03] added support of **ChatGLM**, **Qwen**, **Baichuan**, **Mistral**, **InternLM**!
[2023/12/02] added support for safetensors. Now support all top 10 models in open llm leaderboard.
[2023/12/01] AirLLM 2.0. Support compressions: **3x run time speed up!**
[2023/11/20] AirLLM Initial version!

---

## Quickstart

### 1. Install Package

Install the `betterairllm` package locally:

```bash
pip install betterairllm
```

### 2. Basic Inference Example

Initialize the model using `AutoModel.from_pretrained`. The first time you execute this, the model weights will be decompressed and saved layer-wise locally to save VRAM during inference.

```python
from betterairllm import AutoModel

MAX_LENGTH = 128
# Loads from Hugging Face Hub (or local directory)
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

input_text = ["What is the capital of United States?"]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False
)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
```

> [!NOTE]
> During inference, the original model will first be decomposed and saved layer-wise. Please ensure there is sufficient disk space in the Hugging Face cache directory.

---

## Model Compression - 3x Inference Speed Up!

We support weight-only block-wise quantization-based model compression. This can speed up the disk-load bottleneck by up to **3x** with negligible accuracy loss.

#### Enable Model Compression:

1. Install `bitsandbytes` dependency:
   ```bash
   pip install -U bitsandbytes
   ```
2. Pass the `compression` argument (`4bit` or `8bit`) to the loader:
   ```python
   model = AutoModel.from_pretrained(
       "garage-bAInd/Platypus2-70B-instruct",
       compression='4bit' # specify '8bit' for 8-bit block-wise quantization 
   )
   ```

---

## Class Initialization Parameters

When loading a model through `AutoModel.from_pretrained(...)` or during class construction, the following parameters are supported:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| **model_name** | `str` | *Required* | Hugging Face repository name (e.g. `meta-llama/Llama-2-7b-hf`) or local directory path. |
| **device** | `str` | `"cuda:0"` | Target execution device (e.g. `"cuda:0"`, `"cpu"`). |
| **dense_device** | `Optional[str]` | `None` | Alternative device override specifically for dense model layers. |
| **dtype** | `torch.dtype` | `torch.float16` | Weights precision type (`float16`, `bfloat16`, `float32`). |
| **max_seq_len** | `int` | `2048` | Target sequence length / context window size configuration. |
| **compression** | `Optional[str]` | `None` | Pre-quantize weights to `"4bit"` or `"8bit"`. |
| **prefetching** | `bool` | `True` | Overlap weight disk reading with active computation in a separate thread. |
| **moe_expert_sharding** | `bool` | `False` | Enables sharding of MoE expert blocks during initialization. |
| **moe_strict_streaming** | `bool` | `True` | Enforce sequential expert execution and forbid silent fallback. |
| **moe_allow_dense_fallback** | `bool` | `False` | Permit loading all experts into memory if streaming fails. |
| **os_reserved_ram_mb** | `int` | `8192` | Memory safety reserve to prevent host operating system OOMs. |
| **abort_unsafe_context** | `bool` | `True` | Raise an error early if estimated memory needs exceed system limits. |
| **resume_split_build** | `bool` | `True` | Resume incremental MoE partition splits if interrupted. |
| **lazy_build** | `bool` | `True` | Create layer and expert partitions dynamically during forward passes. |
| **expert_materialization_mode** | `str` | `"direct_slice"` | Method used to read MoE experts (`"direct_slice"` or `"full_shard"`). |
| **layer_cleanup_interval** | `int` | `8` | Cleanup rate of processed layer weights from system memory. |
| **mxfp4_device** | `str` | `"cuda"` | Target device where MXFP4 weights are unpacked/dequantized. |
| **expert_matmul_device** | `str` | `"cuda"` | Target device where MoE expert matrix multiplications are performed. |
| **max_vram_mb** | `int` | `7600` | Safety VRAM ceiling configuration. |
| **keep_nontransformer_resident** | `bool` | `True` | Keep non-transformer layer types (embeddings, heads) resident in VRAM. |
| **quiet_progress** | `bool` | `False` | Turn off console execution progress bars. |

---

## Python API Integration for Diverse Architectures

### ChatGLM:
```python
from betterairllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base")
input_text = ['What is the capital of China?']
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=True)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
print(model.tokenizer.decode(generation_output.sequences[0]))
```

### Qwen (Dense & MoE):
```python
from betterairllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen-7B")
input_text = ['What is the capital of China?']
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
print(model.tokenizer.decode(generation_output.sequences[0]))
```

### Baichuan / InternLM / Mistral:
```python
from betterairllm import AutoModel
model = AutoModel.from_pretrained("baichuan-inc/Baichuan2-7B-Base")
# model = AutoModel.from_pretrained("internlm/internlm-20b")
# model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
input_text = ['What is the capital of China?']
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=128)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
print(model.tokenizer.decode(generation_output.sequences[0]))
```

---

## MacOS

BetterAirLLM supports Apple Silicon chips natively:

* Install [mlx](https://github.com/ml-explore/mlx) and PyTorch.
* Ensure you are running a native ARM64 Python installation.
* Check out the [MacOS example notebook](https://github.com/medenmite/BetterAirLLM/blob/main/air_llm/examples/run_on_macos.ipynb) for detailed setup.

---

## Troubleshooting & FAQ

### 1. safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer
This indicates that the disk partition ran out of space during model partition splitting. Model splitting writes weights layer-by-layer and requires significant disk space. Clean up the Hugging Face cache directory (`~/.cache/huggingface/hub/`) and free up storage space before retrying.

### 2. ValueError: max() arg is an empty sequence
This occurs when loading a non-Llama model using a specific model architecture class. Ensure you load models using `AutoModel` which dynamically selects the correct architecture runtime dispatcher:
```python
# CORRECT
from betterairllm import AutoModel
model = AutoModel.from_pretrained("Qwen/Qwen-7B")
```

### 3. 401 Client Error: Repo model is gated
Gated Hugging Face repositories require explicit agreement and authentication. Pass your Hugging Face authentication token:
```python
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", hf_token='HF_API_TOKEN')
```

### 4. ValueError: Asking to pad but the tokenizer does not have a padding token
If the target model's tokenizer does not support padding tokens, set `padding=False` during tokenization:
```python
input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False # Turn off padding
)
```
