# Configuring the model API

Each user supplies their own provider account and API key. The distributed source contains no project API key; `.env.example` leaves `SAGE_LLM_API_KEY` empty. SAGE reads environment variables and does not automatically load `.env` files.

## SiliconFlow

SiliconFlow uses the OpenAI-compatible backend. Here `openai` names the wire protocol, not the service provider; use a SiliconFlow key for its endpoint. The public endpoint and example model below follow the [SiliconFlow text-generation guide](https://docs.siliconflow.cn/docs/userguide/capabilities/text-generation). Confirm model availability in your own account.

| Variable | Value |
| --- | --- |
| `SAGE_LLM_BACKEND` | `openai` |
| `SAGE_LLM_BASE_URL` | `https://api.siliconflow.cn/v1` |
| `SAGE_LLM_MODEL` | `deepseek-ai/DeepSeek-V3.2` |
| `SAGE_LLM_API_KEY` | Your own SiliconFlow API key |
| `SAGE_LLM_TIMEOUT` | `180` |
| `SAGE_LLM_DISABLE_THINKING` | `1` |

Create a key in the SiliconFlow account console as described in its [quickstart](https://docs.siliconflow.cn/docs/userguide/quickstart). Set the base URL through `/v1`; SAGE appends `/chat/completions` itself.

Windows PowerShell, in the terminal where you will run SAGE:

```powershell
$env:SAGE_LLM_BACKEND = "openai"
$env:SAGE_LLM_BASE_URL = "https://api.siliconflow.cn/v1"
$env:SAGE_LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"
$env:SAGE_LLM_TIMEOUT = "180"
$env:SAGE_LLM_DISABLE_THINKING = "1"
$sageApiKeyInput = Read-Host "SiliconFlow API key" -AsSecureString
$env:SAGE_LLM_API_KEY = [System.Net.NetworkCredential]::new("", $sageApiKeyInput).Password
Remove-Variable sageApiKeyInput
```

Linux/macOS Bash:

```bash
export SAGE_LLM_BACKEND=openai
export SAGE_LLM_BASE_URL=https://api.siliconflow.cn/v1
export SAGE_LLM_MODEL=deepseek-ai/DeepSeek-V3.2
export SAGE_LLM_TIMEOUT=180
export SAGE_LLM_DISABLE_THINKING=1
read -r -s -p "SiliconFlow API key: " SAGE_LLM_API_KEY
export SAGE_LLM_API_KEY
```

The interactive prompt avoids putting the literal key into command history. These settings apply to the current terminal session and its child processes. If you prefer your own environment-file loader, it must explicitly export the values before running SAGE. Renaming `.env.example` to `.env` alone does not activate it.

## Run the paper algorithm

The NPU, SSH and CANN settings are separate; configure them using [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Then:

```bash
python -m pip install -e ".[remote]"
sage emit-cuda --shape 8,8,8 --output runs/source.cu
sage search runs/source.cu --policy index --seed 41 --output runs/index_search
```

This command performs real NPU measurements and model calls. Choose a new output directory for each search. The indexed hosted controller requests structured diagnosis and a decision; the configured model/service must accept the requested JSON-schema format. A different model or provider is a new experiment, not a reproduction of the paper's measured results.

## How the key is handled

`src/sage_fft/llm.py` reads `SAGE_LLM_API_KEY` at request time and adds it to the HTTP `Authorization` header. That header is not part of the serialized request records. Transport failures retain the exception type rather than logging provider error bodies, URLs or headers. Regression tests in `tests/test_model.py` exercise these paths with synthetic keys.

`.gitignore` excludes `.env` and private environment files while retaining the empty `.env.example` template. Do not put an actual key in the template or source code. Existing ignored local files are outside the public-source scan.

For another OpenAI-compatible provider, change the base URL, model ID and key to that provider's settings. The model must support the request parameters used by the selected controller. For local Ollama controls, use the reproduction guide's corresponding study driver and set `SAGE_LLM_BACKEND=ollama` with that server's base URL and installed model ID.
