<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg">
    <img src="assets/logo.png" alt="SAGE-FFT — FFT butterfly logo" width="640">
  </picture>
</p>

[中文说明](README.zh-CN.md)

LLM-directed hierarchical FFT transformation and scheduling for CUDA-to-NPU migration.
Version **0.3.0**. Author and maintainer: **even**.

SAGE-FFT imports registered CUDA FFT pipelines into an FFT contract and stage IR.
The compiler checks grouping, epilogue fusion and core mapping, then emits Ascend C
sources and launch plans. An optional model selects legal transformations during
offline tuning; deployed FFT execution does not call the model.

## Install and test

Python 3.10 or newer is required. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
sage validate-cpu --shape 8,8,8
```

CPU tests use generated inputs and synthetic transport responses. They require
neither an accelerator nor an API key. CPU interpreter timings are not NPU timings.

## Generate a pipeline

```bash
sage emit-cuda --shape 8,8,8 --output runs/reference.cu
sage import-cuda runs/reference.cu
sage migrate runs/reference.cu --variant grouped --blocks 8 --output runs/migrated
sage emit-npu --shape 8,8,8 --variant staged --output runs/staged
sage emit-catalog --workload M3 --output runs/catalog
sage make-inputs --shape 8,8,8 --seed 1 --output runs/input_1
```

The adapter supports the registered literal cuFFT source family, not arbitrary
CUDA programs. The default FP32 3D staged plan uses 20 launches; grouping with
both epilogues uses six. The separate fixed-shape FP16 extension supports local
fusion and is outside the indexed FP32 catalog.

## Run on hardware

See [reproduction](docs/REPRODUCIBILITY.md), [native benchmarks](docs/NATIVE_BENCHMARKS.md)
and [model configuration](docs/LLM_CONFIGURATION.md). CANN, CUDA, model weights
and the optional MindSpore AI CPU plugin are installed separately.

```bash
python -m pip install -e ".[remote]"
# Export the applicable settings documented in .env.example first.
sage search runs/reference.cu --policy index --output runs/index_search
```

Use `--policy greedy` for the analytic control without a model. Searches perform
real target measurements; model-backed policies also contact the configured
service. Each user provides their own credentials through environment variables.
The `.env.example` file is documentation and is not loaded automatically.

## Layout

| Directory | Contents |
| --- | --- |
| `src/sage_fft/` | Contract, IR, compiler, search, model transport and analysis |
| `benchmarks/` | Native FP32/FP16 sources, input generators and study runners |
| `tests/` | Numerical, compiler, recovery and transport regression tests |
| `examples/` | Offline usage examples |
| `docs/` | Architecture, configuration and experiment instructions |
| `scripts/` | Source checks and audits of caller-generated results |

This source distribution contains no recorded experiments, model conversations,
personal machine configuration or paper-version archives. Benchmark commands
create new results in the caller's configured workspace. Seeds do not guarantee
identical hosted-model decisions; report measured outcomes and failed proposals.

## Development

```bash
python -m ruff check src tests benchmarks scripts examples
python scripts/check_repository.py --strict-warnings
python -m build
```

The included GitHub Actions workflow runs CPU checks and builds the package.
Hardware measurements are separate. See [contributing](CONTRIBUTING.md),
[security](SECURITY.md) and [licensing status](LICENSING.md).
