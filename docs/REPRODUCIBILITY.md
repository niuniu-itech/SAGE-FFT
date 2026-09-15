# Reproduction guide

Run from the repository root. Install `python -m pip install -e ".[dev,remote,plots]"`
and configure a fresh workspace for each experiment. See
[model configuration](LLM_CONFIGURATION.md) for API and local runtime settings.

## Fresh FP32 measurements

Configure SSH and SDK paths using `.env.example`, then select new values for
`SAGE_WORKDIR` and `SAGE_REMOTE_ROOT`. The remote root must be dedicated to this
run. `SAGE_NPU_SETUP` can source the installed CANN toolkit and set the compiler
and CMake search paths required by the machine.

For a machine with little space on its system partition, the setup file can
route new build directories to scratch storage. Copy `scripts/cmake_scratch.py`
to that machine and define a wrapper using the actual CMake executable:

```bash
export SAGE_CMAKE_SCRATCH_ROOT=/tmp/sage-builds
cmake() {
  python3 /path/to/SAGE-FFT/scripts/cmake_scratch.py /usr/bin/cmake "$@"
}
```

Existing build directories remain in place. Raw measurements and source files
stay in the selected experiment directories.

```bash
python -m sage_fft.campaign ablation
python -m sage_fft.run_cuda
```

The first command builds 11 configurations for each of seven workloads and
checks three input seeds. Separate forward and inverse checks follow. Completed
cases can be resumed only when their records and artifact identities agree.

The larger FP32 shapes can be measured independently of the policy study:

```python
from sage_fft.config import prepare_workspace
from sage_fft.campaign import Runner
from sage_fft.quality_experiments import scaling

prepare_workspace()
runner = Runner()
try:
    scaling(runner)
finally:
    runner.s.close()
    runner.c.close()
```

This measures staged, grouped and grouped-with-epilogues configurations at
16³, 32³ and 64³ with eight blocks. The larger-shape protocol uses
three warmups and ten timing groups of three executions. The native matched
comparison below uses a longer timing batch; do not attribute differences
between those protocols solely to the implementation.

## Available native FFT baseline

The baseline requires the MindSpore 2.4.0 custom AI CPU FFTWithSize plugin in
addition to CANN. Supply its installation through `ASCEND_CUSTOM_OPP_PATH` and
put its `op_impl/cpu/aicpu_kernel/impl` directory on `LD_LIBRARY_PATH`.
The plugin and vendor SDK are not redistributed.

```bash
python -m benchmarks.native_fft.generate --output runs/native-prepared
python -m benchmarks.build_native --backend npu --source benchmarks/native_fft \
  --build-dir runs/native-build --cann-root "$ASCEND_CANN_PACKAGE_PATH"
```

Build each of `n8_staged`, `n8_sage`, `n16_staged`, and `n16_sage` beneath
`runs/native-prepared/controls`, placing its build under that control's `build`
directory. For example:

```bash
python -m benchmarks.build_native --backend npu \
  --source runs/native-prepared/controls/n8_staged \
  --build-dir runs/native-prepared/controls/n8_staged/build \
  --cann-root "$ASCEND_CANN_PACKAGE_PATH"
python -m benchmarks.native_fft.run --enable-hardware \
  --prepared runs/native-prepared --native-build runs/native-build \
  --output runs/native-results
```

The default matrix is two shapes, four implementations, three input seeds and
three sessions. Each session uses 1,000 warmups and 11 groups of 1,000 complete
pipelines. Forward and complete-pipeline outputs are checked outside timing.
The migrated controls are generated through the FFT IR and share this timing
wrapper. This native matrix covers 8³ and 16³ only.

## Search and feedback controls

Configure the model using [LLM_CONFIGURATION.md](LLM_CONFIGURATION.md). For the
local controls use DeepSeek-R1-Distill-Llama-8B Q4_K_M model, Ollama
0.5.7, and its 6,144-token context. Hosted controls use DeepSeek-V3.2 through the
OpenAI-compatible endpoint. Models and API credentials are supplied by the user.

```bash
python -m benchmarks.search_study --kind local
python -m benchmarks.search_study --kind hosted
```

The local command runs seven policies on M2 and M3, with three search seeds and
12 proposals. The hosted command runs the two final decoder interfaces and
analytic greedy under the same per-run budget. Independent hosted requests may
overlap, while the shared target serializes NPU measurements. Their outputs use
different subdirectories. Configure the appropriate model before each command.
Final selected realizations receive five additional timing rounds with ten
samples per round on input seeds 2 and 3. Those acceptance rounds do not replace
the common candidate-union calibration used for the paper's search comparison.

For exhaustive small-space controls, transfer priors and full visited-candidate
calibration, use the indexed-catalog sequence below. The transfer study requires freshly
measured M1 prior controls and the M2 greedy seed-41 trajectory. It must not read
the held-out workload's answers as priors. Model transport failures, invalid
decisions and rejected candidates remain in the trace.

For the complete local comparison, run the following with the local model
configuration and the same experiment workspace:

```bash
python -m sage_fft.catalog smoke --workloads M1,M2,M3
python -m sage_fft.catalog oracle --workloads M1,M2
python -m sage_fft.transfer_study
python -m sage_fft.cuda_v46 --workloads M1,M2,M3,M4
python -m sage_fft.action_diagnostics
python -m sage_fft.final_calibration
python -m sage_fft.analyze_v46
```

The transfer step requires the completed local `search_study` above. The CUDA
step requires the configured CUDA machine. After both the local calibration and
hosted `search_study` have finished, calibrate their candidate union:

```bash
python -m benchmarks.calibrate_search
```

This uses nine hosted trials directly. The separate `calibrate_hosted` driver
requires the 252-proposal matrix from `hosted_study` and `hosted_schema`; do not
mix those drivers with the nine-trial calibration entry point.

## Additional Qwen controls

Use Qwen2.5-7B-Instruct Q4_K_M through a llama.cpp server built for the host's
CPU and glibc. Keep both model shards together when using a split GGUF. The
example configuration uses an 8,192-token context and 16 CPU threads:

```bash
"$SAGE_LLAMA_SERVER" -m "$SAGE_QWEN_MODEL" --alias qwen2.5-7b-instruct \
  --host 127.0.0.1 --port 18090 -c 8192 -t 16 --jinja
```

Set those two shell variables to the installed executable and model file.
For a remote server, forward its loopback port through SSH; configure the
following endpoint on the control host only after that tunnel is active:

```bash
export SAGE_LLM_BACKEND=openai
export SAGE_LLM_BASE_URL=http://127.0.0.1:18090/v1
export SAGE_LLM_MODEL=qwen2.5-7b-instruct
unset SAGE_LLM_API_KEY
python -m benchmarks.legacy_controls --enable-hardware
```

The local server example has no API key and listens only on loopback. The
driver retains temperature 0.4 and a 40-token output budget. It runs four
policies with three seeds and eight proposals each, transform-only checks,
five calibration rounds per selected implementation, and the complete
18-configuration reference across three rounds. The base and CUDA scaling
measurements must already exist in the same workspace. Missing reference
configurations are built and numerically checked before calibration; they are
never supplied with invented timings.

## FP16 local fusion

[NATIVE_BENCHMARKS.md](NATIVE_BENCHMARKS.md) gives the complete tuning,
confirmation and held-out correctness matrices, input generation, native
build commands and precision gates. Use all six variants. Each speedup divides
the Staged latency by the candidate latency at the same block count. Compare
`grouped6_opt` with `fused3_opt` to isolate fusion with matched twiddle arithmetic.

After generating inputs and compiling all six NPU variants, run the complete
matrix with one command:

```bash
python -m benchmarks.fusion_study --backend npu --enable-hardware \
  --inputs runs/fp16-inputs --source runs/fp16-generated --output runs/fp16-results
python -m benchmarks.fusion_study --backend cuda --enable-hardware \
  --inputs runs/fp16-inputs --source benchmarks/fp16/cuda \
  --binary runs/cuda-build/sage_cuda_fft --output runs/cuda-fp16-results
```

These commands execute 66 NPU cases and 14 CUDA cases respectively. The NPU
source directory must contain each variant's `build/sage_fft`. The CUDA binary
is supplied explicitly. Existing output directories are preserved; choose a new
directory for another complete matrix.

```bash
python -m benchmarks.summarize_fusion runs/fp16-results --output runs/fp16-table.json
```

Aggregation requires all 36 NPU confirmation runs, verifies output hashes and
matching inputs, and rejects mixed binaries or failed numerical checks.

## Acceptance

A complete fresh run retains generated source and IR, compiler logs, executable
hashes, input and output hashes, correctness checks, all timing samples and
model requests/responses without credentials. A numeric failure cannot produce
an accepted timing. CPU tests establish semantic and orchestration behavior;
the corresponding device outputs establish hardware execution. Report observed timings, model outcomes and failed proposals from each run.

Recheck your generated base, larger-shape and static policy outputs independently:

```bash
python -m benchmarks.audit_artifacts "$SAGE_WORKDIR" --output runs/static-audit.json
python scripts/audit_search_runs.py --help
```

The static audit checks executable and source hashes, reconstructs the inputs,
compares saved outputs with an independent reference, and recomputes latency
medians. Failed build and proposal records remain in the report. A successful
artifact audit does not by itself establish that every requested trial ran;
check the expected matrix sizes and search budgets as well.
