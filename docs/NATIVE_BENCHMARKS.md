# Native FFT benchmarks

These local tools expose the fixed `8 x 8 x 8`, batch-one weighted FFT pipeline
`ifftn(fftn(x) * h)`. They include actual AscendC
kernel sources and cuFFT hosts. No binary, vendor header, remote runner or device
credential is distributed. Native compilation and execution must be validated on
the user's installed CANN/CUDA toolchain; packaging and CPU tests do not establish
a new hardware performance result.

| Directory below `benchmarks/` | Implementation | NPU launches |
| --- | --- | ---: |
| `fp32/staged20` | Staged scalar FP32 butterflies | 20 |
| `fp32/sage6` | Grouped scalar FP32 butterflies | 6 |
| `fp16/staged20` | Native half vector butterflies, staged execution | 20 |
| `fp16/grouped6` | Three forward and three inverse axis groups | 6 |
| `fp16/fused5` | Also fuses central forward/multiply/inverse work | 5 |
| `fp16/fused3` | Also fuses both outer-axis pairs inside a tile | 3 |
| `fp16/grouped6_opt` | Six groups with unit/quarter-turn twiddle specialization | 6 |
| `fp16/fused3_opt` | Three groups with the same twiddle specialization | 3 |
| `fp32/cuda` | FP32 cuFFT diagnostic host | — |
| `fp16/cuda` | `CUDA_C_16F` cuFFT plus native half multiply/normalization | — |

FP16 tiles contain 64 complex values, use eight active lanes in each 16-half
vector, and allocate 2560 UB bytes per block. Blocks 1, 4 and 8 are exposed.
FP32 sources retain scalar arithmetic; these are not claimed to be
optimized FP32 vector kernels. Both precision families keep the input immutable
while ping-pong scratch buffers carry intermediate outputs.

## CPU preparation

Run from the repository root with Python 3.10+ and NumPy 1.24+ installed (the
repository development dependencies include pytest):

```bash
python -m benchmarks.generate_inputs --precision fp32 --output build/inputs/fp32
python -m benchmarks.generate_inputs --precision fp16 --output build/inputs/fp16
python -m benchmarks.fp16.generate --output build/generated-fp16
python -m pytest tests/test_native_sources.py -q
```

`generate_inputs` defaults to seeds 1–3 for FP32 and seeds 1–5 for FP16. Seeds
1–3 use `Generator(PCG64(seed))`, draw all real normal components followed by all
imaginary normal components, and draw complex weights with magnitude uniform on
`[0.25, 1)` and phase uniform on `[-pi, pi)`. Values are rounded to complex64
before the FP16 conversion. FP16 seeds 4–5 retain the
held-out rule: 1024 normal `(0, 0.25)` real scalars for `x`, then 1024 uniform
`[-0.5, 0.5)` scalars for `h`, directly rounded to half.

Each input file stores 512 complex numbers as 1024 little-endian real scalars in
interleaved real/imaginary order, C-order shape `(8, 8, 8)`. It is 4096 bytes for
FP32 or 2048 bytes for FP16. `benchmarks/input_reference_sha256.json` contains deterministic input checksums
for regression testing. Generated `input_manifest.json` records the
NumPy version and actual hashes. RNG and transcendental implementations may
change across library/platform versions: use the hash test to verify exact
reproduction, and preserve the manifest when using another environment.

The FP16 generator emits the full arithmetic, including both twiddle-specialized
controls. It uses only checked-in host/CMake templates under
`benchmarks/templates`; it has no dependency on an older private folder. Calling
`generate(output, variants)` or its CLI writes to the specified output directory.
Importing the modules writes no inputs or sources. `SOURCE_PROVENANCE.json` records checksums of the supplied kernel sources.
Hosts provide portable configuration, timing metadata and argument/file checks.

## Build locally on an Ascend machine

Install a compatible CANN toolkit, compiler and CMake 3.18+. Set the toolkit root
to the directory containing `compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake`.
The default target is `Ascend310P1`; changing `--soc` does not establish support
or performance on a different device. No CANN headers are bundled.

```bash
export ASCEND_CANN_PACKAGE_PATH=/path/to/ascend-toolkit/latest
python -m benchmarks.build_native --backend npu \
  --source benchmarks/fp16/fused3_opt --build-dir build/npu-fp16-fused3-opt \
  --cann-env /path/to/ascend-toolkit/set_env.sh --soc Ascend310P1
```

`--cann-root` overrides the environment. `ASCEND_HOME_PATH` is also accepted.
`--cann-env` optionally sources a caller-supplied `set_env.sh` in a child Bash;
omit it when the current shell already has the SDK environment. Use `--compiler`
and `--jobs` for local toolchain/build choices. The helper configures/builds only;
it does not open a device. Direct CMake invocation is supported as well:

```bash
cmake -S benchmarks/fp32/sage6 -B build/npu-fp32-sage6 \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH" \
  -DSOC_VERSION=Ascend310P1 -DCMAKE_BUILD_TYPE=Release
cmake --build build/npu-fp32-sage6 --parallel 2
```

## Build locally on a CUDA machine

Use a CUDA toolkit with cuFFT half-precision support and a supported GPU. Architecture `80` is the A100 default,
configurable with `--cuda-arch`; toolkit and graph support remain local build/run
requirements. The bundled source includes `cufftXtMakePlanMany` with half input,
output and execution types rather than a float surrogate.

```bash
python -m benchmarks.build_native --backend cuda \
  --source benchmarks/fp16/cuda --build-dir build/cuda-fp16 --cuda-arch 80
python -m benchmarks.build_native --backend cuda \
  --source benchmarks/fp32/cuda --build-dir build/cuda-fp32 --cuda-arch 80
```

For multi-configuration generators, the executable may be inside `Release/`.

## Explicit hardware execution and output verification

These commands run **locally** on the machine where they are invoked and require
the explicit `--enable-hardware` flag. Select a new output path for every case;
the runner refuses to overwrite prior run artifacts.

```bash
python -m benchmarks.run_native --enable-hardware --backend npu --precision fp16 \
  --binary build/npu-fp16-fused3-opt/sage_fft --source benchmarks/fp16/fused3_opt \
  --inputs build/inputs/fp16/input_1 --output build/runs/npu-fused3-opt-s1-b4.bin \
  --iterations 10000 --blocks 4 --device 0 \
  --cann-env /path/to/ascend-toolkit/set_env.sh

python -m benchmarks.run_native --enable-hardware --backend cuda --precision fp16 \
  --binary build/cuda-fp16/sage_cuda_fft --source benchmarks/fp16/cuda \
  --inputs build/inputs/fp16/input_1 --output build/runs/cuda-fp16-s1-graph100.bin \
  --iterations 10000 --mode graph100 --device 0
```

For FP32 select the corresponding source, binary, input directory and
`--precision fp32`. Native NPU binaries accept
`INPUT_DIR OUTPUT [ITERATIONS=10000] [BLOCKS=4] [DEVICE=0]`; CUDA binaries accept
`INPUT_DIR OUTPUT ITERATIONS MODE [DEVICE=0]`. These low-level binaries execute
hardware directly; the Python runner supplies the opt-in and validation layer.

Each native host saves `OUTPUT` before timing and `OUTPUT.after` after timing.
The runner records stdout/stderr and `OUTPUT.result.json` with input, executable,
source and output hashes, all eleven raw samples, correctness errors and failure
details. Source hashes describe the supplied source directory; the runner does
not cryptographically prove that a caller-provided executable was built from it.
Retain the build manifest alongside the run. A failed numeric check keeps
`accepted_timing=false` and exits unsuccessfully.

The CPU oracle is NumPy complex128 `ifftn(fftn(x) * h)` on the actual quantized
input files. Validation runs on both saved outputs. Standalone verification:

```bash
python -m benchmarks.verify_outputs --precision fp16 \
  --inputs build/inputs/fp16/input_1 --output build/runs/npu-fused3-opt-s1-b4.bin
```

| Precision | Required numerical gate |
| --- | --- |
| FP32 | Every complex error `abs(y-ref) <= 1e-4 + 1e-4*abs(ref)`; finite outputs |
| FP16 | Relative L2 `<= 0.003` **and** every complex error `<= 0.003 + 0.003*abs(ref)`; finite outputs |

For a zero reference, relative L2 is zero only for an exactly zero result;
otherwise the FP16 relative gate fails. File sizes must match exactly. FP16
does not use the FP32 gate or claim a `2e-4` hardware requirement.

## Timing and experiment protocol

All native hosts run one pipeline to save the first output, then 1000 warmup
pipelines and a synchronization outside timing. Each of eleven groups times
`N` complete pipelines followed by one stream synchronization using
`std::chrono::steady_clock`; results are microseconds **per pipeline**. The
method label is `host_batch_submit_sync`. It includes host submission and final
device completion, and excludes initialization, allocation, transfers and
warmup. Empty `submit_us`/`drain_us` arrays retained in the CUDA diagnostic output
are not separate measurements.

CUDA supports `direct`, `graph1`, `graph100` and an `empty` diagnostic control.
Graph capture, instantiation, and one graph warmup launch happen outside timing;
the extra graph warmup covers 1 or 100 pipelines and is separately recorded.
`graph100` requires `N` divisible by 100. The graph contains repeated pipelines
with the same immutable input. The empty control is checked against the input
and always has `accepted_timing=false`; it is not an FFT result. Direct mode is
the FP32 long-loop protocol. Graph modes are separate CUDA diagnostics.

The benchmark matrix uses these phases:

1. FP32: seeds 1–3, N in `{1,10,100,1000,10000}`, four NPU blocks and CUDA direct.
   The stability check requires aggregate per-call medians at N=1000
   and N=10000 to differ by at most 2%; investigate a failure.
2. FP16 tuning: seed 1, N=1000, all six NPU variants at 1/4/8 blocks and CUDA
   direct/graph1/graph100. Use deterministic shuffled case order (seed 439).
3. FP16 confirmation: seeds 1–3, N=10000, all six variants at matched four and
   eight blocks; retain all eleven samples per case. Select CUDA mode from its
   aggregate result, keeping mode selection explicit.
4. FP16 held-out correctness: seeds 4–5, N=1 and four blocks on NPU;
   N=100 with graph100 on CUDA. The hosts retain the same warmup/group protocol.

Run one case at a time with unique output names; the helper intentionally does
not schedule concurrent device jobs. CPU tests evaluate plan ordering with an
independent radix-2 implementation, check generated/bundled source and input
hashes, and test precision/metadata gates. They do not simulate hardware timing,
prove half arithmetic on a device, or constitute a CANN/CUDA compilation test.
