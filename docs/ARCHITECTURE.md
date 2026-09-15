# Implementation architecture

SAGE reconstructs a registered cuFFT pipeline as a target FFT graph, exposes legal boundary and schedule changes, then measures the resulting NPU plans. Source registration, numerical validation and measured performance remain separate stages.

## Contract and transformation space

`fft_ir.py` defines the supported contract: one to three power-of-two axes, positive batch size, interleaved FP32 complex values, contiguous row-major layout, natural frequency bins and explicit inverse normalization. `cuda_source.py` emits the registered source family. `import_cuda` checks that family; it is not a general CUDA parser.

`mask_ir.py` exposes three transformation levels:

| Level | Representation | Effect |
| --- | --- | --- |
| Pipeline | Independent multiply and scale fusion flags | Attach an epilogue to completed forward or inverse values |
| FFT | Forward and inverse masks for every axis | Split or merge adjacent radix-2 stage groups |
| Schedule | AI Core block count and line-packing limit | Change work allocation and packed lines |

A mask bit of one materializes a stage result; zero keeps that boundary inside a kernel. Butterfly order and twiddle signs remain fixed. Kernels buffer a complete axis, so changing masks alone does not reduce UB capacity. Packing affects occupancy and parallel groups. Guards check supported shapes, action parameters, ownership and resource limits.

The `fft_ir.py` action dictionary uses one grouping factor and a shared epilogue flag for the uniform-group controls. The indexed catalog uses independent masks and epilogue flags; its action index belongs to the current menu and is distinct from the uniform-group configuration ID.

## Code generation and execution

`migration.compile_cuda` connects the registered source adapter to realization construction and IR-driven lowering. `migration.search_cuda` connects the source contract to Algorithm 1 and exports the returned best plan.

`lower.py` emits Ascend C kernels, a host and CMake configuration. `catalog.py` builds the workload-specific set of stage-group and epilogue kernels. A realization becomes an ordered list of catalog kernel IDs. `live_target.py` keeps a target process available during search and returns actual outputs and timing samples for each submitted plan.

Catalog execution avoids compiling a new binary for every proposal. The static controls in `final_calibration.py` separately build selected realizations and compare their outputs with catalog composition. They provide evidence for those checked cases, rather than a proof about every supported toolchain.

## Model policy and state updates

`structured_search.py` constructs a legal menu from the current graph and filters visited realizations. A model receives the graph configuration, analytical resource features and the policy's visible history. In the action-index interface it selects `j`; the compiler resolves `menu[j]` to a typed transformation and applies the same legality checks used by other policies.

The file retains analytic greedy, random and evolutionary controls, direct-field emission, missing-history and permuted-history controls. `transfer_study.py` adds brief diagnoses and optional measured records from other workloads. Such records provide context; their latency is not a measurement of the new workload, and model weights do not change.

A correct unseen candidate becomes the current state even when it is slower. The incumbent changes only when measured latency improves. Invalid proposals consume the proposal budget and remain in the logs. The final result is the fastest verified incumbent. A generated diagnosis is not validated performance evidence.

Candidate-specific lowering and numerical errors are logged while search continues. Shared compilation failures, broken SSH connections and exited target processes propagate. Both fresh and cached target records must pass correctness. The initial state is checked before the loop, and `run_search` returns the best verified realization, including on resumed runs.

`llm.py` adapts requests to OpenAI-compatible or Ollama chat endpoints. Authentication is added to transport headers, not the saved request object. `config.py` reads runtime configuration and opens SSH connections only when an explicit hardware driver requests one.

## Separate native FP16 extension

`benchmarks/fp16` implements the fixed `8×8×8` weighted pipeline with native half vector operations, central forward/multiply/inverse fusion and outer plane fusion. Its grouped and fused variants include matched twiddle-specialization controls. These kernels are not actions in the indexed FP32 catalog, and their speedup does not measure model-policy effectiveness.

`benchmarks/fp32` retains the scalar staged and grouped controls. See [NATIVE_BENCHMARKS.md](NATIVE_BENCHMARKS.md) for build commands, numerical gates and the common host-submit timing protocol. Optional native builds and device runs were not newly established by the portable CPU test suite.

## Evidence and entry points

`sage` exposes offline validation, inspection, input generation and source emission. Explicit study modules write raw outputs under the configured work directory; analysis modules derive summaries from those records. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the required execution order.
