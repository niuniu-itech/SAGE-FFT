# Contributing

Keep changes focused on an observable behavior, numerical property or reproducible result. Describe the problem, the change and the checks that establish it. Project licensing and author details remain pending; see [LICENSING.md](LICENSING.md).

## Development checks

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests benchmarks scripts
python -m build
python scripts/check_repository.py
```

Default tests must remain CPU-only. Mock model and SSH transport, use synthetic secrets and avoid import-time filesystem or network work. A source-generation test must not require a vendor SDK. Changes to native hosts or kernels also need a separately recorded build and device run before claiming hardware validation.

For semantic changes, test forward and inverse transforms as well as the weighted pipeline: paired errors can cancel in a round trip. Preserve dtype-specific error gates, input immutability and dependency order. For search changes, retain invalid proposals, model time and baseline budgets; do not silently change the candidate space between policies.

## Results and documentation

A benchmark result should carry its own input, source and binary hashes, raw samples, environment, model configuration and numerical checks. Keep bulky outputs outside version control and document how to reproduce them.

Update the relevant command examples when changing CLI arguments or experiment prerequisites. Keep the indexed FP32 catalog and fixed-shape FP16 extension distinct in code and claims. Report CPU, native-build, device and hosted-model validation separately.

Do not commit keys, passwords, private endpoints or account-specific paths. Use environment variables and redact logs before sharing a minimal reproduction. See [SECURITY.md](SECURITY.md) for handling sensitive reports.
