# Author: even
"""Compare staged and grouped FP32 plans using the independent CPU oracle."""

import json

from sage_fft.fft_ir import Contract, execute, metrics, stages
from sage_fft.validate_semantics import inputs, reference


def main():
    contract = Contract((8, 8, 8), 1)
    x, weights = inputs(contract, 1)
    expected = reference(contract, x, weights)
    for name, action in (
        ("staged", {"group": 1, "cores": 4, "fusion": False}),
        ("grouped", {"group": 16, "cores": 4, "fusion": True}),
    ):
        result = metrics(execute(contract, action, x, weights), expected)
        print(json.dumps({"variant": name, "launches": len(stages(contract, action)), **result}))
        if not result["pass_correctness"]:
            raise SystemExit("CPU semantic validation failed")


if __name__ == "__main__":
    main()
