# Indexed FFT catalog study

M1 contains 384 realizations of a 64-point FFT with batch four and paired
forward/inverse masks. M2 contains 6,144 realizations on 8 by 8. M3 contains
147,456 realizations on 8³. M2 and M3 use independent axis masks, epilogues,
core allocation and strided-line packing. All factorizations are radix-2.

The catalog compiles contiguous stage groups and epilogues. Validate every
candidate output and save source/binary identities, launch plans and raw timings.
Allocation, transfers, model calls and plan loading are outside pipeline timing.
Exhaustive controls use randomized order and independent remeasurement rounds.

M2/M3 search uses seeds 41, 42 and 43 and 12 proposals per condition: greedy,
random, evolution, indexed LLM, typed LLM without history, typed LLM with
history, and permuted-history control. Invalid proposals consume budget.
Current state accepts any new correct realization; the returned incumbent is
the fastest verified state. Record model and device costs separately.
