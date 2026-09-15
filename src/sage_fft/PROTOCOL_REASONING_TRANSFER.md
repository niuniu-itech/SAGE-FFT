# Diagnosis and transfer controls

M4 is an 8 by 16, batch-one pipeline. Nine conditions use seeds 41, 42 and 43
with 12 proposals each: direct typed, reasoned typed, reasoned indexed,
typed related prior, typed unrelated prior, indexed related prior, greedy,
random and evolution. LLM conditions allow 512 output tokens and use a 6,144
token context. Diagnoses are limited to 500 characters.

Related context contains the four fastest M2 greedy seed-41 records from 12
measurements. Unrelated context contains four records from 12 M1 configurations
covering staged/grouped boundaries, three core counts and both epilogues.
Context must not include M4 target answers. Record source-data acquisition cost.
Remeasure selected and visited candidates on input seeds 2 and 3. A best-observed
reference is not a global optimum unless the entire space is measured.
