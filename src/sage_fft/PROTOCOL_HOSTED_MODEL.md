# Hosted model comparison

Use a configured OpenAI-compatible DeepSeek-V3.2 service with temperature 0.4,
512 output tokens, thinking disabled and JSON-object output. The M3 workload
is 8³. Five conditions use seeds 41, 42 and 43 with 12 proposals each: indexed
diagnosis, typed diagnosis, typed diagnosis without history, typed diagnosis
with related M2 prior, and analytic greedy. This produces 180 proposals.
Invalid responses consume budget. Save responses before target evaluation so
an interrupted device measurement can resume without requesting a new decision.
Remeasure visited candidates in five randomized blocks on input seeds 2 and 3.
Report acceptance, latency, model time and tokens from the current run.
