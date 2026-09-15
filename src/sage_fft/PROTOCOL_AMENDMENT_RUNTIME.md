# Local model runtime

Local controls use DeepSeek-R1-Distill-Llama-8B Q4_K_M through Ollama with
6,144 context tokens, temperature 0.4 and a 192-token output budget. Configure
the installed model explicitly. Both action interfaces request required output
fields. Compiler guards independently check legality and reject duplicates.
The persistent target process keeps initialization outside pipeline timing.
Model inference and output validation are also outside pipeline timing.
