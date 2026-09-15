# Security and sensitive information

The portable CPU suite needs no credentials. Runtime model and hardware configuration comes from environment variables; `.env.example` is documentation, not an automatically loaded secret file. Keep real keys, passwords, private endpoints and personal paths outside version control.

Model transport adds authentication to HTTP headers and omits it from saved request metadata. Transport errors retain an error type instead of provider error bodies. Prompts and successful responses can still contain information supplied by the caller or provider, so inspect experiment logs before sharing them. Use HTTPS for remote model endpoints and do not place credentials in a URL.

SSH clients use existing known-host keys and reject unknown hosts. Verify a new server through your normal administrative channel before adding its key. Remote study drivers execute compiler and benchmark commands on the configured machine; native binaries execute on the machine where they are launched. Neither path is a sandbox for untrusted code or model-generated programs.

No private security contact or supported-release policy has been configured yet. The project owner should establish one before publication. Do not put credentials, exploitable private endpoints or sensitive datasets into a public issue. A useful private report contains the affected revision, a minimal reproduction with synthetic values, and the expected and observed behavior.
