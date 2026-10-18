#!/usr/bin/env bash
# Dev runner for alx.
#
#   ./run.sh                 interactive REPL
#   ./run.sh "some prompt"   one-shot: send the prompt, print the reply, exit
#
# Always uses the GitHub Copilot provider, through the installed `copilot`
# CLI and its login. Settings (alx.toml, env) still apply for everything else,
# e.g. ALX_MODEL. Use `cargo run` directly for other providers.
set -euo pipefail

# Run from the repo root so cargo and .env are found from anywhere.
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -gt 0 ]]; then
	printf '%s\nexit\n' "$*" | cargo run --quiet --bin alx
else
	exec cargo run --quiet --bin alx -- --provider copilot
fi
