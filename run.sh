#!/usr/bin/env bash
# Dev runner for alx.
#
#   ./run.sh                 interactive REPL
#   ./run.sh "some prompt"   one-shot: send the prompt, print the reply, exit
#
# Always uses the GitHub Copilot provider, through the installed `copilot`
# CLI and its login. Env: ALX_MODEL (optional), COPILOT_CLI_PATH and
# COPILOT_HOME (optional). Use `cargo run` directly for other providers.
set -euo pipefail

export ALX_PROVIDER=copilot

# Run from the repo root so cargo and .env are found from anywhere.
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -gt 0 ]]; then
	printf '%s\nexit\n' "$*" | cargo run --quiet --bin alx
else
	exec cargo run --quiet --bin alx
fi
