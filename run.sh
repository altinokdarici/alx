#!/usr/bin/env bash
# Dev runner for alx.
#
#   ./run.sh                 interactive REPL
#   ./run.sh "some prompt"   one-shot: send the prompt, print the reply, exit
#
# Env: CLAUDE_API_KEY (or .env in the repo root), ALX_MODEL (optional).
set -euo pipefail

# Run from the repo root so cargo and .env are found from anywhere.
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -gt 0 ]]; then
	printf '%s\nexit\n' "$*" | cargo run --quiet --bin alx
else
	exec cargo run --quiet --bin alx
fi
