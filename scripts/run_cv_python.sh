#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="."
cd "$PROJECT_ROOT"

conda run -n cv python "$@"
