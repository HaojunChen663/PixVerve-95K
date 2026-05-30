#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ADAPTER_TYPE=lora
export ATTENTION=global
exec "${SCRIPT_DIR}/infer.sh" "$@"
