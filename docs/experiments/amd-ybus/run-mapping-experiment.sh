#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
cargo run --offline --example inspect_amd_mapping -- docs/experiments/amd-ybus/mapping-results
