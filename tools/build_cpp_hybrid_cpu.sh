#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/cpp/ecsp_cpp_solver_hybrid.cpp"
OUT="${1:-$ROOT/build/ecsp_cpp_solver_hybrid}"
mkdir -p "$(dirname "$OUT")"
if [[ -n "${CXX:-}" ]]; then
  COMPILER="$CXX"
elif command -v clang++ >/dev/null 2>&1; then
  COMPILER=clang++
elif command -v g++ >/dev/null 2>&1; then
  COMPILER=g++
else
  echo "ERROR: no C++17 compiler found (clang++ or g++)." >&2
  exit 2
fi
FLAGS=(-O3 -std=c++17 -DNDEBUG -Wall -Wextra -Wpedantic)
if [[ "$(uname -s)" == "Darwin" ]]; then
  FLAGS+=(-mcpu=native)
elif [[ "$(uname -m)" == "x86_64" || "$(uname -m)" == "aarch64" ]]; then
  FLAGS+=(-march=native)
fi
if ! "$COMPILER" "${FLAGS[@]}" "$SRC" -o "$OUT"; then
  echo "Native tuning flag failed; retrying portable C++17 build." >&2
  FLAGS=(-O3 -std=c++17 -DNDEBUG -Wall -Wextra -Wpedantic)
  "$COMPILER" "${FLAGS[@]}" "$SRC" -o "$OUT"
fi
"$OUT" --version
printf 'Built %s with %s\n' "$OUT" "$COMPILER"
