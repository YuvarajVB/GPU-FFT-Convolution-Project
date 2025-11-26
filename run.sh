#!/usr/bin/env bash
set -e
echo "[1] Building..."
make build
echo "[2] Running (data/input.png -> output/output.png)..."
mkdir -p output
./bin/fft_filter data/input.png output/output.png
echo "[3] Done. Output: output/output.png"
