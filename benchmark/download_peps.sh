#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/benchmark_data/peps"
mkdir -p "$TARGET_DIR"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Clone repo without full history
git clone --filter=blob:none --no-checkout https://github.com/python/peps.git "$TMP_DIR"
cd "$TMP_DIR"

git sparse-checkout init --cone
git sparse-checkout set peps
git checkout main

# Move only pep-XXXX.rst files from peps/ and subdirs into target
find peps -maxdepth 2 -type f -regextype posix-extended -regex ".*/pep-[0-9]{4}\.rst$" -exec mv {} "$TARGET_DIR/" \;

# Remove everything else in target
find "$TARGET_DIR" -type f ! -regextype posix-extended -regex ".*/pep-[0-9]{4}\.rst$" -delete
find "$TARGET_DIR" -mindepth 1 -type d -exec rm -rf {} +

echo "Downloaded only pep-XXXX.rst files into $TARGET_DIR"
