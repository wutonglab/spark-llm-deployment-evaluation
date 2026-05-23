#!/usr/bin/env bash
# Mirror frameworks/ and case-studies/ from the repo root into src/ so mdbook
# can find them. The root copies stay so GitHub renders them directly when
# someone browses the repo on github.com.
#
# Everything else under src/ (SUMMARY.md, README.md, methodology.md,
# glossary.md, hardware-portability/) is a first-class source and is NOT
# touched by this script.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

rm -rf src/frameworks src/case-studies
mkdir -p src/frameworks src/case-studies

cp -R frameworks/. src/frameworks/
cp -R case-studies/. src/case-studies/

echo "src/frameworks and src/case-studies refreshed from repo root."
