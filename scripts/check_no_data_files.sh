#!/usr/bin/env bash
# Reject staged data files outside tests/*/data/ and docs/.
# Pre-commit passes candidate file paths as arguments.
set -euo pipefail
status=0
for f in "$@"; do
  case "$f" in
    tests/*/data/*|docs/*)
      # allowed
      ;;
    *)
      echo "ERROR: data file outside allowed paths: $f"
      status=1
      ;;
  esac
done
exit $status
