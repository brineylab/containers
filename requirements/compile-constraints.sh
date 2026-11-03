#!/usr/bin/env bash
# Resolves every image's pip requirements together into constraints.txt, so shared packages get one
# version everywhere. Existing pins are kept; pass --upgrade (or --upgrade-package NAME) to move them.
set -euo pipefail
cd "$(dirname "$0")"
uv pip compile pip.txt datascience_pip.txt ai-ml_pip.txt jupyter_pip.txt \
    --python-version 3.12 --python-platform x86_64-manylinux_2_39 \
    --annotation-style line --custom-compile-command requirements/compile-constraints.sh \
    -o constraints.txt "$@"
