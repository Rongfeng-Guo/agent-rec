#!/usr/bin/env bash
set -euo pipefail

python --version
which python
nvidia-smi || true
git status --short
git branch --show-current
git log --oneline --decorate -n 5
