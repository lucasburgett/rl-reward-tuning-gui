#!/usr/bin/env bash
set -euo pipefail
docker run --rm -it -v "$PWD:/app" rl-template-cpu:latest "$@"