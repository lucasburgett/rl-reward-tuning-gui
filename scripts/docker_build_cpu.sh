#!/usr/bin/env bash
set -euo pipefail
docker build -f docker/CPU.Dockerfile -t rl-template-cpu:latest .