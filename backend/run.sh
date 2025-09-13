#!/usr/bin/env bash
set -euo pipefail

# Create directories if missing
mkdir -p ../models
mkdir -p ../data

# Run the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000