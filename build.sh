#!/usr/bin/env bash

set -euo pipefail

cd $(dirname $0)

# Create a virtual environment to run our code
VENV_NAME=".venv-build"

export PATH=$PATH:$HOME/.local/bin
source $VENV_NAME/bin/activate

uv run pyinstaller --onefile -p src src/main.py
tar -czvf dist/archive.tar.gz ./dist/main meta.json
