#!/bin/sh
cd $(dirname $0)

# Create a virtual environment to run our code
VENV_NAME=".venv-build"

export PATH=$PATH:$HOME/.local/bin

if ! uv pip install pyinstaller -Uq; then
  exit 1
fi

uv run pyinstaller --onefile -p src src/main.py
tar -czvf dist/archive.tar.gz ./dist/main meta.json
