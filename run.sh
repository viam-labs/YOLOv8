#!/bin/bash

set -euo pipefail

cd $(dirname $0)

VENV_NAME="${VIAM_MODULE_DATA}/venv"
PYTHON="$VENV_NAME/bin/python"

# uncomment for hot reloading
# export PATH=$PATH:$HOME/.local/bin
# uv venv $VENV_NAME
# source $VENV_NAME/bin/activate

uv pip install -r requirements.txt
# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
echo "Starting module..."
exec $PYTHON -m src $@
