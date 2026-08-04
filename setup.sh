#!/usr/bin/env bash

set -euo pipefail

cd $(dirname $0)

# Create a virtual environment to run our code
VENV_NAME=".venv-build"
PYTHON="$VENV_NAME/bin/python"
ENV_ERROR="This module requires Python >=3.8, pip, and virtualenv to be installed."

export PATH=$PATH:$HOME/.local/bin

if [ ! "$(command -v uv)" ]; then
  if [ ! "$(command -v curl)" ]; then
    echo "curl is required to install UV. please install curl on this system to continue."
    exit 1
  fi
  echo "Installing uv command"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! uv venv $VENV_NAME; then
  echo "unable to create required virtual environment"
  exit 1
fi

source $VENV_NAME/bin/activate

if ! uv pip install -r requirements.txt; then
  echo "unable to sync requirements to venv"
  exit 1
fi

# ultralytics declares a hard dependency on opencv-python, the GUI build, which
# links Qt and so needs libGL, libX11, libxcb and friends present on whatever
# machine runs the module. This module never opens a window, so swap in the
# headless wheel: identical cv2 API and version, no Qt linkage, and nothing for
# an operator to install on the host. It has to happen after the requirements
# install, because that is what drags opencv-python in.
#
# The uninstall is tolerated if it finds nothing, so this keeps working if a
# future ultralytics stops depending on opencv-python.
echo "Replacing opencv-python with opencv-python-headless"
uv pip uninstall opencv-python || true
if ! uv pip install opencv-python-headless; then
  echo "unable to install opencv-python-headless"
  exit 1
fi

if ! uv pip install pyinstaller -Uq; then
  exit 1
fi
