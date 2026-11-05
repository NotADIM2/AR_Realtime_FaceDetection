#!/usr/bin/env sh
# Run on Linux or macOS. The first run creates .venv and installs the dependencies.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "Setting up AR Face Filters. This only happens once and needs internet access."
    python3 -m venv .venv
    .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
fi

exec .venv/bin/python main.py "$@"
