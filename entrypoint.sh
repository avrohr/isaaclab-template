#!/bin/sh
# if the code dir exists, install in editable mode
if [ -f "pyproject.toml" ]; then
  uv pip install -e .
fi
# then hand off to uv run
exec uv run "$@"