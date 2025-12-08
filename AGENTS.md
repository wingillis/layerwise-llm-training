All python code must be run with `uv run`. For example, `uv run script.py`.

Whenever a python file is run that exists in a non-parent directory, like `tests/`, you must run it as a module, like `uv run python -m tests.test_materialization_simple` or `uv run python -m pytest tests/test_materialization_simple.py`.