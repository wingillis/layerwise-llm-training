All python code must be run with `uv run`. For example, `uv run script.py`.

Whenever a python file is run that exists in a non-parent directory, like `tests/`, you must run it as a module, like `uv run python -m tests.test_materialization_simple` or `uv run python -m pytest tests/test_materialization_simple.py`.

## Code quality check

After performing a batch of edits, use `ruff` to check the code quality and fix them. If `ruff` cannot fix them, fix them manually.
Example: `uv run ruff check /path/to/code`.

Use `ruff` to format code as well. Example: `uv run ruff format /path/to/code`.