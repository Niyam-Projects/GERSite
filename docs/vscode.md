# Visual Studio Code for GERSite Development

## Recommended Extensions

- **Python** (`ms-python.python`) — linting, debugging, IntelliSense
- **Pylance** (`ms-python.vscode-pylance`) — fast type-checking
- **Pylint** (`ms-python.pylint`) — inline lint feedback (config in `pyproject.toml`)
- **Even Better TOML** (`tamasfe.even-better-toml`) — syntax highlighting for `pyproject.toml` and `config.gers.yaml`
- **DuckDB SQL** — syntax highlighting for DuckDB SQL embedded in flows

## Python Interpreter

Select the `uv`-managed virtual environment:

1. Open the command palette (`Ctrl+Shift+P`)
2. Run **Python: Select Interpreter**
3. Choose the interpreter at `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux/macOS)

## Pylint Configuration

Pylint settings are in `pyproject.toml` under `[tool.pylint.*]`. VS Code's
Pylint extension picks these up automatically — no separate `.pylintrc` needed.

## Running Flows from the Terminal

The integrated terminal supports all `just` recipes:

```powershell
just run saipan         # full pipeline for Saipan
just nb-ingest          # open Flow 1 as a Marimo notebook
just test               # run the test suite
just lint               # flake8 + pylint
```

## Debugging a Flow

1. Open the flow file (e.g., `flows/ingest_sources.py`) in the editor.
2. Add breakpoints as needed.
3. In the Run and Debug panel, launch with the Python debugger.

The flows add `lib/` to `sys.path` automatically, so `duckdb_helpers`,
`scoring`, `occupancy`, and `spatial_utils` are importable without any
extra `PYTHONPATH` configuration.
