# GitHub Codespace

The Codespace configuration for GERSite is in `.github/copilot-setup-steps.yml`.
It pre-installs `uv` and all pipeline dependencies so the environment is ready
immediately after the Codespace starts.

## Starting a Codespace

Open the repo on GitHub → **Code → Codespaces → Create codespace on main**.

Once the setup steps complete, all `just` recipes are available immediately:

```bash
just run saipan     # smallest AOI — good smoke test
just test           # run the test suite
just lint           # flake8 + pylint
```

## Running Tests

```bash
just test           # uv run pytest tests/ -v
```

## Running Flows as Notebooks

Marimo notebooks can be opened from inside the Codespace:

```bash
just nb-ingest      # opens flows/ingest_sources.py in Marimo
```

## Included VS Code Extensions

The Codespace ships with the extensions defined in `.vscode/extensions.json`,
which typically include Python, Pylance, and the Marimo extension.

## Notes

- The `storage.root` path in `config.gers.yaml` points to a local data directory.
  In a Codespace, update this to a path under `/workspaces/` or mount cloud
  storage for large Bronze-layer ingests.
- FEMA and NSI downloads are several GB per AOI. Use `saipan` or `guam` for
  fast end-to-end tests in a Codespace.
