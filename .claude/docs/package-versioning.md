# Package versioning

Semantic-version bumps of the `openpois` Python package + Vue site + Sphinx docs. **This is distinct from dataset versioning** (`versions:` block in `config.yaml`) — see [data-versioning.md](data-versioning.md) for that.

## Files to update on every bump

Five files declare the package version. All must move together.

| File | Field | Notes |
|---|---|---|
| [src/openpois/__init__.py](../../src/openpois/__init__.py) | `__version__` | Canonical source for `import openpois; openpois.__version__`. |
| [pyproject.toml](../../pyproject.toml) | `version` | Static string under `[project]`. We do **not** use `dynamic = ["version"]` — the prior setup had no resolver and would have failed a fresh build. |
| [pyproject.toml](../../pyproject.toml) | `Development Status` classifier | Bump when crossing thresholds: `3 - Alpha` → `4 - Beta` → `5 - Production/Stable`. Don't bump on every release. |
| [docs/conf.py](../../docs/conf.py) | `release` | Sphinx version surfaced in the rendered docs at openpois.org/docs/. |
| [site/package.json](../../site/package.json) | `"version"` | Vue site at openpois.org. Largely cosmetic but keep in lock-step. |
| [CITATION.cff](../../CITATION.cff) | `version`, `date-released` | GitHub renders both in the "Cite this repository" panel. `date-released` should match the tag date. |

## Conventions

- **Semver**: patch for bug fixes, minor for backwards-compatible features, major for breaking API changes. Crossing 1.0.0 signals a stable public API.
- **Tag prefix**: `vX.Y.Z` (e.g., `v1.0.0`).
- **Annotated tags only** (`git tag -a`, never lightweight). GitHub Releases requires a tag object; lightweight tags don't carry a message.
- **Date format in CITATION.cff**: ISO `YYYY-MM-DD`, quoted.

## Workflow

```bash
# 1. Edit the five files above. Sanity-check:
git diff -- src/openpois/__init__.py pyproject.toml docs/conf.py \
            site/package.json CITATION.cff

# 2. Verify install + tests still pass.
pytest -q

# 3. Commit with a single-purpose message ("Bump to vX.Y.Z, …").

# 4. Tag annotated, then push commit + tag together.
git tag -a vX.Y.Z -m "OpenPOIs X.Y.Z — <release theme>"
git push origin main vX.Y.Z

# 5. Verify the tag on remote.
git ls-remote --tags origin vX.Y.Z
```

## Verifying a tag is correctly placed

```bash
git for-each-ref --format='%(refname:short) %(objectname:short) -> %(*objectname:short) %(*subject)' refs/tags/vX.Y.Z
```

The first SHA is the annotated-tag *object* (its own hash); the second is the *commit* it points to. Both differ on purpose — that's how annotated tags work. The tag object SHA is what `git ls-remote --tags` reports.

## After tagging

GitHub auto-renders the tag under **Releases** as a "tag-only" entry. To promote it to a full **Release** (release notes, asset attachments, RSS feed), open `https://github.com/henryspatialanalysis/openpois/releases/tag/vX.Y.Z` and click *Create release from tag*. This is manual and per-release.

## What does NOT need to bump

- `versions:` block in `config.yaml` — that's data versioning, on its own monthly cadence.
- `cff-version: 1.2.0` in `CITATION.cff` — that's the CFF spec version, not the package version. Leave alone.
- Conda env name in `environment.yml` — env name is `openpois`, not version-suffixed.

## Triggers for a major bump (1.0 → 2.0)

- Breaking API change in `openpois.*` public modules (rename, remove, signature change without back-compat shim).
- Schema change in published Source Coop data that breaks existing readers.
- Renaming or restructuring `config.yaml`'s top-level keys (scripts read these by name).

Cosmetic changes, internal refactors, dependency bumps that don't change the public surface, and dataset refreshes are minor or patch.
