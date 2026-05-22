# Releasing to PyPI

Releases are published by `.github/workflows/release.yml`, triggered manually
from the GitHub Actions UI. It uses **PyPI Trusted Publishing (OIDC)** — there
is no API token or secret stored in the repository.

## One-time setup (you must do this once)

1. Create the project on PyPI (or reserve the name) — the first upload can also
   create it, but the trusted publisher must be configured first.
2. On PyPI: **Account → Publishing → Add a pending publisher** (or, if the
   project exists, **Project → Settings → Publishing**) with:
   - PyPI Project Name: `agentic-conversations-hdf5`
   - Owner: `mattjala`
   - Repository: `agentic-conversations-hdf5`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
3. In the GitHub repo: **Settings → Environments → New environment** named
   `pypi` (optionally add required reviewers to gate publishes).

## Cutting a release

The base version lives in `pyproject.toml` under `[project] version`. Bump it
and commit *before* dispatching the workflow.

1. Edit `pyproject.toml` to the new base version (e.g. `0.1.0` → `0.2.0`),
   commit, and push to `main`.
2. Go to **Actions → Release → Run workflow** and pick:
   - **release** — publishes the base version as-is (tag `v0.2.0`).
   - **pre-release** — auto-numbers as `v{base}pre{N}`, where `N` is the count
     of existing `v{base}pre*` tags (so the first pre-release for `0.2.0` is
     `v0.2.0pre0`, the next is `v0.2.0pre1`, and so on). Marked as a
     pre-release on GitHub.

The workflow rewrites `pyproject.toml` in-flight with the resolved version,
builds the sdist + wheel, publishes to PyPI, creates the git tag, and creates
the GitHub Release with auto-generated notes.

## Verify a build locally before dispatching

```bash
python -m build
twine check dist/*
# optional: install the wheel in a clean venv and smoke-test
python -m venv /tmp/relcheck && /tmp/relcheck/bin/pip install dist/*.whl
/tmp/relcheck/bin/agentic-conversations-hdf5 --help
```

## TestPyPI dry run (optional)

To rehearse without touching real PyPI, add a second trusted publisher on
test.pypi.org and a step using `pypa/gh-action-pypi-publish` with
`repository-url: https://test.pypi.org/legacy/`.
