# Documentation sources

The published documentation starts at `index.rst`. Keep this file out of the
Sphinx toctree; it is only a contributor note for people editing the docs.

Narrative pages may be written in Markdown or reStructuredText. API reference
pages are generated from public docstrings and explicit public objects, so do
not add private helper dumps to `api.rst`.

Install the package and documentation dependencies from the repository root:

```bash
python -m pip install -e ".[docs]"
```

Build locally:

```bash
python -m sphinx -b html docs docs/_build/html
```

Run the warning-as-error build used for release checks:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Do not duplicate the root package README here. The root README should remain a
concise project overview; detailed explanations belong in the documentation
pages under `docs/`.
