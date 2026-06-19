# Release checklist

- [ ] Confirm `LICENSE` exists and matches `pyproject.toml` metadata.
- [ ] Run `python -m pip install -e ".[test]"`.
- [ ] Run `python -m pytest`.
- [ ] Run the synthetic examples.
- [ ] Run the benchmark smoke script.
- [ ] Confirm optional SpatialData tests skip or pass.
- [ ] Confirm optional PyDESeq2 tests skip or pass.
- [ ] Confirm the README quickstart works from a clean environment.
- [ ] Update `__version__`.
- [ ] Tag the release.
