# Notebooks — initial exploration

The project's first two steps, dated February 2026: decoding `.fit` files with
the Garmin SDK, then building session features (cardiac decoupling, running
mask, efficiency).

They are kept because they show where the project started. **They are not
runnable as is**: they read two sample `.fit` files that are not versioned,
`data/` being ignored by git and watched by a test fixture. Their imports,
however, still resolve against the current code.

Outputs were stripped. They weighed 1.2 MB of Plotly figures that GitHub's
notebook preview does not render — that weight bought nothing.

What these notebooks did by hand now lives in `src/layer_a_signal/` and runs
with `python -m src.cli ingest`. The notebooks themselves are in French.
