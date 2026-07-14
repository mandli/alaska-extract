# Changelog

## Unreleased
- Added `--slp` to extract sea-level pressure (wrf-python hydrostatic reduction)
  as the pressure field; recommended for storm-surge (GeoClaw) forcing over
  high terrain, where `--psfc` (surface pressure) yields sub-realistic values.
  `--slp`/`--psfc` are mutually exclusive; recorded surface pressure source is
  now also reflected in `wrf_extract_use_slp`.
- Clarified `--psfc` help/README: it is surface pressure at terrain elevation,
  not sea-level pressure.

## 0.2.0
- Shared plotting/movie helpers (lazy matplotlib imports)
- Added --version flags
- Added ruff/black/pre-commit and CI lint step
