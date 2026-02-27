# alaska-extract

Two small command-line tools for working with WRF outputs:

1. `wrf_extract_uvp_at_z.py` (pixi env: `wrf`)
   - Takes WRF NetCDF files (files/dirs/globs)
   - Extracts **U/V (m s-1)** and **pressure (Pa)** at a target geometric height
   - Concatenates across Time into a compact NetCDF
   - Optional quicklook plots + movie

2. `wrf_regrid_uvp_to_latlon.py` (pixi env: `regrid`)
   - Takes the extracted file (must include XLAT/XLONG)
   - Regrids to a **regular lat/lon** grid using xESMF
   - Fills missing regions by nearest-neighbor from valid data (coast → inland)
   - Relaxes **only the filled region** toward background values with a smooth taper
   - Diagnostics plots + diagnostic movie (anchored colorbars by default)

## Install (pixi)

```bash
pixi install
pixi run -e wrf check-wrf
pixi run -e regrid check-esmf
```

## Extract

```bash
pixi run -e wrf extract -- \
  /path/to/h01_files/*.nc \
  -o uvp.nc --keep-geo --verbose \
  --z 10 --z-ref agl
```

> If `--` gives you trouble in zsh, remove it; it’s only needed if your task definition includes it.

## Regrid

```bash
pixi run -e regrid regrid -- \
  uvp.nc -o uvp_ll.nc --dlon 0.05 --dlat 0.05 --force \
  --diag --diag-anom --diag-include-coverage \
  --diag-movie --diag-movie-fps 12
```

## Outputs / conventions

- `U`, `V` in **m s-1**
- `P` in **Pa**
- Background pressure CLI is **hPa** (`--p-background 1013.25`)

## Testing

```bash
pixi run -e regrid test
```

## Developer tools

Format + lint:

```bash
pixi run -e regrid format
pixi run -e regrid lint
```

Pre-commit:

```bash
pixi run -e regrid precommit-install
```

Both scripts support `--version`.
