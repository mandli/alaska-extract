#!/usr/bin/env python3
"""
wrf_regrid_uvp_to_latlon.py

Regrid extracted WRF U/V/P (curvilinear XLAT/XLONG) to a regular lat/lon grid using xESMF,
then fill/relax missing regions to nominal background values with a smooth taper.

Design goals:
- Keep WRF-like units: U/V in m s-1, P in Pa (background specified in hPa).
- Handle Alaska/Pacific dateline robustly via a "continuous longitude" convention around a median center.
- Avoid "moving streaks" in movies by using time-dependent NaN validity during fill/relax.
- Produce diagnostics (single frame and movie) that contour the original missing mask so you can verify behavior.

Requires (regrid env):
  numpy, xarray, netcdf4, scipy, xesmf, esmpy, matplotlib, ffmpeg (only for movies)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import xesmf as xe
from scipy.ndimage import distance_transform_edt

from alaska_extract.version import VERSION
from alaska_extract.plotting import movie_from_frames

from alaska_extract.common import (
    decode_wrf_times,
    find_ffmpeg,
    infer_lon_center,
    lon_continuous_about,
    sanitize_netcdf_attrs,
    vprint,
)


# =============================================================================
# Grid helpers
# =============================================================================

def _make_1d_axis(lo: float, hi: float, d: float) -> np.ndarray:
    if d <= 0:
        raise ValueError("Resolution must be > 0")
    if hi < lo:
        lo, hi = hi, lo
    n = int(np.floor((hi - lo) / d + 0.5)) + 1
    return lo + d * np.arange(n, dtype=np.float64)


def _make_target_grid(lon0: float, lon1: float, lat0: float, lat1: float, dlon: float, dlat: float) -> xr.Dataset:
    lon_t = _make_1d_axis(lon0, lon1, dlon)
    lat_t = _make_1d_axis(lat0, lat1, dlat)
    return xr.Dataset(coords={"lon": ("lon", lon_t), "lat": ("lat", lat_t)})


def _infer_bbox_from_file(ds: xr.Dataset, pad_deg: float = 0.0) -> Tuple[float, float, float, float, float]:
    if "XLAT" not in ds or "XLONG" not in ds:
        raise KeyError("Input must contain XLAT/XLONG. Re-run extractor with --keep-geo.")
    lat = ds["XLAT"].values.astype(np.float64)
    lon = ds["XLONG"].values.astype(np.float64)
    lon_center = infer_lon_center(lon)
    lon_c = lon_continuous_about(lon, center_deg=lon_center)

    lat_min = max(-90.0, float(np.nanmin(lat)) - pad_deg)
    lat_max = min(90.0, float(np.nanmax(lat)) + pad_deg)
    lon_min = float(np.nanmin(lon_c)) - pad_deg
    lon_max = float(np.nanmax(lon_c)) + pad_deg
    return lon_min, lon_max, lat_min, lat_max, lon_center


def _decode_time_for_title(ds: xr.Dataset, t_index: int) -> str:
    if "Times" in ds:
        try:
            v = ds["Times"].isel(Time=t_index).values
            # netCDF char arrays sometimes come through as bytes or S1 arrays
            if hasattr(v, "tobytes"):
                return decode_wrf_times(v.tobytes())
            return decode_wrf_times(v)
        except Exception:
            pass
    return ""


# =============================================================================
# Optional metrics
# =============================================================================

def add_latlon_metrics(ds_ll: xr.Dataset, dlon_deg: float, dlat_deg: float, R: float) -> xr.Dataset:
    lat = ds_ll["lat"].astype(np.float64)
    lon = ds_ll["lon"].astype(np.float64)
    dlon_rad = np.deg2rad(float(dlon_deg))
    dlat_rad = np.deg2rad(float(dlat_deg))

    dx_lat = (R * np.cos(np.deg2rad(lat)) * dlon_rad).astype(np.float64)  # (lat,)
    dy = float(R * dlat_rad)

    dx2 = xr.DataArray(
        dx_lat.values[:, None] * np.ones((lat.size, lon.size), dtype=np.float64),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="dx_m",
        attrs={"long_name": "Grid spacing in x-direction", "units": "m"},
    )
    dy2 = xr.DataArray(
        dy * np.ones((lat.size, lon.size), dtype=np.float64),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="dy_m",
        attrs={"long_name": "Grid spacing in y-direction", "units": "m"},
    )
    area2 = xr.DataArray(
        dx2.values * dy2.values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="cell_area_m2",
        attrs={"long_name": "Approximate grid cell area", "units": "m2"},
    )

    ds_ll["dx_m"] = dx2
    ds_ll["dy_m"] = dy2
    ds_ll["cell_area_m2"] = area2
    ds_ll.attrs["metrics_assumption"] = "Spherical Earth; dx=R*cos(lat)*dlon, dy=R*dlat"
    ds_ll.attrs["earth_radius_m"] = float(R)
    return ds_ll


# =============================================================================
# Fill + relax
# =============================================================================

def inpaint_nearest_2d(field2d: np.ndarray, valid2d: np.ndarray) -> np.ndarray:
    field2d = np.asarray(field2d, dtype=np.float64)
    valid2d = np.asarray(valid2d, dtype=bool)
    if not np.any(valid2d):
        return field2d
    invalid = ~valid2d
    _, (iy, ix) = distance_transform_edt(invalid, return_indices=True)
    filled = field2d.copy()
    filled[invalid] = field2d[iy[invalid], ix[invalid]]
    return filled


def relax_invalid_to_bg(
    filled2d: np.ndarray,
    valid2d: np.ndarray,
    *,
    dlat_deg: float,
    dlon_deg: float,
    radius_deg: float,
    bg_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Relax ONLY invalid region (where ~valid2d) toward bg_value with cosine taper
    vs. distance to nearest valid cell.

    Returns: relaxed2d, weight2d, dist_deg
    """
    filled2d = np.asarray(filled2d, dtype=np.float64)
    valid2d = np.asarray(valid2d, dtype=bool)
    invalid = ~valid2d

    dist_pix = distance_transform_edt(invalid).astype(np.float64)
    dist_deg = dist_pix * np.sqrt(abs(dlat_deg) * abs(dlon_deg))

    if radius_deg <= 0:
        out = filled2d.copy()
        out[invalid] = bg_value
        w = np.ones_like(out, dtype=np.float64)
        w[invalid] = 0.0
        return out, w, dist_deg

    r = float(radius_deg)
    x = np.clip(dist_deg / r, 0.0, 1.0)
    w = 0.5 * (1.0 + np.cos(np.pi * x))  # 1 at boundary, 0 at radius
    w[(invalid) & (dist_deg >= r)] = 0.0

    out = filled2d.copy()
    out[invalid] = w[invalid] * filled2d[invalid] + (1.0 - w[invalid]) * bg_value
    return out, w, dist_deg


def fill_and_relax(
    ds: xr.Dataset,
    *,
    dlat_deg: float,
    dlon_deg: float,
    radius_deg: float,
    p_background_hpa: float,
    coverage_threshold: float,
    use_domain_edge: bool,
    use_nan_edge: bool,
    do_fill: bool,
    do_relax: bool,
    verbose: bool,
) -> xr.Dataset:
    """
    Build reference masks at Time=0 for diagnostics/contours, but apply time-dependent
    NaN validity during fill/relax to avoid time-varying holes ("streaks").
    """
    if "coverage" not in ds:
        raise KeyError("Expected ds['coverage'] to exist before calling fill_and_relax")

    ds = ds.copy()
    p_bg_pa = float(p_background_hpa) * 100.0
    cov = ds["coverage"].values
    cov_valid = np.ones_like(cov, dtype=bool)
    if use_domain_edge:
        cov_valid = np.isfinite(cov) & (cov > coverage_threshold)

    # Reference validity from pressure at t=0 (for contours)
    p0 = ds["P"].isel(Time=0).values
    p_valid0 = np.isfinite(p0) if use_nan_edge else np.ones_like(p0, dtype=bool)
    valid_ref = cov_valid & p_valid0

    ds["orig_valid_mask"] = xr.DataArray(
        valid_ref.astype(np.int8),
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        attrs={"long_name": "Original valid-data mask (P at Time=0)", "units": "1"},
    )
    ds["orig_invalid_mask"] = xr.DataArray(
        (~valid_ref).astype(np.int8),
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        attrs={"long_name": "Original invalid-data mask (missing before fill/relax)", "units": "1"},
    )

    # Union missing mask for contouring (footprint + any NaNs in U/V/P at Time=0)
    miss = ~cov_valid
    if use_nan_edge:
        for vv in ("U", "V", "P"):
            if vv in ds:
                miss |= ~np.isfinite(ds[vv].isel(Time=0).values)

    ds["original_missing_mask"] = xr.DataArray(
        miss.astype(np.int8),
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        attrs={"long_name": "Union missing mask at Time=0 (footprint + NaNs)", "units": "1"},
    )

    # Distance-to-valid (based on valid_ref, for diagnostics only)
    invalid_ref = ~valid_ref
    dist_pix = distance_transform_edt(invalid_ref).astype(np.float64)
    dist_deg = dist_pix * np.sqrt(abs(dlat_deg) * abs(dlon_deg))
    ds["dist_to_valid_deg"] = xr.DataArray(
        dist_deg,
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        attrs={"long_name": "Approx distance to nearest valid cell (deg)", "units": "deg"},
    )

    blend_weight = None

    for var in [v for v in ("U", "V", "P") if v in ds]:
        bg = 0.0 if var in ("U", "V") else p_bg_pa
        da = ds[var]
        out = da.copy()

        for t in range(out.sizes["Time"]):
            f = out.isel(Time=t).values.astype(np.float64)

            if use_nan_edge:
                valid_t = cov_valid & np.isfinite(f)
            else:
                valid_t = cov_valid

            if verbose and t == 0:
                vprint(verbose, f"[fill+relax] {var}: valid_ref_frac={float(np.mean(valid_ref)):.3f} "
                                f"valid_t0_frac={float(np.mean(valid_t)):.3f}")

            f_filled = inpaint_nearest_2d(f, valid_t) if do_fill else f.copy()

            if do_relax:
                f_relaxed, w, _ = relax_invalid_to_bg(
                    f_filled,
                    valid_t,
                    dlat_deg=dlat_deg,
                    dlon_deg=dlon_deg,
                    radius_deg=radius_deg,
                    bg_value=bg,
                )
            else:
                f_relaxed = f_filled
                w = np.ones_like(f_relaxed, dtype=np.float64)
                w[~valid_t] = 0.0
                if not do_fill:
                    f_relaxed[~valid_t] = bg

            out.values[t, :, :] = f_relaxed
            if blend_weight is None:
                blend_weight = w

        ds[var] = out

    if blend_weight is None:
        blend_weight = np.ones((ds.sizes["lat"], ds.sizes["lon"]), dtype=np.float64)

    ds["blend_weight"] = xr.DataArray(
        blend_weight,
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        attrs={"long_name": "Relaxation weight (1 at boundary, 0 beyond radius)", "units": "1"},
    )

    ds.attrs["background_pressure_hPa"] = float(p_background_hpa)
    ds.attrs["background_pressure_Pa"] = float(p_bg_pa)
    ds.attrs["blend_radius_deg"] = float(radius_deg)
    ds.attrs["blend_use_domain_edge"] = int(bool(use_domain_edge))
    ds.attrs["blend_use_nan_edge"] = int(bool(use_nan_edge))
    ds.attrs["blend_coverage_threshold"] = float(coverage_threshold)
    ds.attrs["fill_missing"] = int(bool(do_fill))
    ds.attrs["relax_filled"] = int(bool(do_relax))
    ds.attrs = sanitize_netcdf_attrs(ds.attrs)
    return ds


# =============================================================================
# Plotting / diagnostics
# =============================================================================

def _matplotlib_setup(show: bool) -> None:
    if not show:
        import matplotlib
        matplotlib.use("Agg")


def quicklook_plot(ds_ll: xr.Dataset, kind: str, t_index: int, out_png: Optional[str], show: bool) -> None:
    _matplotlib_setup(show)
    import matplotlib.pyplot as plt

    lon = ds_ll["lon"].values
    lat = ds_ll["lat"].values
    Lon, Lat = np.meshgrid(lon, lat)

    def get2d(name: str):
        da = ds_ll[name]
        return da.isel(Time=t_index).values if "Time" in da.dims else da.values

    U = get2d("U")
    V = get2d("V")
    P = get2d("P")
    spd = np.sqrt(U * U + V * V)

    title_time = _decode_time_for_title(ds_ll, t_index)

    if kind == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        m0 = axes[0].pcolormesh(Lon, Lat, np.ma.masked_invalid(spd), shading="auto")
        fig.colorbar(m0, ax=axes[0]).set_label("Wind speed (m/s)")
        axes[0].set_title("Wind speed")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        m1 = axes[1].pcolormesh(Lon, Lat, np.ma.masked_invalid(P), shading="auto")
        fig.colorbar(m1, ax=axes[1]).set_label("Pressure (Pa)")
        axes[1].set_title("Pressure")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")

        if title_time:
            fig.suptitle(title_time)
    else:
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        if kind == "speed":
            m = ax.pcolormesh(Lon, Lat, np.ma.masked_invalid(spd), shading="auto")
            fig.colorbar(m, ax=ax).set_label("Wind speed (m/s)")
            ax.set_title("Wind speed")
        elif kind == "pressure":
            m = ax.pcolormesh(Lon, Lat, np.ma.masked_invalid(P), shading="auto")
            fig.colorbar(m, ax=ax).set_label("Pressure (Pa)")
            ax.set_title("Pressure")
        else:
            raise ValueError(kind)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if title_time:
            ax.set_title(f"{ax.get_title()} | {title_time}")

    if show:
        plt.show()
    else:
        if out_png is None:
            out_png = "regridded_quicklook.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote plot: {out_png}")


def diagnostic_plot(
    ds_ll: xr.Dataset,
    t_index: int,
    out_png: str,
    show: bool,
    *,
    use_anom: bool = False,
    include_coverage: bool = False,
    include_distance: bool = True,
    contour_original_missing: bool = True,
    fixed_scale: bool = True,
    pmin: float = 1.0,
    pmax: float = 99.0,
) -> None:
    """
    Diagnostic panel plot:
      - wind speed
      - pressure or pressure anomaly
      - optional coverage
      - optional distance-to-valid
      - blend weight

    Anchored color scales by default (fixed_scale=True).
    """
    _matplotlib_setup(show)
    import matplotlib.pyplot as plt

    lon = ds_ll["lon"].values
    lat = ds_ll["lat"].values
    Lon, Lat = np.meshgrid(lon, lat)

    U = ds_ll["U"].isel(Time=t_index).values
    V = ds_ll["V"].isel(Time=t_index).values
    P = ds_ll["P"].isel(Time=t_index).values
    spd = np.sqrt(U * U + V * V)

    w = ds_ll["blend_weight"].values if "blend_weight" in ds_ll else None
    cov = ds_ll["coverage"].values if include_coverage and "coverage" in ds_ll else None
    dist = ds_ll["dist_to_valid_deg"].values if include_distance and "dist_to_valid_deg" in ds_ll else None

    vm = ds_ll["orig_valid_mask"].values.astype(float) if "orig_valid_mask" in ds_ll else None
    om = ds_ll["original_missing_mask"].values.astype(float) if (contour_original_missing and "original_missing_mask" in ds_ll) else None

    Pbg = float(ds_ll.attrs.get("background_pressure_Pa", 101325.0))
    title_time = _decode_time_for_title(ds_ll, t_index)

    # Fixed scales computed per-field over all times (robust for movies + single frames)
    def _range_for(arr3d: np.ndarray):
        a = arr3d[np.isfinite(arr3d)]
        if a.size == 0:
            return 0.0, 1.0
        return float(np.percentile(a, pmin)), float(np.percentile(a, pmax))

    if fixed_scale:
        spd_vmin, spd_vmax = _range_for(np.sqrt(ds_ll["U"].values**2 + ds_ll["V"].values**2))
        if use_anom:
            p_vmin, p_vmax = _range_for(ds_ll["P"].values - Pbg)
        else:
            p_vmin, p_vmax = _range_for(ds_ll["P"].values)
    else:
        spd_vmin = spd_vmax = p_vmin = p_vmax = None

    panels = []
    panels.append(("Wind speed", spd, "Wind speed (m/s)", {"vmin": spd_vmin, "vmax": spd_vmax}))
    if use_anom:
        panels.append(("Pressure anomaly", P - Pbg, "P - Pbg (Pa)", {"vmin": p_vmin, "vmax": p_vmax}))
    else:
        panels.append(("Pressure", P, "Pressure (Pa)", {"vmin": p_vmin, "vmax": p_vmax}))

    if cov is not None:
        panels.append(("Coverage", cov, "Coverage (1≈mapped)", {"vmin": 0.0, "vmax": 1.0}))
    if dist is not None:
        panels.append(("Dist to valid", dist, "Distance (deg)", {}))
    if w is not None:
        panels.append(("Blend weight", w, "Blend weight", {"vmin": 0.0, "vmax": 1.0}))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (title, arr, cblabel, opts) in zip(axes, panels):
        marr = np.ma.masked_invalid(np.asarray(arr, dtype=np.float64))
        # drop None vmin/vmax entries
        opts = {k: v for k, v in opts.items() if v is not None}
        m = ax.pcolormesh(Lon, Lat, marr, shading="auto", **opts)
        cb = fig.colorbar(m, ax=ax)
        cb.set_label(cblabel)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if vm is not None:
            ax.contour(Lon, Lat, vm, levels=[0.5], linewidths=0.9, alpha=0.9)
        if om is not None:
            ax.contour(Lon, Lat, om, levels=[0.5], linewidths=1.1, alpha=0.9, linestyles="--")

    if title_time:
        fig.suptitle(title_time)

    if show:
        plt.show()
    else:
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote diagnostic plot: {out_png}")


def diag_movie(
    ds_ll: xr.Dataset,
    out_mp4: str,
    *,
    fps: int = 12,
    every: int = 1,
    frames_dir: Optional[str] = None,
    ffmpeg: str = "ffmpeg",
    keep_frames: bool = False,
    use_anom: bool = False,
    include_coverage: bool = False,
    include_distance: bool = True,
    contour_original_missing: bool = True,
    fixed_scale: bool = True,
    pmin: float = 1.0,
    pmax: float = 99.0,
    verbose: bool = False,
) -> None:
    out_mp4 = str(out_mp4)
    tsize = int(ds_ll.sizes.get("Time", 1))
    every = max(1, int(every))
    frames_path = Path(frames_dir) if frames_dir is not None else Path("_diag_frames")
    frames_path.mkdir(parents=True, exist_ok=True)

    vprint(verbose, f"[diag-movie] Time={tsize}, every={every}, fps={fps}, frames_dir={frames_path}")

    frame_idx = 0
    for t in range(0, tsize, every):
        png = frames_path / f"frame_{frame_idx:05d}.png"
        diagnostic_plot(
            ds_ll,
            t_index=t,
            out_png=str(png),
            show=False,
            use_anom=use_anom,
            include_coverage=include_coverage,
            include_distance=include_distance,
            contour_original_missing=contour_original_missing,
            fixed_scale=fixed_scale,
            pmin=pmin,
            pmax=pmax,
        )
        frame_idx += 1

    # Assemble MP4 from frames
    movie_from_frames(frames_path, Path(out_mp4), fps=int(fps), ffmpeg=ffmpeg, verbose=bool(verbose),
                      keep_frames=bool(keep_frames))


    if not keep_frames:
        shutil.rmtree(frames_path, ignore_errors=True)
        vprint(verbose, f"[diag-movie] Removed frames_dir={frames_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regrid extracted WRF U/V/P to regular lat/lon using xESMF.")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    p.add_argument("input", help="Input NetCDF (from extractor; must contain XLAT/XLONG + U/V/P).")
    p.add_argument("-o", "--output", required=True, help="Output NetCDF (regular lat/lon).")
    p.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    p.add_argument("--bbox", nargs=4, type=float, metavar=("LON0", "LON1", "LAT0", "LAT1"),
                   default=None, help="Target bounds. If omitted, inferred from input.")
    p.add_argument("--infer-pad", type=float, default=0.0, help="Padding added around inferred bbox (deg).")
    p.add_argument("--dlon", type=float, required=True, help="Target lon resolution (deg).")
    p.add_argument("--dlat", type=float, required=True, help="Target lat resolution (deg).")

    p.add_argument("--method", choices=["bilinear", "nearest_s2d"], default="bilinear")
    p.add_argument("--reuse-weights", action="store_true", help="Reuse <output>.weights.nc if present.")

    p.add_argument("--blend-radius-deg", type=float, default=2.0)
    p.add_argument("--p-background", type=float, default=1013.25, help="Background pressure (hPa).")
    p.add_argument("--coverage-threshold", type=float, default=0.99)

    p.add_argument("--no-domain-edge", action="store_true", help="Don't use coverage footprint in validity.")
    p.add_argument("--no-nan-edge", action="store_true", help="Don't use per-time NaNs in validity.")

    p.add_argument("--fill-missing", action="store_true", help="Enable nearest-neighbor fill (default ON).")
    p.add_argument("--no-fill-missing", action="store_true", help="Disable nearest-neighbor fill.")
    p.add_argument("--relax-filled", action="store_true", help="Enable relaxation (default ON).")
    p.add_argument("--no-relax-filled", action="store_true", help="Disable relaxation.")

    p.add_argument("--add-metrics", action="store_true")
    p.add_argument("--earth-radius-m", type=float, default=6371000.0)

    # plotting / diagnostics
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-time", type=int, default=0)
    p.add_argument("--plot-kind", choices=["speed", "pressure", "both"], default="both")
    p.add_argument("--plot-out", default=None)
    p.add_argument("--plot-show", action="store_true")

    p.add_argument("--diag", action="store_true")
    p.add_argument("--diag-out", default=None)
    p.add_argument("--diag-show", action="store_true")
    p.add_argument("--diag-anom", action="store_true")
    p.add_argument("--diag-include-coverage", action="store_true")
    p.add_argument("--diag-no-distance", action="store_true")
    p.add_argument("--diag-no-orig-contour", action="store_true")
    p.add_argument("--diag-fixed-scale", action="store_true", default=True,
                   help="Anchor colorbars (default ON).")
    p.add_argument("--no-diag-fixed-scale", dest="diag_fixed_scale", action="store_false")

    p.add_argument("--diag-pmin", type=float, default=1.0)
    p.add_argument("--diag-pmax", type=float, default=99.0)

    p.add_argument("--diag-movie", action="store_true")
    p.add_argument("--diag-movie-out", default=None)
    p.add_argument("--diag-movie-fps", type=int, default=12)
    p.add_argument("--diag-movie-every", type=int, default=1)
    p.add_argument("--diag-movie-frames-dir", default=None)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--ffmpeg", default="ffmpeg")

    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.input).expanduser()
    out = Path(args.output).expanduser()

    do_regrid = True
    if out.exists() and not args.force:
        do_regrid = False
        vprint(args.verbose, f"[main] output exists, skipping regrid: {out}")

    if do_regrid:
        ds = xr.open_dataset(inp)

        if args.bbox is None:
            lon0, lon1, lat0, lat1, lon_center = _infer_bbox_from_file(ds, pad_deg=float(args.infer_pad))
        else:
            LON0, LON1, LAT0, LAT1 = args.bbox
            lon_center = infer_lon_center(ds["XLONG"].values)
            lon0 = float(lon_continuous_about(np.array(LON0), center_deg=lon_center)) - float(args.infer_pad)
            lon1 = float(lon_continuous_about(np.array(LON1), center_deg=lon_center)) + float(args.infer_pad)
            lat0 = max(-90.0, min(LAT0, LAT1) - float(args.infer_pad))
            lat1 = min(90.0, max(LAT0, LAT1) + float(args.infer_pad))
            if lon1 < lon0:
                lon0, lon1 = lon1, lon0

        vprint(args.verbose, f"[bbox] lon=[{lon0:.3f},{lon1:.3f}] lat=[{lat0:.3f},{lat1:.3f}] center≈{lon_center:.2f}")

        ds_tgt = _make_target_grid(lon0, lon1, lat0, lat1, float(args.dlon), float(args.dlat))

        ds_src = ds.assign_coords(
            lon=(("south_north", "west_east"), lon_continuous_about(ds["XLONG"].values, center_deg=lon_center)),
            lat=(("south_north", "west_east"), ds["XLAT"].values),
        )

        vars_to_regrid = [v for v in ("U", "V", "P") if v in ds_src.data_vars]
        if not vars_to_regrid:
            raise KeyError("No U/V/P found in input.")
        ds_in = ds_src[vars_to_regrid]

        weights_path = str(out.with_suffix(".weights.nc")) if args.reuse_weights else None
        reuse = bool(weights_path and Path(weights_path).exists())

        regridder = xe.Regridder(
            ds_in, ds_tgt, args.method,
            periodic=False,
            unmapped_to_nan=True,
            reuse_weights=reuse,
            filename=weights_path,
        )

        ds_out = regridder(ds_in)

        ones_src = xr.ones_like(ds_in["P"].isel(Time=0), dtype=np.float64)
        cov = regridder(ones_src)
        ds_out["coverage"] = cov.assign_attrs({"long_name": "Regridding coverage (1≈mapped, 0≈unmapped)", "units": "1"})

        # carry Times/XTIME if present
        for v in ("Times", "XTIME"):
            if v in ds:
                ds_out[v] = ds[v]

        if "U" in ds_out:
            ds_out["U"].attrs.setdefault("units", "m s-1")
        if "V" in ds_out:
            ds_out["V"].attrs.setdefault("units", "m s-1")
        if "P" in ds_out:
            ds_out["P"].attrs.setdefault("units", "Pa")

        ds_out.attrs.update(ds.attrs)
        ds_out.attrs["regrid_method"] = args.method
        ds_out.attrs["regrid_lon_center_deg"] = float(lon_center)
        ds_out.attrs["regrid_bbox_continuous"] = f"{lon0},{lon1},{lat0},{lat1}"
        ds_out.attrs["regrid_dlon_dlat"] = f"{args.dlon},{args.dlat}"
        ds_out.attrs["source_file"] = str(inp)

        # defaults ON unless explicitly disabled
        do_fill = True
        do_relax = True
        if args.no_fill_missing:
            do_fill = False
        if args.no_relax_filled:
            do_relax = False
        if args.fill_missing:
            do_fill = True
        if args.relax_filled:
            do_relax = True

        ds_out = fill_and_relax(
            ds_out,
            dlat_deg=float(args.dlat),
            dlon_deg=float(args.dlon),
            radius_deg=float(args.blend_radius_deg),
            p_background_hpa=float(args.p_background),
            coverage_threshold=float(args.coverage_threshold),
            use_domain_edge=not args.no_domain_edge,
            use_nan_edge=not args.no_nan_edge,
            do_fill=do_fill,
            do_relax=do_relax,
            verbose=bool(args.verbose),
        )

        if args.add_metrics:
            ds_out = add_latlon_metrics(ds_out, float(args.dlon), float(args.dlat), float(args.earth_radius_m))

        ds_out.attrs = sanitize_netcdf_attrs(ds_out.attrs)

        if out.exists() and args.force:
            out.unlink()

        encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
        ds_out.to_netcdf(out, encoding=encoding)
        print(f"Wrote: {out}")
        if weights_path:
            print(f"Weights: {weights_path}")

    # Always open for plotting/diagnostics
    ds_plot = xr.open_dataset(out)

    if args.plot:
        plot_out = args.plot_out
        if plot_out is None and not args.plot_show:
            plot_out = str(out.with_suffix(f".{args.plot_kind}.png"))
        quicklook_plot(ds_plot, args.plot_kind, args.plot_time, plot_out, args.plot_show)

    if args.diag:
        diag_out = args.diag_out
        if diag_out is None and not args.diag_show:
            diag_out = str(out.with_suffix(".diag.png"))
        diagnostic_plot(
            ds_plot,
            t_index=args.plot_time,
            out_png=diag_out,
            show=args.diag_show,
            use_anom=args.diag_anom,
            include_coverage=args.diag_include_coverage,
            include_distance=not args.diag_no_distance,
            contour_original_missing=not args.diag_no_orig_contour,
            fixed_scale=bool(args.diag_fixed_scale),
            pmin=float(args.diag_pmin),
            pmax=float(args.diag_pmax),
        )

    if args.diag_movie:
        movie_out = args.diag_movie_out or str(out.with_suffix(".diag.mp4"))
        frames_dir = args.diag_movie_frames_dir or (str(out.with_suffix("")) + "_diag_frames")
        diag_movie(
            ds_plot,
            out_mp4=movie_out,
            fps=int(args.diag_movie_fps),
            every=int(args.diag_movie_every),
            frames_dir=frames_dir,
            ffmpeg=str(args.ffmpeg),
            keep_frames=bool(args.keep_frames),
            use_anom=bool(args.diag_anom),
            include_coverage=bool(args.diag_include_coverage),
            include_distance=not args.diag_no_distance,
            contour_original_missing=not args.diag_no_orig_contour,
            fixed_scale=bool(args.diag_fixed_scale),
            pmin=float(args.diag_pmin),
            pmax=float(args.diag_pmax),
            verbose=bool(args.verbose),
        )


if __name__ == "__main__":
    main()
