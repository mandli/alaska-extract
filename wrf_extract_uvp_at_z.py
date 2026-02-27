#!/usr/bin/env python3
"""
Concatenate WRF NetCDF files and extract U/V winds + pressure at a target geometric height,
then optionally make quick-look plots and ffmpeg movies.

Output file contains:
  Times(Time, DateStrLen)
  XTIME(Time)
  U(Time, south_north, west_east)  [m/s] at target z
  V(Time, south_north, west_east)  [m/s] at target z
  P(Time, south_north, west_east)  [Pa ] at target z
Optionally: XLAT, XLONG, HGT (static) via --keep-geo

Plotting/Movie:
  --plot with matplotlib quicklooks, --movie creates MP4 via ffmpeg
  --plot-domain crops by indices (I0 I1 J0 J1)
  --plot-bbox crops by lon/lat bounds (LON0 LON1 LAT0 LAT1) using XLONG/XLAT

Robustness:
  - If requested z-level yields all-NaN from interplevel(), can fall back to
    (zmin + dz) where zmin is the global minimum of the chosen z_field.
  - Plotting masks NaNs to avoid colorbar failures over land/below-ground regions.

Requires:
  - numpy
  - netCDF4
  - wrf-python
  - matplotlib (for plotting/movie frames)
  - ffmpeg (optional, only for --movie)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from netCDF4 import Dataset
from wrf import ALL_TIMES, getvar, interplevel


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract U/V and pressure from WRF at a given z-level and concatenate over Time."
    )
    p.add_argument("files", nargs="+", help="Input WRF NetCDF files, in time order.")
    p.add_argument("-o", "--output", required=True, help="Output NetCDF filename.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists. If not set and output exists, extraction is skipped "
             "(but plot/movie can still run).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (prints diagnostics like z-range and NaN fractions, plus extra details).",
    )

    p.add_argument("--z", type=float, default=0.0, help="Target height in meters. Default: 0 (MSL).")
    p.add_argument(
        "--z-ref",
        choices=["msl", "agl"],
        default="msl",
        help="Interpret --z as 'msl' (mean sea level) or 'agl' (above ground level). Default: msl.",
    )

    # Robust z selection / fallback
    p.add_argument(
        "--z-auto",
        action="store_true",
        help="Ignore --z and instead use (zmin + dz) where zmin is the minimum of the chosen z_field "
             "for each file. Use --z-auto-dz to set dz.",
    )
    p.add_argument(
        "--z-auto-dz",
        type=float,
        default=10.0,
        help="dz in meters for --z-auto or fallback. Default: 10.",
    )
    p.add_argument(
        "--no-z-fallback",
        action="store_true",
        help="Disable automatic fallback if interplevel returns all-NaN at requested z.",
    )

    p.add_argument("--compression", type=int, default=4, help="NetCDF deflate level (0-9). Default: 4.")
    p.add_argument(
        "--keep-geo",
        action="store_true",
        default=True,
        help="Write XLAT/XLONG/HGT to output (recommended; required for --plot-bbox).",
    )

    # Plotting
    p.add_argument("--plot", action="store_true", help="Generate a quick-look plot from the collated file.")
    p.add_argument("--plot-time", type=int, default=0, help="Time index to plot (0-based). Default: 0.")
    p.add_argument(
        "--plot-kind",
        choices=["speed", "pressure", "quiver", "both"],
        default="speed",
        help="What to plot. 'both' makes a 1x2 subplot (speed + pressure). Default: speed.",
    )
    p.add_argument("--plot-out", default=None, help="Output png filename. Default is derived from nc name.")
    p.add_argument("--plot-show", action="store_true", help="Show interactively instead of saving a PNG.")
    p.add_argument("--plot-step", type=int, default=1, help="Subsample every N grid points (>=1). Default: 1.")
    p.add_argument(
        "--plot-domain",
        nargs=4,
        type=int,
        metavar=("I0", "I1", "J0", "J1"),
        default=None,
        help="Crop by indices: I0 I1 J0 J1 (I=west_east, J=south_north), as Python slices [I0:I1], [J0:J1].",
    )
    p.add_argument(
        "--plot-bbox",
        nargs=4,
        type=float,
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
        default=None,
        help="Crop by lon/lat box (requires XLONG/XLAT in output; use --keep-geo).",
    )

    # Movies
    p.add_argument("--movie", action="store_true", help="Make an MP4 movie via ffmpeg from the collated output.")
    p.add_argument(
        "--movie-kind",
        choices=["speed", "pressure", "quiver", "both"],
        default="speed",
        help="Field to animate. Default: speed.",
    )
    p.add_argument("--movie-fps", type=int, default=10, help="Movie FPS. Default: 10.")
    p.add_argument("--movie-every", type=int, default=1, help="Use every Nth timestep (>=1). Default: 1.")
    p.add_argument("--movie-out", default=None, help="Movie filename. Default: <base>_<kind>.mp4")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (name or full path). Default: ffmpeg")
    p.add_argument("--keep-frames", action="store_true", help="Keep intermediate PNG frames.")

    # Movie scaling controls
    p.add_argument(
        "--movie-fixed-scale",
        action="store_true",
        default=True,
        help="Use fixed color scale across frames (default: on).",
    )
    p.add_argument(
        "--no-movie-fixed-scale",
        dest="movie_fixed_scale",
        action="store_false",
        help="Disable fixed movie scaling (per-frame autoscale; may flicker).",
    )
    p.add_argument(
        "--movie-scale",
        choices=["minmax", "percentile"],
        default="minmax",
        help="How to compute fixed scaling. Default: minmax.",
    )
    p.add_argument("--movie-pmin", type=float, default=1.0, help="Lower percentile for percentile scaling. Default: 1.")
    p.add_argument("--movie-pmax", type=float, default=99.0, help="Upper percentile for percentile scaling. Default: 99.")

    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def _ensure_same_hgrid(nc: Dataset, sn: int, we: int) -> None:
    if nc.dimensions["south_north"].size != sn or nc.dimensions["west_east"].size != we:
        raise ValueError(
            f"Horizontal grid mismatch: expected (south_north={sn}, west_east={we}) "
            f"got ({nc.dimensions['south_north'].size}, {nc.dimensions['west_east'].size})"
        )


def _get_static_fields(nc: Dataset) -> dict:
    out = {}
    if "XLAT" in nc.variables:
        out["XLAT"] = nc.variables["XLAT"][0, :, :].astype("f4")
    if "XLONG" in nc.variables:
        out["XLONG"] = nc.variables["XLONG"][0, :, :].astype("f4")
    if "HGT" in nc.variables:
        v = nc.variables["HGT"]
        out["HGT"] = (v[0, :, :] if v.ndim == 3 else v[:, :]).astype("f4")
    return out


def _read_timestr(nc: Dataset, t_index: int) -> str:
    if "Times" not in nc.variables:
        return f"t={t_index}"
    ts = nc.variables["Times"][t_index, :].tobytes().decode("utf-8", errors="ignore").strip()
    return ts if ts else f"t={t_index}"


def _get_xy(nc: Dataset) -> Tuple[np.ndarray, np.ndarray, str, str]:
    has_geo = ("XLAT" in nc.variables) and ("XLONG" in nc.variables)
    if has_geo:
        lat = np.asarray(nc.variables["XLAT"][:, :], dtype=np.float64)
        lon = np.asarray(nc.variables["XLONG"][:, :], dtype=np.float64)

        # Unwrap longitude to avoid antimeridian artifacts in pcolormesh
        lon_u = _unwrap_lon(lon)

        return lon_u, lat, "Longitude (unwrapped)", "Latitude"

    we = nc.dimensions["west_east"].size
    sn = nc.dimensions["south_north"].size
    x = np.arange(we)[None, :].repeat(sn, axis=0)
    y = np.arange(sn)[:, None].repeat(we, axis=1)
    return x, y, "west_east index", "south_north index"


def _domain_slices_from_indices(
    nc: Dataset,
    plot_domain: Optional[List[int]],
    plot_step: int,
) -> Tuple[slice, slice]:
    """Return (south_north slice, west_east slice)."""
    if plot_step < 1:
        raise ValueError("--plot-step must be >= 1")

    sn = nc.dimensions["south_north"].size
    we = nc.dimensions["west_east"].size

    if plot_domain is None:
        return slice(0, sn, plot_step), slice(0, we, plot_step)

    i0, i1, j0, j1 = plot_domain
    if not (0 <= i0 <= we and 0 <= i1 <= we and 0 <= j0 <= sn and 0 <= j1 <= sn):
        raise ValueError(
            f"--plot-domain out of bounds for grid (west_east={we}, south_north={sn}). "
            f"Got I0,I1,J0,J1 = {i0},{i1},{j0},{j1}"
        )
    if i1 <= i0 or j1 <= j0:
        raise ValueError(f"--plot-domain must have I1>I0 and J1>J0. Got {i0},{i1},{j0},{j1}")

    return slice(j0, j1, plot_step), slice(i0, i1, plot_step)


def _bbox_to_index_domain(nc: Dataset, bbox: List[float]) -> List[int]:
    """
    Convert lon/lat bbox to an index-domain [I0,I1,J0,J1] that covers the bbox.

    bbox: [lon0, lon1, lat0, lat1]
    Requires XLONG/XLAT in the collated file.

    Dateline handling:
      - We do bbox logic in [0,360) space.
      - If lon0_360 <= lon1_360: standard range.
      - If lon0_360 > lon1_360: interpret as crossing the dateline, i.e.
            lon in [lon0_360, 360) OR lon in [0, lon1_360].

    Note: This assumes your bbox does not intend to select "the long way around" the globe.
    """
    if "XLAT" not in nc.variables or "XLONG" not in nc.variables:
        raise KeyError("XLAT/XLONG not found; --plot-bbox requires --keep-geo when creating output.")

    lon0, lon1, lat0, lat1 = bbox
    la = min(lat0, lat1)
    lb = max(lat0, lat1)

    lat = np.asarray(nc.variables["XLAT"][:, :], dtype=np.float64)
    lon = np.asarray(nc.variables["XLONG"][:, :], dtype=np.float64)

    lon360 = _lon_to_360(lon)
    lon0_360 = float(_lon_to_360(lon0))
    lon1_360 = float(_lon_to_360(lon1))

    lat_mask = (lat >= la) & (lat <= lb)

    if lon0_360 <= lon1_360:
        lon_mask = (lon360 >= lon0_360) & (lon360 <= lon1_360)
    else:
        # crosses dateline
        lon_mask = (lon360 >= lon0_360) | (lon360 <= lon1_360)

    mask = lat_mask & lon_mask

    if not np.any(mask):
        raise ValueError(
            "No grid points found inside --plot-bbox; check bounds and lon convention. "
            f"(Interpreted lon in [0,360): lon0={lon0_360:.2f}, lon1={lon1_360:.2f}, "
            f"lat=[{la:.2f},{lb:.2f}])"
        )

    jj, ii = np.where(mask)
    j0 = int(jj.min())
    j1 = int(jj.max()) + 1
    i0 = int(ii.min())
    i1 = int(ii.max()) + 1
    return [i0, i1, j0, j1]


def _resolve_plot_domain(
    nc: Dataset,
    plot_domain: Optional[List[int]],
    plot_bbox: Optional[List[float]],
) -> Optional[List[int]]:
    if plot_domain is not None and plot_bbox is not None:
        raise ValueError("Use only one of --plot-domain or --plot-bbox (not both).")
    if plot_bbox is not None:
        return _bbox_to_index_domain(nc, plot_bbox)
    return plot_domain


def _nan_frac(a: np.ndarray) -> float:
    aa = np.asarray(a)
    if aa.size == 0:
        return float("nan")
    return float(np.isnan(aa).mean())



def _interplevel_timewise(field, z_field, z_target):
    """Apply wrf.interplevel in a time-safe way.

    wrf-python's interplevel() can be strict about shape matching and can raise:
        ValueError: arguments 0 and 1 must have the same shape
    when handed 4D inputs that don't align exactly (e.g., time-dependent z_field
    handling, or unexpected squeezing).

    This helper forces 3D-per-time interpolation for 4D inputs and then stacks.
    """
    from wrf import interplevel  # local import keeps module import light

    field_ndim = getattr(field, "ndim", np.asarray(field).ndim)
    z_ndim = getattr(z_field, "ndim", np.asarray(z_field).ndim)

    # 3D case: just do it
    if field_ndim == 3:
        return np.asarray(interplevel(field, z_field, z_target))

    if field_ndim != 4:
        raise ValueError(f"Expected field to be 3D or 4D, got ndim={field_ndim}")

    nt = field.shape[0]
    out = []
    for t in range(nt):
        f3 = field.isel(Time=t) if hasattr(field, "isel") else field[t, ...]
        if z_ndim == 4:
            z3 = z_field.isel(Time=t) if hasattr(z_field, "isel") else z_field[t, ...]
        else:
            z3 = z_field
        out.append(np.asarray(interplevel(f3, z3, z_target)))

    return np.stack(out, axis=0)

def _unwrap_lon(lon_deg: np.ndarray) -> np.ndarray:
    """Unwrap longitude along west_east so there is no antimeridian jump (for plotting)."""
    lon = np.asarray(lon_deg, dtype=np.float64)
    return np.rad2deg(np.unwrap(np.deg2rad(lon), axis=1))


def _lon_to_360(lon_deg: np.ndarray) -> np.ndarray:
    """Convert longitude(s) to [0, 360) for robust bbox masking."""
    lon = np.asarray(lon_deg, dtype=np.float64)
    return np.mod(lon, 360.0)

# -----------------------------
# Plotting
# -----------------------------
def quicklook_plot(
    nc_path: str,
    kind: str,
    t_index: int,
    out_png: Optional[str],
    show: bool,
    plot_step: int = 1,
    plot_domain: Optional[List[int]] = None,
    plot_bbox: Optional[List[float]] = None,
    speed_vmin: Optional[float] = None,
    speed_vmax: Optional[float] = None,
    pres_vmin: Optional[float] = None,
    pres_vmax: Optional[float] = None,
) -> None:
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Dataset(nc_path) as nc:
        nt = nc.dimensions["Time"].size
        if t_index < 0 or t_index >= nt:
            raise IndexError(f"--plot-time {t_index} out of range (Time size = {nt})")

        resolved_domain = _resolve_plot_domain(nc, plot_domain, plot_bbox)

        x, y, xlabel, ylabel = _get_xy(nc)
        jsl, isl = _domain_slices_from_indices(nc, resolved_domain, plot_step)

        xs = x[jsl, isl]
        ys = y[jsl, isl]

        u = nc.variables["U"][t_index, :, :][jsl, isl]
        v = nc.variables["V"][t_index, :, :][jsl, isl]
        p = nc.variables["P"][t_index, :, :][jsl, isl]

        timestr = _read_timestr(nc, t_index)

        # Mask invalids for stable plotting (NaNs over land/below-ground)
        u = np.ma.masked_invalid(u)
        v = np.ma.masked_invalid(v)
        p = np.ma.masked_invalid(p)

        if kind == "both":
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

            spd = np.sqrt(u * u + v * v)
            spd = np.ma.masked_invalid(spd)

            if spd.mask.all():
                raise ValueError("Wind speed field is all-NaN/masked in requested plot region/time.")
            if p.mask.all():
                raise ValueError("Pressure field is all-NaN/masked in requested plot region/time.")

            m0 = axes[0].pcolormesh(xs, ys, spd, shading="auto", vmin=speed_vmin, vmax=speed_vmax)
            fig.colorbar(m0, ax=axes[0]).set_label("Wind speed (m/s)")
            axes[0].set_title("Wind speed")

            m1 = axes[1].pcolormesh(xs, ys, p, shading="auto", vmin=pres_vmin, vmax=pres_vmax)
            fig.colorbar(m1, ax=axes[1]).set_label("Pressure (Pa)")
            axes[1].set_title("Pressure")

            for ax in axes:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            fig.suptitle(timestr)

        else:
            fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

            if kind == "speed":
                spd = np.sqrt(u * u + v * v)
                spd = np.ma.masked_invalid(spd)
                if spd.mask.all():
                    raise ValueError("Wind speed field is all-NaN/masked in requested plot region/time.")
                m = ax.pcolormesh(xs, ys, spd, shading="auto", vmin=speed_vmin, vmax=speed_vmax)
                fig.colorbar(m, ax=ax).set_label("Wind speed (m/s)")
                ax.set_title(f"Wind speed ({timestr})")

            elif kind == "pressure":
                if p.mask.all():
                    raise ValueError("Pressure field is all-NaN/masked in requested plot region/time.")
                m = ax.pcolormesh(xs, ys, p, shading="auto", vmin=pres_vmin, vmax=pres_vmax)
                fig.colorbar(m, ax=ax).set_label("Pressure (Pa)")
                ax.set_title(f"Pressure ({timestr})")

            elif kind == "quiver":
                spd = np.sqrt(u * u + v * v)
                spd = np.ma.masked_invalid(spd)
                if spd.mask.all():
                    raise ValueError("Wind speed field is all-NaN/masked in requested plot region/time.")
                m = ax.pcolormesh(xs, ys, spd, shading="auto", vmin=speed_vmin, vmax=speed_vmax)
                fig.colorbar(m, ax=ax).set_label("Wind speed (m/s)")

                sn2, we2 = u.shape
                step_y = max(1, sn2 // 40)
                step_x = max(1, we2 // 40)

                ax.quiver(
                    xs[::step_y, ::step_x],
                    ys[::step_y, ::step_x],
                    u[::step_y, ::step_x],
                    v[::step_y, ::step_x],
                    angles="xy",
                    scale_units="xy",
                )
                ax.set_title(f"Wind vectors ({timestr})")

            else:
                raise ValueError(f"Unknown plot kind: {kind}")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        if show:
            plt.show()
        else:
            if out_png is None:
                base = os.path.splitext(os.path.basename(nc_path))[0]
                out_png = f"{base}_{kind}_t{t_index:04d}.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"Wrote plot: {out_png}")


# -----------------------------
# Movie scaling + ffmpeg
# -----------------------------
def _compute_movie_limits(
    nc_path: str,
    kind: str,
    every: int,
    plot_step: int,
    plot_domain: Optional[List[int]],
    plot_bbox: Optional[List[float]],
    scale_mode: str,
    pmin: float,
    pmax: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if every < 1:
        raise ValueError("--movie-every must be >= 1")
    if plot_step < 1:
        raise ValueError("--plot-step must be >= 1")

    with Dataset(nc_path) as nc:
        nt = nc.dimensions["Time"].size
        resolved_domain = _resolve_plot_domain(nc, plot_domain, plot_bbox)
        jsl, isl = _domain_slices_from_indices(nc, resolved_domain, plot_step)

    need_speed = kind in ("speed", "quiver", "both")
    need_pres = kind in ("pressure", "both")

    if scale_mode == "minmax":
        spd_min = math.inf
        spd_max = -math.inf
        p_min = math.inf
        p_max = -math.inf

        with Dataset(nc_path) as nc:
            U = nc.variables["U"]
            V = nc.variables["V"]
            P = nc.variables["P"]

            for t in range(0, nt, every):
                if need_speed:
                    u = U[t, :, :][jsl, isl]
                    v = V[t, :, :][jsl, isl]
                    spd = np.sqrt(u * u + v * v)
                    # ignore NaNs
                    if not np.isnan(spd).all():
                        spd_min = min(spd_min, float(np.nanmin(spd)))
                        spd_max = max(spd_max, float(np.nanmax(spd)))

                if need_pres:
                    pp = P[t, :, :][jsl, isl]
                    if not np.isnan(pp).all():
                        p_min = min(p_min, float(np.nanmin(pp)))
                        p_max = max(p_max, float(np.nanmax(pp)))

        speed_vmin = (spd_min if need_speed and spd_min != math.inf else None)
        speed_vmax = (spd_max if need_speed and spd_max != -math.inf else None)
        pres_vmin = (p_min if need_pres and p_min != math.inf else None)
        pres_vmax = (p_max if need_pres and p_max != -math.inf else None)
        return speed_vmin, speed_vmax, pres_vmin, pres_vmax

    if scale_mode == "percentile":
        if not (0.0 <= pmin < pmax <= 100.0):
            raise ValueError("--movie-pmin/--movie-pmax must satisfy 0 <= pmin < pmax <= 100")

        speed_samples: List[np.ndarray] = []
        pres_samples: List[np.ndarray] = []

        with Dataset(nc_path) as nc:
            U = nc.variables["U"]
            V = nc.variables["V"]
            P = nc.variables["P"]

            for t in range(0, nt, every):
                if need_speed:
                    u = U[t, :, :][jsl, isl]
                    v = V[t, :, :][jsl, isl]
                    spd = np.sqrt(u * u + v * v)
                    spd = spd.astype(np.float32, copy=False)
                    spd = spd[np.isfinite(spd)]
                    if spd.size:
                        speed_samples.append(spd)

                if need_pres:
                    pp = P[t, :, :][jsl, isl].astype(np.float32, copy=False)
                    pp = pp[np.isfinite(pp)]
                    if pp.size:
                        pres_samples.append(pp)

        speed_vmin = speed_vmax = None
        pres_vmin = pres_vmax = None

        if need_speed and speed_samples:
            all_spd = np.concatenate(speed_samples)
            speed_vmin = float(np.nanpercentile(all_spd, pmin))
            speed_vmax = float(np.nanpercentile(all_spd, pmax))

        if need_pres and pres_samples:
            all_p = np.concatenate(pres_samples)
            pres_vmin = float(np.nanpercentile(all_p, pmin))
            pres_vmax = float(np.nanpercentile(all_p, pmax))

        return speed_vmin, speed_vmax, pres_vmin, pres_vmax

    raise ValueError(f"Unknown scale mode: {scale_mode}")


def make_movie(
    nc_path: str,
    kind: str,
    fps: int,
    every: int,
    out_mp4: Optional[str],
    ffmpeg_exe: str,
    keep_frames: bool,
    verbose: bool,
    plot_step: int = 1,
    plot_domain: Optional[List[int]] = None,
    plot_bbox: Optional[List[float]] = None,
    fixed_scale: bool = True,
    scale_mode: str = "minmax",
    pmin: float = 1.0,
    pmax: float = 99.0,
) -> None:
    if every < 1:
        raise ValueError("--movie-every must be >= 1")
    if fps < 1:
        raise ValueError("--movie-fps must be >= 1")
    if plot_step < 1:
        raise ValueError("--plot-step must be >= 1")

    if shutil.which(ffmpeg_exe) is None and not Path(ffmpeg_exe).exists():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg_exe}")

    with Dataset(nc_path) as nc:
        nt = nc.dimensions["Time"].size

    base = os.path.splitext(os.path.basename(nc_path))[0]
    if out_mp4 is None:
        out_mp4 = f"{base}_{kind}.mp4"

    speed_vmin = speed_vmax = None
    pres_vmin = pres_vmax = None

    if fixed_scale and kind in ("speed", "pressure", "quiver", "both"):
        speed_vmin, speed_vmax, pres_vmin, pres_vmax = _compute_movie_limits(
            nc_path=nc_path,
            kind=kind,
            every=every,
            plot_step=plot_step,
            plot_domain=plot_domain,
            plot_bbox=plot_bbox,
            scale_mode=scale_mode,
            pmin=pmin,
            pmax=pmax,
        )
        if verbose:
            print(
                "Movie fixed scale:",
                f"speed=({speed_vmin}, {speed_vmax})",
                f"pressure=({pres_vmin}, {pres_vmax})",
                f"mode={scale_mode}",
            )

    tmpdir = Path(tempfile.mkdtemp(prefix="wrf_frames_"))
    pattern = tmpdir / "frame_%06d.png"

    # Render frames
    frame_idx = 0
    for t in range(0, nt, every):
        if (frame_idx % 10 == 0) or (t == 0) or (t + every >= nt):
            print(f"Rendering frame {frame_idx+1} / {((nt-1)//every)+1} (t={t})")

        frame_file = tmpdir / f"frame_{frame_idx:06d}.png"
        quicklook_plot(
            nc_path,
            kind,
            t,
            out_png=str(frame_file),
            show=False,
            plot_step=plot_step,
            plot_domain=plot_domain,
            plot_bbox=plot_bbox,
            speed_vmin=speed_vmin,
            speed_vmax=speed_vmax,
            pres_vmin=pres_vmin,
            pres_vmax=pres_vmax,
        )
        frame_idx += 1

    # Encode with ffmpeg
    cmd = [
        ffmpeg_exe,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(pattern),
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_mp4),
    ]
    if verbose:
        print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote movie: {out_mp4}")

    if not keep_frames:
        for p in tmpdir.glob("frame_*.png"):
            p.unlink()
        tmpdir.rmdir()
    else:
        print(f"Kept frames in: {tmpdir}")


# -----------------------------
# Extraction
# -----------------------------
def _choose_target_z(
    z_field: np.ndarray,
    requested_z: float,
    z_auto: bool,
    dz: float,
) -> float:
    zmin = float(np.nanmin(np.asarray(z_field)))
    if z_auto:
        return zmin + dz
    return requested_z


def _interplevel_with_optional_fallback(
    ua: np.ndarray,
    va: np.ndarray,
    pres_pa: np.ndarray,
    z_field: np.ndarray,
    z_target: float,
    allow_fallback: bool,
    dz: float,
    verbose: bool,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """
    Returns (ua_z, va_z, p_z, z_used, used_fallback)
    """
    ua_z = _interplevel_timewise(ua, z_field, z_target)
    va_z = _interplevel_timewise(va, z_field, z_target)
    p_z = _interplevel_timewise(pres_pa, z_field, z_target)

    nan_all = (
        np.isnan(np.asarray(ua_z)).all()
        and np.isnan(np.asarray(va_z)).all()
        and np.isnan(np.asarray(p_z)).all()
    )

    if verbose:
        zmin = float(np.nanmin(np.asarray(z_field)))
        zmax = float(np.nanmax(np.asarray(z_field)))
        print(f"[{label}] z_field range: {zmin:.2f} .. {zmax:.2f} m, target={z_target:.2f}")
        print(
            f"  NaN frac ua_z={_nan_frac(ua_z):.3f}, va_z={_nan_frac(va_z):.3f}, p_z={_nan_frac(p_z):.3f}"
        )

    if (not nan_all) or (not allow_fallback):
        return np.asarray(ua_z), np.asarray(va_z), np.asarray(p_z), float(z_target), False

    # Fallback: z = zmin + dz
    zmin = float(np.nanmin(np.asarray(z_field)))
    z_fallback = zmin + dz
    if verbose:
        print(f"[{label}] interplevel all-NaN at z={z_target:.2f}; falling back to z={z_fallback:.2f} (zmin+dz)")

    ua_z2 = _interplevel_timewise(ua, z_field, z_fallback)
    va_z2 = _interplevel_timewise(va, z_field, z_fallback)
    p_z2 = _interplevel_timewise(pres_pa, z_field, z_fallback)

    if verbose:
        print(
            f"  Fallback NaN frac ua_z={_nan_frac(ua_z2):.3f}, va_z={_nan_frac(va_z2):.3f}, p_z={_nan_frac(p_z2):.3f}"
        )

    return np.asarray(ua_z2), np.asarray(va_z2), np.asarray(p_z2), float(z_fallback), True

def _expand_inputs(inputs: List[str], verbose: bool = False) -> List[str]:
    """Expand user inputs (files/dirs/globs) into a sorted list of NetCDF files.

    Rules:
      - If an entry is a directory, include *.nc in that directory (non-recursive).
      - If an entry contains glob metacharacters (* ? [), expand it.
      - Otherwise, treat as a file path.

    Returns a de-duplicated list preserving sorted order.
    """
    import glob
    import os

    expanded: List[str] = []
    for item in inputs:
        item = os.path.expanduser(os.path.expandvars(item))
        if any(ch in item for ch in ["*", "?", "["]):
            expanded.extend(sorted(glob.glob(item)))
        elif os.path.isdir(item):
            expanded.extend(sorted(glob.glob(os.path.join(item, "*.nc"))))
        else:
            expanded.append(item)

    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for f in expanded:
        if f not in seen:
            seen.add(f)
            out.append(f)

    if verbose:
        print(f"[inputs] expanded to {len(out)} files")

    if len(out) == 0:
        raise FileNotFoundError("No input NetCDF files found from provided paths/globs/directories.")

    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    files: List[str] = _expand_inputs(args.files, verbose=args.verbose)
    out_path: str = args.output

    output_exists = os.path.exists(out_path)

    # Decide whether to (re)build the output file
    do_extract = True
    if output_exists and not args.force:
        do_extract = False
        print(f"Output exists, skipping extraction (use --force to overwrite): {out_path}")
    elif output_exists and args.force:
        print(f"Output exists, overwriting due to --force: {out_path}")
        os.remove(out_path)

    if do_extract:
        # Define grid from first file
        with Dataset(files[0]) as nc0:
            sn = nc0.dimensions["south_north"].size
            we = nc0.dimensions["west_east"].size
            datestrlen = nc0.dimensions["DateStrLen"].size if "DateStrLen" in nc0.dimensions else 19
            static = _get_static_fields(nc0)

        # Create output file with unlimited Time
        with Dataset(out_path, "w", format="NETCDF4") as out_nc:
            out_nc.createDimension("Time", None)
            out_nc.createDimension("DateStrLen", datestrlen)
            out_nc.createDimension("south_north", sn)
            out_nc.createDimension("west_east", we)

            out_nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
            out_nc.createVariable("XTIME", "f4", ("Time",))

            comp = dict(zlib=(args.compression > 0), complevel=args.compression)
            chunks = (1, sn, we)

            v_u = out_nc.createVariable("U", "f4", ("Time", "south_north", "west_east"), chunksizes=chunks, **comp)
            v_v = out_nc.createVariable("V", "f4", ("Time", "south_north", "west_east"), chunksizes=chunks, **comp)
            v_p = out_nc.createVariable("P", "f4", ("Time", "south_north", "west_east"), chunksizes=chunks, **comp)

            v_u.units = "m s-1"
            v_v.units = "m s-1"
            v_p.units = "Pa"

            # Store requested settings; effective z may vary by file if fallback/auto is used
            out_nc.setncattr("wrf_extract_requested_z_m", float(args.z))
            out_nc.setncattr("wrf_extract_z_ref", args.z_ref)
            out_nc.setncattr("wrf_extract_z_auto", int(bool(args.z_auto)))
            out_nc.setncattr("wrf_extract_z_auto_dz_m", float(args.z_auto_dz))
            out_nc.setncattr("wrf_extract_z_fallback_enabled", int(not args.no_z_fallback))
            out_nc.setncattr("source_files", " ".join(files))

            if args.keep_geo:
                if "XLAT" in static:
                    v_lat = out_nc.createVariable("XLAT", "f4", ("south_north", "west_east"))
                    v_lat[:] = static["XLAT"]
                    v_lat.units = "degrees_north"
                if "XLONG" in static:
                    v_lon = out_nc.createVariable("XLONG", "f4", ("south_north", "west_east"))
                    v_lon[:] = static["XLONG"]
                    v_lon.units = "degrees_east"
                if "HGT" in static:
                    v_hgt = out_nc.createVariable("HGT", "f4", ("south_north", "west_east"))
                    v_hgt[:] = static["HGT"]
                    v_hgt.units = "m"

            t_out = 0
            for fi, f in enumerate(files, start=1):
                print(f"[{fi}/{len(files)}] Processing: {os.path.basename(f)}")
                with Dataset(f) as nc:
                    _ensure_same_hgrid(nc, sn, we)

                    times = nc.variables["Times"][:, :]
                    xtime = nc.variables["XTIME"][:].astype("f4")
                    nt = times.shape[0]

                    # WRF-aware vars
                    ua = getvar(nc, "ua", timeidx=ALL_TIMES)  # m/s
                    va = getvar(nc, "va", timeidx=ALL_TIMES)  # m/s
                    pres_hpa = getvar(nc, "pressure", timeidx=ALL_TIMES)  # hPa
                    pres_pa = pres_hpa * 100.0
                    z_msl = getvar(nc, "z", timeidx=ALL_TIMES)  # m MSL (mass grid)

                    if args.z_ref == "agl":
                        if "HGT" not in nc.variables:
                            raise KeyError("Requested --z-ref agl but HGT is missing in file.")
                        hgt = nc.variables["HGT"][0, :, :] if nc.variables["HGT"].ndim == 3 else nc.variables["HGT"][:, :]
                        z_field = z_msl - hgt[None, None, :, :]
                    else:
                        z_field = z_msl

                    # Choose requested z for this file (auto or user-specified)
                    z_req = _choose_target_z(
                        z_field=z_field,
                        requested_z=float(args.z),
                        z_auto=bool(args.z_auto),
                        dz=float(args.z_auto_dz),
                    )

                    # Interpolate with optional fallback if all-NaN
                    ua_z, va_z, p_z, z_used, used_fallback = _interplevel_with_optional_fallback(
                        ua=ua,
                        va=va,
                        pres_pa=pres_pa,
                        z_field=z_field,
                        z_target=z_req,
                        allow_fallback=(not args.no_z_fallback) and (not args.z_auto),
                        dz=float(args.z_auto_dz),
                        verbose=bool(args.verbose),
                        label=os.path.basename(f),
                    )

                    if used_fallback:
                        print(f"  NOTE: used fallback z={z_used:.2f} m for {os.path.basename(f)}")

                    # Write time + fields
                    out_nc.variables["Times"][t_out:t_out + nt, :] = times
                    out_nc.variables["XTIME"][t_out:t_out + nt] = xtime
                    out_nc.variables["U"][t_out:t_out + nt, :, :] = np.asarray(ua_z, dtype="f4")
                    out_nc.variables["V"][t_out:t_out + nt, :, :] = np.asarray(va_z, dtype="f4")
                    out_nc.variables["P"][t_out:t_out + nt, :, :] = np.asarray(p_z, dtype="f4")

                    t_out += nt

            # Update long_name after the fact (since effective z may vary by file)
            v_u.long_name = f"U wind interpolated to z (see attributes; requested z={float(args.z)} m, ref={args.z_ref})"
            v_v.long_name = f"V wind interpolated to z (see attributes; requested z={float(args.z)} m, ref={args.z_ref})"
            v_p.long_name = f"Pressure interpolated to z (see attributes; requested z={float(args.z)} m, ref={args.z_ref})"

            out_nc.sync()

        print(f"Wrote collated file: {out_path}")

    # Plots/movies should run even if extraction was skipped (output already existed).
    if args.plot:
        quicklook_plot(
            out_path,
            args.plot_kind,
            args.plot_time,
            args.plot_out,
            args.plot_show,
            plot_step=args.plot_step,
            plot_domain=args.plot_domain,
            plot_bbox=args.plot_bbox,
        )

    if args.movie:
        make_movie(
            out_path,
            args.movie_kind,
            args.movie_fps,
            args.movie_every,
            args.movie_out,
            args.ffmpeg,
            args.keep_frames,
            verbose=bool(args.verbose),
            plot_step=args.plot_step,
            plot_domain=args.plot_domain,
            plot_bbox=args.plot_bbox,
            fixed_scale=args.movie_fixed_scale,
            scale_mode=args.movie_scale,
            pmin=args.movie_pmin,
            pmax=args.movie_pmax,
        )


if __name__ == "__main__":
    main()
