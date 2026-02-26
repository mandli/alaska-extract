#!/usr/bin/env python3
"""
wrf_extract_uvp_at_z.py

Concatenate a set of WRF NetCDF files and extract U/V winds + pressure at a target geometric height,
then optionally make quick-look plots and ffmpeg movies.

Outputs a compact NetCDF with the original full Time dimension:
  - Times(Time, DateStrLen)  [bytes/char]
  - XTIME(Time)              [minutes since simulation start, per WRF]
  - U(Time, south_north, west_east)  [m s-1] at target z
  - V(Time, south_north, west_east)  [m s-1] at target z
  - P(Time, south_north, west_east)  [Pa] at target z
Optional with --keep-geo:
  - XLAT(south_north, west_east)  [deg_north]
  - XLONG(south_north, west_east) [deg_east]
  - HGT(south_north, west_east)   [m] terrain height

Notes on z-level selection:
  - z-ref=msl uses WRF geopotential height (PH+PHB)/g
  - z-ref=agl uses (PH+PHB)/g - HGT
  - If requested z yields all-NaN (below terrain everywhere), the script can fall back to
    z = zmin + dz (default dz=10m) unless --no-z-fallback is set.
  - --z-auto forces z = zmin + dz per file.

Input expansion:
  - You can pass files, directories, or glob patterns. Directories are searched for *.nc.
  - Default ordering is by filename. Use --sort time to sort by Times[0] from each file.

Requires (wrf env):
  numpy, netCDF4, wrf-python, xarray (optional), matplotlib (optional), ffmpeg (optional)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from netCDF4 import Dataset

from wrf import getvar, interplevel

from alaska_extract.common import decode_wrf_times, find_ffmpeg, vprint, lon_continuous_about
from alaska_extract.version import VERSION
from alaska_extract.plotting import FixedScale, compute_fixed_scale, plot_uvp_quicklook, movie_from_frames


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract U/V and pressure from WRF at a given z-level and concatenate over Time."
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    p.add_argument("inputs", nargs="+", help="Input files/dirs/globs (e.g. h01_files/*.nc).")
    p.add_argument("-o", "--output", required=True, help="Output NetCDF filename.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists. If not set and output exists, extraction is skipped "
             "(but plot/movie can still run).",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")

    p.add_argument("--sort", choices=["name", "time"], default="name",
                   help="How to order expanded inputs. Default: name.")
    p.add_argument("--glob", default="*.nc",
                   help="When an input is a directory, glob pattern to collect files. Default: *.nc")

    p.add_argument("--z", type=float, default=0.0, help="Target height in meters. Default: 0 (MSL).")
    p.add_argument("--z-ref", choices=["msl", "agl"], default="msl",
                   help="Interpret --z as mean sea level (msl) or above ground (agl). Default: msl.")

    p.add_argument("--z-auto", action="store_true",
                   help="Use z = zmin + dz per file (ignores --z).")
    p.add_argument("--z-auto-dz", type=float, default=10.0,
                   help="dz in meters for --z-auto or fallback. Default: 10.")
    p.add_argument("--no-z-fallback", action="store_true",
                   help="Disable fallback if interplevel returns all-NaN at requested z.")

    p.add_argument("--compression", type=int, default=4, help="NetCDF deflate level (0-9). Default: 4.")

    geo = p.add_mutually_exclusive_group()
    geo.add_argument("--keep-geo", action="store_true", default=True,
                     help="Write XLAT/XLONG/HGT to output (default: on).")
    geo.add_argument("--no-keep-geo", dest="keep_geo", action="store_false",
                     help="Do not store XLAT/XLONG/HGT in output.")

    # Plotting
    p.add_argument("--plot", action="store_true", help="Generate a quick-look plot from the collated file.")
    p.add_argument("--plot-time", type=int, default=0, help="Time index to plot (0-based). Default: 0.")
    p.add_argument("--plot-kind", choices=["speed", "pressure", "quiver", "both"], default="both")
    p.add_argument("--plot-out", default=None, help="Output png filename. Default: derived from nc name.")
    p.add_argument("--plot-show", action="store_true", help="Show interactively instead of saving a PNG.")
    p.add_argument("--plot-step", type=int, default=1, help="Subsample every N grid points (>=1). Default: 1.")
    p.add_argument("--plot-bbox", nargs=4, type=float, metavar=("LON0", "LON1", "LAT0", "LAT1"), default=None,
                   help="Crop by lon/lat box (requires XLONG/XLAT).")

    # Movies
    p.add_argument("--movie", action="store_true", help="Make an MP4 movie via ffmpeg.")
    p.add_argument("--movie-kind", choices=["speed", "pressure", "quiver", "both"], default="both")
    p.add_argument("--movie-fps", type=int, default=10)
    p.add_argument("--movie-every", type=int, default=1)
    p.add_argument("--movie-out", default=None)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--keep-frames", action="store_true")

    # Movie scaling controls (anchored by default)
    p.add_argument("--movie-fixed-scale", action="store_true", default=True,
                   help="Use fixed color scale across frames (default: on).")
    p.add_argument("--no-movie-fixed-scale", dest="movie_fixed_scale", action="store_false",
                   help="Disable fixed scaling (per-frame autoscale).")
    p.add_argument("--movie-scale", choices=["minmax", "percentile"], default="percentile")
    p.add_argument("--movie-pmin", type=float, default=1.0)
    p.add_argument("--movie-pmax", type=float, default=99.0)

    return p.parse_args()


# -----------------------------
# Input expansion + sorting
# -----------------------------
def _has_wildcards(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "["])


def expand_inputs(inputs: List[str], dir_glob: str, verbose: bool) -> List[Path]:
    out: List[Path] = []
    for token in inputs:
        p = Path(token).expanduser()
        # explicit wildcards
        if _has_wildcards(token):
            parent = p.parent if str(p.parent) != "" else Path(".")
            pattern = p.name
            matches = sorted(parent.glob(pattern))
            out.extend([m for m in matches if m.is_file()])
            continue

        if p.is_dir():
            matches = sorted(p.glob(dir_glob))
            out.extend([m for m in matches if m.is_file()])
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"Input not found: {token}")

    # de-duplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for f in out:
        key = str(f.resolve())
        if key not in seen:
            uniq.append(f)
            seen.add(key)

    vprint(verbose, f"[inputs] expanded to {len(uniq)} files")
    return uniq


def _first_timestr(path: Path) -> str:
    with Dataset(path, "r") as nc:
        if "Times" not in nc.variables:
            return ""
        v = nc.variables["Times"][0, :]
        return decode_wrf_times(v.tobytes())


def sort_inputs(files: List[Path], how: str, verbose: bool) -> List[Path]:
    if how == "name":
        return sorted(files, key=lambda p: str(p))
    # time sort
    pairs = [(f, _first_timestr(f)) for f in files]
    if verbose:
        for f, t in pairs[:5]:
            print(f"[sort] {t}  {f}")
    return [f for f, _ in sorted(pairs, key=lambda x: x[1])]


# -----------------------------
# Core extraction
# -----------------------------
def _z_field(nc: Dataset, z_ref: str):
    """Return z field (Time, bottom_top_stag, sn, we) as a wrf-python variable."""
    ph = getvar(nc, "PH", timeidx=None)
    phb = getvar(nc, "PHB", timeidx=None)
    g = 9.81
    z = (ph + phb) / g  # geopotential height (m, MSL) on stag levels
    if z_ref == "agl":
        hgt = getvar(nc, "HGT", timeidx=0)
        z = z - hgt
    return z


def _interp_at_z(nc: Dataset, varname: str, z: float, zref: str):
    zf = _z_field(nc, zref)
    v = getvar(nc, varname, timeidx=None)
    return interplevel(v, zf, z)


def _read_static_geo(nc: Dataset):
    out = {}
    if "XLAT" in nc.variables:
        out["XLAT"] = nc.variables["XLAT"][0, :, :].astype("f4")
    if "XLONG" in nc.variables:
        out["XLONG"] = nc.variables["XLONG"][0, :, :].astype("f4")
    if "HGT" in nc.variables:
        v = nc.variables["HGT"]
        out["HGT"] = (v[0, :, :] if v.ndim == 3 else v[:, :]).astype("f4")
    return out


def extract(files: List[Path], out_nc: Path, *, z: float, zref: str, z_auto: bool, dz: float,
            no_fallback: bool, keep_geo: bool, compression: int, verbose: bool) -> None:
    # Inspect first file for dimensions
    with Dataset(files[0], "r") as nc0:
        sn = nc0.dimensions["south_north"].size
        we = nc0.dimensions["west_east"].size
        datestrlen = nc0.dimensions["DateStrLen"].size if "DateStrLen" in nc0.dimensions else 19

        static = _read_static_geo(nc0) if keep_geo else {}

    # Gather time counts
    counts = []
    for f in files:
        with Dataset(f, "r") as nc:
            if nc.dimensions["south_north"].size != sn or nc.dimensions["west_east"].size != we:
                raise ValueError(f"Grid mismatch in {f}")
            counts.append(nc.dimensions["Time"].size)
    nt = int(sum(counts))
    vprint(verbose, f"[extract] total Time={nt} from {len(files)} files")

    # Prepare output
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    if out_nc.exists():
        out_nc.unlink()

    with Dataset(out_nc, "w", format="NETCDF4") as out:
        out.createDimension("Time", nt)
        out.createDimension("south_north", sn)
        out.createDimension("west_east", we)
        out.createDimension("DateStrLen", datestrlen)

        Times = out.createVariable("Times", "S1", ("Time", "DateStrLen"))
        XTIME = out.createVariable("XTIME", "f8", ("Time",))
        Uo = out.createVariable("U", "f4", ("Time", "south_north", "west_east"),
                                zlib=True, complevel=compression)
        Vo = out.createVariable("V", "f4", ("Time", "south_north", "west_east"),
                                zlib=True, complevel=compression)
        Po = out.createVariable("P", "f4", ("Time", "south_north", "west_east"),
                                zlib=True, complevel=compression)

        Uo.units = "m s-1"
        Vo.units = "m s-1"
        Po.units = "Pa"

        if keep_geo and static:
            out.createVariable("XLAT", "f4", ("south_north", "west_east"))[:] = static["XLAT"]
            out["XLAT"].units = "degrees_north"
            out.createVariable("XLONG", "f4", ("south_north", "west_east"))[:] = static["XLONG"]
            out["XLONG"].units = "degrees_east"
            if "HGT" in static:
                out.createVariable("HGT", "f4", ("south_north", "west_east"))[:] = static["HGT"]
                out["HGT"].units = "m"

        # Copy global attrs from first file
        with Dataset(files[0], "r") as nc0:
            for k in nc0.ncattrs():
                try:
                    out.setncattr(k, nc0.getncattr(k))
                except Exception:
                    pass

        out.setncattr("extract_z_m", float(z))
        out.setncattr("extract_z_ref", str(zref))
        out.setncattr("extract_z_auto", int(bool(z_auto)))
        out.setncattr("extract_z_auto_dz_m", float(dz))

        t0 = 0
        for f in files:
            with Dataset(f, "r") as nc:
                nT = nc.dimensions["Time"].size
                # Times and XTIME
                if "Times" in nc.variables:
                    Times[t0:t0+nT, :] = nc.variables["Times"][:, :]
                if "XTIME" in nc.variables:
                    XTIME[t0:t0+nT] = nc.variables["XTIME"][:]
                else:
                    XTIME[t0:t0+nT] = np.arange(nT, dtype=np.float64)

                # choose z per file if needed
                z_use = float(z)
                if z_auto:
                    zf = _z_field(nc, zref)
                    zmin = float(np.nanmin(zf.values))
                    z_use = zmin + float(dz)
                ua = _interp_at_z(nc, "ua", z_use, zref)
                va = _interp_at_z(nc, "va", z_use, zref)
                p = _interp_at_z(nc, "p", z_use, zref)  # Pa

                # fallback if all NaN
                if (not no_fallback) and (np.isnan(ua.values).all() or np.isnan(va.values).all() or np.isnan(p.values).all()):
                    zf = _z_field(nc, zref)
                    zmin = float(np.nanmin(zf.values))
                    z_fb = zmin + float(dz)
                    vprint(verbose, f"[{f.name}] all-NaN at z={z_use:.2f}, fallback z={z_fb:.2f}")
                    ua = _interp_at_z(nc, "ua", z_fb, zref)
                    va = _interp_at_z(nc, "va", z_fb, zref)
                    p = _interp_at_z(nc, "p", z_fb, zref)

                if verbose:
                    zf = _z_field(nc, zref)
                    zmin, zmax = float(np.nanmin(zf.values)), float(np.nanmax(zf.values))
                    nan_u = float(np.mean(~np.isfinite(ua.values)))
                    nan_v = float(np.mean(~np.isfinite(va.values)))
                    nan_p = float(np.mean(~np.isfinite(p.values)))
                    print(f"[{f.name}] z_field range: {zmin:.2f} .. {zmax:.2f} m ({zref}), target={z_use:.2f}")
                    print(f"  NaN frac ua_z={nan_u:.3f}, va_z={nan_v:.3f}, p_z={nan_p:.3f}")

                Uo[t0:t0+nT, :, :] = ua.values.astype("f4")
                Vo[t0:t0+nT, :, :] = va.values.astype("f4")
                Po[t0:t0+nT, :, :] = p.values.astype("f4")

                t0 += nT


# -----------------------------
# Plot/movie helpers (quicklook only; lat/lon plots prefer regrid script)
# -----------------------------
def _decode_time_for_title(ds, t_index: int) -> str:
    if "Times" not in ds.variables:
        return ""
    v = ds["Times"][t_index, :]
    try:
        return decode_wrf_times(v.tobytes())
    except Exception:
        return decode_wrf_times(v)


def _plot_quicklook(
    ds,
    t_index: int,
    kind: str,
    out_png: Optional[str],
    show: bool,
    step: int,
    bbox: Optional[Tuple[float, float, float, float]],
    *,
    fixed_scale: bool = False,
    scale_speed: Optional["FixedScale"] = None,
    scale_pressure: Optional["FixedScale"] = None,
) -> None:
    """
    Thin wrapper around alaska_extract.plotting.plot_uvp_quicklook.

    Handles:
      - lon "continuous" unwrap for Alaska/Pacific
      - optional bbox cropping on lon/lat
      - downsampling via `step`
    """
    # Pull fields
    U = np.asarray(ds["U"][t_index, :, :], dtype=np.float64)
    V = np.asarray(ds["V"][t_index, :, :], dtype=np.float64)
    P = np.asarray(ds["P"][t_index, :, :], dtype=np.float64)

    # x/y: prefer lon/lat if present
    if "XLAT" in ds.variables and "XLONG" in ds.variables:
        lat = np.asarray(ds["XLAT"][:, :], dtype=np.float64)
        lon = np.asarray(ds["XLONG"][:, :], dtype=np.float64)
        lon = lon_continuous_about(lon)  # continuous for Alaska/Pacific
        X, Y = lon, lat
        if bbox is not None:
            lon0, lon1, lat0, lat1 = bbox
            m = (X >= lon0) & (X <= lon1) & (Y >= lat0) & (Y <= lat1)
            if np.any(m):
                jj, ii = np.where(m)
                j0, j1 = int(jj.min()), int(jj.max()) + 1
                i0, i1 = int(ii.min()), int(ii.max()) + 1
                X = X[j0:j1, i0:i1]
                Y = Y[j0:j1, i0:i1]
                U = U[j0:j1, i0:i1]
                V = V[j0:j1, i0:i1]
                P = P[j0:j1, i0:i1]
    else:
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))

    # downsample
    step = max(1, int(step))
    X = X[::step, ::step]
    Y = Y[::step, ::step]
    U = U[::step, ::step]
    V = V[::step, ::step]
    P = P[::step, ::step]

    title_time = _decode_time_for_title(ds, t_index)
    plot_uvp_quicklook(
        X,
        Y,
        U,
        V,
        P,
        title=title_time,
        kind=kind,
        out_png=out_png,
        show=show,
        fixed_scale=fixed_scale,
        scale_speed=scale_speed,
        scale_pressure=scale_pressure,
    )


def _compute_movie_scales(
    ds,
    *,
    fixed_scale: bool,
    scale_mode: str,
    pmin: float,
    pmax: float,
) -> Tuple[Optional[FixedScale], Optional[FixedScale]]:
    """
    Compute anchored scales for movies. Returns (speed_scale, pressure_scale).
    """
    if not fixed_scale:
        return None, None
    U = np.asarray(ds["U"][:], dtype=np.float64)
    V = np.asarray(ds["V"][:], dtype=np.float64)
    P = np.asarray(ds["P"][:], dtype=np.float64)

    spd = np.sqrt(U * U + V * V)
    sc_spd = compute_fixed_scale(spd, mode=("minmax" if scale_mode == "minmax" else "percentile"),
                                 pmin=float(pmin), pmax=float(pmax))
    sc_p = compute_fixed_scale(P, mode=("minmax" if scale_mode == "minmax" else "percentile"),
                               pmin=float(pmin), pmax=float(pmax))
    return sc_spd, sc_p


def make_movie(
    nc_path: Path,
    kind: str,
    fps: int,
    every: int,
    out_mp4: Optional[str],
    ffmpeg: str,
    keep_frames: bool,
    fixed_scale: bool,
    scale_mode: str,
    pmin: float,
    pmax: float,
    verbose: bool,
    step: int,
    bbox: Optional[Tuple[float, float, float, float]],
) -> None:
    """
    Render a simple MP4 movie from the extracted file.

    Notes:
      - Uses anchored color scales by default (see --movie-fixed-scale).
      - Opens the dataset once (xarray) for speed and consistency.
    """
    base = nc_path.with_suffix("")
    if out_mp4 is None:
        out_mp4 = str(base) + f".{kind}.mp4"

    tmpdir = Path(tempfile.mkdtemp(prefix="uvp_frames_"))
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        nt = int(ds.sizes.get("Time", 1))
        every = max(1, int(every))

        sc_spd, sc_p = _compute_movie_scales(
            ds, fixed_scale=bool(fixed_scale), scale_mode=str(scale_mode), pmin=float(pmin), pmax=float(pmax)
        )
        if fixed_scale and sc_spd is not None and sc_p is not None:
            vprint(verbose, f"[movie] speed scale: {sc_spd.vmin:.3g} .. {sc_spd.vmax:.3g}")
            vprint(verbose, f"[movie] press scale: {sc_p.vmin:.3g} .. {sc_p.vmax:.3g}")

        frame = 0
        for t in range(0, nt, every):
            png = tmpdir / f"frame_{frame:05d}.png"
            _plot_quicklook(
                ds,
                t,
                kind,
                str(png),
                False,
                step,
                bbox,
                fixed_scale=bool(fixed_scale),
                scale_speed=sc_spd,
                scale_pressure=sc_p,
            )
            frame += 1

        ds.close()
        movie_from_frames(tmpdir, Path(out_mp4), fps=int(fps), ffmpeg=ffmpeg, verbose=bool(verbose),
                          keep_frames=bool(keep_frames))

        if keep_frames:
            keep = Path(str(base) + f"_{kind}_frames")
            if keep.exists():
                shutil.rmtree(keep)
            shutil.copytree(tmpdir, keep)
            print(f"Kept frames: {keep}")

    finally:
        if not keep_frames:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    out_nc = Path(args.output).expanduser()

    files = expand_inputs(args.inputs, args.glob, args.verbose)
    files = sort_inputs(files, args.sort, args.verbose)

    if out_nc.exists() and not args.force:
        vprint(args.verbose, f"[main] output exists, skipping extraction: {out_nc}")
    else:
        extract(
            files,
            out_nc,
            z=float(args.z),
            zref=str(args.z_ref),
            z_auto=bool(args.z_auto),
            dz=float(args.z_auto_dz),
            no_fallback=bool(args.no_z_fallback),
            keep_geo=bool(args.keep_geo),
            compression=int(args.compression),
            verbose=bool(args.verbose),
        )
        print(f"Wrote: {out_nc}")

    # Optional quicklook/movie operate on existing output
    if args.plot or args.movie:
        import xarray as xr
        ds = xr.open_dataset(out_nc)

        bbox = tuple(args.plot_bbox) if args.plot_bbox is not None else None

        if args.plot:
            out_png = args.plot_out
            if out_png is None and not args.plot_show:
                out_png = str(out_nc.with_suffix(f".{args.plot_kind}.png"))
            sc_spd = sc_p = None
            if args.plot_fixed_scale:
                sc_spd, sc_p = _compute_movie_scales(ds, fixed_scale=True, scale_mode="percentile", pmin=1.0, pmax=99.0)
            _plot_quicklook(ds, args.plot_time, args.plot_kind, out_png, args.plot_show,
                            max(1, args.plot_step), bbox, fixed_scale=bool(args.plot_fixed_scale),
                            scale_speed=sc_spd, scale_pressure=sc_p)

        if args.movie:
            make_movie(
                out_nc,
                kind=args.movie_kind,
                fps=args.movie_fps,
                every=args.movie_every,
                out_mp4=args.movie_out,
                ffmpeg=args.ffmpeg,
                keep_frames=args.keep_frames,
                fixed_scale=bool(args.movie_fixed_scale),
                scale_mode=str(args.movie_scale),
                pmin=float(args.movie_pmin),
                pmax=float(args.movie_pmax),
                verbose=bool(args.verbose),
                step=max(1, args.plot_step),
                bbox=bbox,
            )

        ds.close()


if __name__ == "__main__":
    main()
