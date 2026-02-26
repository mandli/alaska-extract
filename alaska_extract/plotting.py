"""Plotting and movie helpers shared by extractor/regridder.

This module deliberately imports matplotlib only inside functions so importing
scripts remains light. ffmpeg discovery lives in alaska_extract.common.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .common import decode_wrf_times, find_ffmpeg, vprint


@dataclass
class FixedScale:
    vmin: float
    vmax: float


def _mpl_setup(show: bool) -> None:
    if not show:
        import matplotlib
        matplotlib.use("Agg")


def decode_time_title(ds, t_index: int) -> str:
    """Decode WRF Times to a clean string for titles."""
    if ds is None or "Times" not in ds:
        return ""
    try:
        v = ds["Times"].isel(Time=t_index).values
        if hasattr(v, "tobytes"):
            return decode_wrf_times(v.tobytes())
        return decode_wrf_times(v)
    except Exception:
        try:
            return str(ds["Times"].isel(Time=t_index).values)
        except Exception:
            return ""


def compute_fixed_scale(arr: np.ndarray, mode: str = "percentile", pmin: float = 1.0, pmax: float = 99.0) -> FixedScale:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return FixedScale(0.0, 1.0)
    if mode == "minmax":
        return FixedScale(float(a.min()), float(a.max()))
    return FixedScale(float(np.percentile(a, pmin)), float(np.percentile(a, pmax)))


def plot_uvp_quicklook(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    U2d: np.ndarray,
    V2d: np.ndarray,
    P2d: np.ndarray,
    *,
    title: str = "",
    kind: str = "both",
    out_png: Optional[str] = None,
    show: bool = False,
    fixed_scale: bool = False,
    scale_speed: Optional[FixedScale] = None,
    scale_pressure: Optional[FixedScale] = None,
) -> None:
    """Quicklook plot for speed/pressure/quiver/both."""
    _mpl_setup(show)
    import matplotlib.pyplot as plt

    spd = np.sqrt(np.asarray(U2d, dtype=np.float64) ** 2 + np.asarray(V2d, dtype=np.float64) ** 2)
    P = np.asarray(P2d, dtype=np.float64)

    if kind == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        opts0 = {}
        if fixed_scale and scale_speed is not None:
            opts0.update({"vmin": scale_speed.vmin, "vmax": scale_speed.vmax})
        m0 = axes[0].pcolormesh(lon2d, lat2d, np.ma.masked_invalid(spd), shading="auto", **opts0)
        cb0 = fig.colorbar(m0, ax=axes[0])
        cb0.set_label("Wind speed (m/s)")
        axes[0].set_title("Wind speed")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        opts1 = {}
        if fixed_scale and scale_pressure is not None:
            opts1.update({"vmin": scale_pressure.vmin, "vmax": scale_pressure.vmax})
        m1 = axes[1].pcolormesh(lon2d, lat2d, np.ma.masked_invalid(P), shading="auto", **opts1)
        cb1 = fig.colorbar(m1, ax=axes[1])
        cb1.set_label("Pressure (Pa)")
        axes[1].set_title("Pressure")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")

        if title:
            fig.suptitle(title)

    else:
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        if kind in ("speed", "quiver"):
            opts = {}
            if fixed_scale and scale_speed is not None:
                opts.update({"vmin": scale_speed.vmin, "vmax": scale_speed.vmax})
            m = ax.pcolormesh(lon2d, lat2d, np.ma.masked_invalid(spd), shading="auto", **opts)
            cb = fig.colorbar(m, ax=ax)
            cb.set_label("Wind speed (m/s)")
            if kind == "quiver":
                ax.quiver(lon2d, lat2d, U2d, V2d, scale=250)
            ax.set_title("Wind speed" if kind == "speed" else "Wind vectors")
        elif kind == "pressure":
            opts = {}
            if fixed_scale and scale_pressure is not None:
                opts.update({"vmin": scale_pressure.vmin, "vmax": scale_pressure.vmax})
            m = ax.pcolormesh(lon2d, lat2d, np.ma.masked_invalid(P), shading="auto", **opts)
            cb = fig.colorbar(m, ax=ax)
            cb.set_label("Pressure (Pa)")
            ax.set_title("Pressure")
        else:
            raise ValueError(kind)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if title:
            ax.set_title(f"{ax.get_title()} | {title}")

    if show:
        plt.show()
    else:
        out_png = out_png or "quicklook.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote plot: {out_png}")


def movie_from_frames(
    frames_dir: Path,
    out_mp4: Path,
    *,
    fps: int = 12,
    ffmpeg: str = "ffmpeg",
    verbose: bool = False,
    keep_frames: bool = False,
) -> None:
    ff = find_ffmpeg(ffmpeg)
    cmd = [
        ff, "-y",
        "-framerate", str(int(fps)),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_mp4),
    ]
    vprint(verbose, "[movie] ffmpeg:", " ".join(cmd))
    import subprocess
    subprocess.run(cmd, check=True)
    print(f"Wrote movie: {out_mp4}")
    if not keep_frames:
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
        vprint(verbose, f"[movie] Removed frames_dir={frames_dir}")
