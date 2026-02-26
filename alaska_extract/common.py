"""
Shared helpers for alaska_extract scripts.

This module is intentionally lightweight so it can be imported from both pixi
environments (wrf and regrid). Avoid importing heavy optional deps at module import
time (e.g., xesmf, wrf-python).
"""
from __future__ import annotations

import ast

import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


def vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def find_ffmpeg(ffmpeg: str = "ffmpeg") -> str:
    """Return an ffmpeg executable path or raise RuntimeError."""
    if os.path.isabs(ffmpeg):
        if os.path.exists(ffmpeg) and os.access(ffmpeg, os.X_OK):
            return ffmpeg
        raise RuntimeError(f"ffmpeg not executable: {ffmpeg}")
    path = shutil.which(ffmpeg)
    if path:
        return path
    raise RuntimeError(f"ffmpeg not found ('{ffmpeg}'). Install ffmpeg or pass --ffmpeg /path/to/ffmpeg")


def decode_wrf_times(value) -> str:
    """
    Decode WRF 'Times' (char array) or bytes into a clean string.
    Accepts:
      - numpy array of shape (DateStrLen,) with dtype 'S1' or 'U1'
      - bytes/bytearray
      - str
    """
    if value is None:
        return ""
    if isinstance(value, str):
        v = value.strip()
        # Some pipelines accidentally stringify bytes, yielding "b'...'"
        if (v.startswith("b\'") and v.endswith("\'")) or (v.startswith('b"') and v.endswith('"')):
            try:
                b = ast.literal_eval(v)
                if isinstance(b, (bytes, bytearray)):
                    return bytes(b).decode("utf-8", errors="ignore").strip()
            except Exception:
                pass
        return v
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="ignore").strip()

    arr = np.asarray(value)
    # Many WRF files store Times as 'S1' char arrays
    try:
        if arr.dtype.kind in {"S", "U"}:
            if arr.dtype.kind == "S":
                # join bytes
                b = b"".join([bytes(x) for x in arr.reshape(-1).tolist()])
                return b.decode("utf-8", errors="ignore").strip()
            else:
                return "".join(arr.reshape(-1).tolist()).strip()
    except Exception:
        pass

    # Fallback
    return str(value).strip()


def sanitize_netcdf_attrs(attrs: dict) -> dict:
    """
    netCDF4 does not like bool (numpy b1) attrs.
    Convert:
      - bool / np.bool_ -> int
      - Path -> str
    """
    out = {}
    for k, v in (attrs or {}).items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = int(v)
        elif isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def lon_continuous_about(lon_deg: np.ndarray, center_deg: Optional[float] = None) -> np.ndarray:
    """
    Shift longitudes into a continuous window around center_deg.
    Result is in [center-180, center+180), which may extend outside [-180, 180).

    Robust for curvilinear grids (unlike np.unwrap along an axis).
    """
    lon = np.asarray(lon_deg, dtype=np.float64)
    if center_deg is None:
        lon180 = ((lon + 180.0) % 360.0) - 180.0
        center_deg = float(np.nanmedian(lon180))
    return center_deg + (((lon - center_deg + 180.0) % 360.0) - 180.0)


def infer_lon_center(lon_deg: np.ndarray) -> float:
    lon = np.asarray(lon_deg, dtype=np.float64)
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    return float(np.nanmedian(lon180))


def global_minmax(arr: np.ndarray, mask_nan: bool = True) -> Tuple[float, float]:
    """Min/max ignoring NaNs by default."""
    a = np.asarray(arr, dtype=np.float64)
    if mask_nan:
        return float(np.nanmin(a)), float(np.nanmax(a))
    return float(a.min()), float(a.max())


def percentile_range(arr: np.ndarray, pmin: float, pmax: float) -> Tuple[float, float]:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    return float(np.percentile(a, pmin)), float(np.percentile(a, pmax))
