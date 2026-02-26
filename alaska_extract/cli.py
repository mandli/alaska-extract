"""Command-line entrypoint.

This provides a consistent interface while still allowing pixi to manage
separate environments for extraction vs regridding.

In practice:
  - In the `wrf` pixi env, use:  alaska-extract extract [args...]
  - In the `regrid` pixi env, use: alaska-extract regrid [args...]

Each subcommand simply delegates to the corresponding top-level script in the
repository checkout. This avoids import-time dependency issues across envs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    # alaska_extract/cli.py -> alaska_extract -> repo root
    return Path(__file__).resolve().parents[1]


def _script_path(name: str) -> Path:
    root = _repo_root()
    p = root / name
    if not p.exists():
        raise FileNotFoundError(f"Cannot find {name} at repo root: {p}")
    return p


def _run_script(script: Path, argv: list[str]) -> int:
    cmd = [sys.executable, str(script)] + argv
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alaska-extract", description="Alaska WRF extract/regrid utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="Extract U/V/P from WRF at a z-level")
    p_ex.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to wrf_extract_uvp_at_z.py")

    p_rg = sub.add_parser("regrid", help="Regrid extracted U/V/P to regular lat/lon (xESMF)")
    p_rg.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to wrf_regrid_uvp_to_latlon.py")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = build_parser()
    ns = p.parse_args(argv)

    if ns.cmd == "extract":
        return _run_script(_script_path("wrf_extract_uvp_at_z.py"), ns.args)
    if ns.cmd == "regrid":
        return _run_script(_script_path("wrf_regrid_uvp_to_latlon.py"), ns.args)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
