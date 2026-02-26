import numpy as np
from alaska_extract.common import lon_continuous_about, decode_wrf_times, sanitize_netcdf_attrs

def test_lon_continuous_about_wrap():
    lon = np.array([-179.0, 179.0, 178.0, -178.0])
    out = lon_continuous_about(lon, center_deg=180.0)
    # Should be continuous near 180 and include values > 180
    assert np.max(out) - np.min(out) < 10.0

def test_decode_wrf_times_bytes():
    s = b"2011-11-01_06:00:00"
    assert decode_wrf_times(s) == "2011-11-01_06:00:00"

def test_sanitize_attrs_bool():
    d = {"a": True, "b": np.bool_(False)}
    out = sanitize_netcdf_attrs(d)
    assert out["a"] == 1 and out["b"] == 0
