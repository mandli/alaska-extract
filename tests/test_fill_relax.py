import numpy as np
from wrf_regrid_uvp_to_latlon import inpaint_nearest_2d, relax_invalid_to_bg

def test_inpaint_fills_nans():
    a = np.array([[1.0, np.nan], [2.0, np.nan]])
    valid = np.isfinite(a)
    filled = inpaint_nearest_2d(a, valid)
    assert np.all(np.isfinite(filled))
    # valid cells unchanged
    assert filled[0,0] == 1.0 and filled[1,0] == 2.0

def test_relax_only_invalid():
    filled = np.array([[10.0, 10.0],[10.0, 10.0]])
    valid = np.array([[True, True],[True, False]])
    out, w, dist = relax_invalid_to_bg(
        filled, valid, dlat_deg=1.0, dlon_deg=1.0, radius_deg=2.0, bg_value=0.0
    )
    # valid region unchanged
    assert out[0,0] == 10.0
    # invalid region relaxed toward bg
    assert out[1,1] < 10.0
    assert 0.0 <= w[1,1] <= 1.0
