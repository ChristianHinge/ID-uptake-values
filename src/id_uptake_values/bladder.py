import warnings

import numpy as np

# Normalised axial position (C1 = 0, sacrum = -1) of the FDG bladder spike.
BLADDER_WINDOW = (-1.0, -1.25)


def _edge_level(y, band):
    vals = np.where(y[..., band] > 0, y[..., band], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan band -> nan
        return np.nanmedian(vals, axis=-1, keepdims=True)


def cut_bladder(y, x, window=BLADDER_WINDOW, edge_width=0.02):
    """Cap the bladder spike just below the sacrum.

    Inside `window` each curve is capped at a straight line between its median
    level in a band of `edge_width` on either side of the window. Values below
    that line are untouched, so curves without a spike are unchanged. Zeros
    (outside the scan FOV) are ignored when taking the edge levels; if one edge
    has no data the other edge's level is used, if neither has, nothing is cut.

    y : (..., n) curves on grid x. Returns a copy; curves are not renormalised.
    """
    hi, lo = max(window), min(window)
    y = np.array(y, dtype=float, copy=True)
    head = (x >= hi) & (x <= hi + edge_width)
    feet = (x <= lo) & (x >= lo - edge_width)
    inside = (x < hi) & (x > lo)

    y_head, y_feet = _edge_level(y, head), _edge_level(y, feet)
    y_head = np.where(np.isnan(y_head), y_feet, y_head)
    y_feet = np.where(np.isnan(y_feet), y_head, y_feet)

    x_head, x_feet = hi + edge_width / 2, lo - edge_width / 2  # band centres
    t = (x[inside] - x_head) / (x_feet - x_head)
    baseline = y_head + t * (y_feet - y_head)
    y[..., inside] = np.fmin(y[..., inside], baseline)  # fmin: nan baseline -> no cut
    return y
