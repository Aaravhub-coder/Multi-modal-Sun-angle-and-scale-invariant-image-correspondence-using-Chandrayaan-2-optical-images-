"""
Evaluation metrics for a set of matches.

NOTE: RMSE here is the residual of the INLIER matches against the fitted
transform. It measures how self-consistent the registration is, not the true
error against ground truth (we don't have independent checkpoints yet).
"""
import numpy as np


def apply_affine(M, pts):
    """Apply a 2x3 affine matrix to (N,2) points."""
    return pts @ M[:, :2].T + M[:, 2]


def coverage(points, shape, grid=(4, 4)):
    """
    Spatial-distribution check: fraction of cells in a grid x grid partition of
    the image that contain at least one match (1.0 = fully uniform coverage).
    `points` are (x, y) in an image of size `shape` = (H, W).
    """
    if len(points) == 0:
        return 0.0
    H, W = shape
    gx = np.clip((points[:, 0] / W * grid[1]).astype(int), 0, grid[1] - 1)
    gy = np.clip((points[:, 1] / H * grid[0]).astype(int), 0, grid[0] - 1)
    return len({(a, b) for a, b in zip(gy, gx)}) / (grid[0] * grid[1])


def compute_metrics(src_pts, ref_pts, inlier_mask, M, src_shape, pixel_m=None, n_candidates=None):
    """
    src_pts, ref_pts : (N,2) matched points (x, y) used for the fit
    inlier_mask      : (N,) bool
    M                : 2x3 affine (source -> reference) or None
    n_candidates     : number of matches before RANSAC (after ratio test)
    """
    n = int(len(src_pts))
    k = int(inlier_mask.sum()) if inlier_mask is not None else 0
    out = {
        "matches_before_ransac": n_candidates if n_candidates is not None else n,
        "ncm_inliers": k,
        "inlier_ratio": (k / n) if n else 0.0,
        "success": bool(M is not None and k >= 15),
    }
    if M is not None and k > 0:
        pred = apply_affine(M, src_pts[inlier_mask])
        err = np.linalg.norm(pred - ref_pts[inlier_mask], axis=1)
        out["rmse_px"] = float(np.sqrt(np.mean(err ** 2)))
        out["max_err_px"] = float(err.max())
        if pixel_m:
            out["rmse_m"] = out["rmse_px"] * pixel_m
        out["coverage_4x4"] = coverage(src_pts[inlier_mask], src_shape, (4, 4))
        # transform parameters (similarity): scale and rotation
        a, b = M[0, 0], M[1, 0]
        out["scale"] = float(np.hypot(a, b))
        out["rotation_deg"] = float(np.degrees(np.arctan2(b, a)))
    return out
