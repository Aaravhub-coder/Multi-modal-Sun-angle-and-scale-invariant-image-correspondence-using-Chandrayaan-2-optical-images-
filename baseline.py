"""
Baseline registration: SIFT + ratio test + RANSAC similarity transform.

This is the reference pipeline. Later, the paper's modules (WSGGF scale space,
MGLOH, FSC, ...) replace stages one at a time and are compared against it.
"""
import csv
import json
import os

import cv2
import numpy as np

from metrics import compute_metrics


def _u8(img):
    return (img.pixels * 255).astype(np.uint8)


def detect(img, scale=1.0, nfeatures=8000):
    """
    SIFT keypoints + descriptors, only inside the valid-pixel mask.

    SIFT upsamples the image 2x internally, so on 20+ megapixel images it can
    run out of memory. `scale` < 1 detects on a smaller copy; keypoint
    coordinates are returned in the ORIGINAL pixel frame (divided by scale).
    Use the SAME scale for source and reference so they stay comparable.
    """
    pix = _u8(img)
    mask = cv2.erode(img.mask.astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
    if scale < 1.0:
        size = (int(pix.shape[1] * scale), int(pix.shape[0] * scale))
        pix = cv2.resize(pix, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02)
    kps, desc = sift.detectAndCompute(pix, mask)
    pts = np.float32([k.pt for k in kps]).reshape(-1, 2) / scale
    return pts, desc


def match_ratio(desc_s, desc_r, ratio=0.8):
    """Nearest-neighbour ratio test (Lowe)."""
    if desc_s is None or desc_r is None or len(desc_s) < 2 or len(desc_r) < 2:
        return []
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_s, desc_r, k=2)
    return [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]


def estimate(src_pts, ref_pts, thresh=3.0, min_scale=0.2, max_scale=5.0):
    """Robust similarity transform (rotation + uniform scale + shift)."""
    if len(src_pts) < 3:
        return None, np.zeros(len(src_pts), bool)
    M, inl = cv2.estimateAffinePartial2D(
        src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=thresh,
        maxIters=5000, confidence=0.999)
    if M is None:
        return None, np.zeros(len(src_pts), bool)
    # Reject degenerate fits (e.g. everything collapsed to a point): the
    # similarity scale must be physically plausible.
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not (min_scale <= scale <= max_scale):
        return None, np.zeros(len(src_pts), bool)
    return M, inl.ravel().astype(bool)


def warp(img, M, out_shape):
    H, W = out_shape
    return cv2.warpAffine(_u8(img), M, (W, H), flags=cv2.INTER_LINEAR)


def _draw_matches(src, ref, s_pts, r_pts, path, max_lines=80, H=900):
    def fit(a):
        s = H / a.shape[0]
        return cv2.cvtColor(cv2.resize(a, (max(1, int(a.shape[1] * s)), H),
                                       interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR), s
    a, sa = fit(_u8(src)); b, sb = fit(_u8(ref))
    canvas = np.hstack([a, b])
    off = a.shape[1]
    idx = np.linspace(0, len(s_pts) - 1, min(max_lines, len(s_pts))).astype(int) if len(s_pts) else []
    for i in idx:
        p = (int(s_pts[i, 0] * sa), int(s_pts[i, 1] * sa))
        q = (int(r_pts[i, 0] * sb) + off, int(r_pts[i, 1] * sb))
        cv2.line(canvas, p, q, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def run_baseline(src, ref, out_dir="outputs", ratio=0.8, thresh=3.0, max_side=4000):
    """src, ref: preprocessed Img objects. Returns a metrics dict."""
    os.makedirs(out_dir, exist_ok=True)

    longest = max(src.pixels.shape + ref.pixels.shape)
    scale = min(1.0, max_side / longest)          # one scale for both images
    pts_s, d_s = detect(src, scale)
    pts_r, d_r = detect(ref, scale)
    print(f"[baseline] detection scale {scale:.2f}; keypoints: source {len(pts_s)}, reference {len(pts_r)}")

    good = match_ratio(d_s, d_r, ratio)
    src_pts = pts_s[[m.queryIdx for m in good]].reshape(-1, 2)
    ref_pts = pts_r[[m.trainIdx for m in good]].reshape(-1, 2)
    print(f"[baseline] {len(good)} matches after ratio test")

    M, inl = estimate(src_pts, ref_pts, thresh)
    pixel_m = ref.meta.get("res_m") or src.meta.get("res_m")
    metrics = compute_metrics(src_pts, ref_pts, inl, M, src.pixels.shape,
                              pixel_m=src.meta.get("res_m"), n_candidates=len(good))

    # --- outputs -----------------------------------------------------------
    with open(os.path.join(out_dir, "matches.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_x", "src_y", "ref_x", "ref_y", "inlier"])
        for (sx, sy), (rx, ry), k in zip(src_pts, ref_pts, inl):
            w.writerow([f"{sx:.3f}", f"{sy:.3f}", f"{rx:.3f}", f"{ry:.3f}", int(k)])

    if M is not None:
        warped = warp(src, M, ref.pixels.shape)
        cv2.imwrite(os.path.join(out_dir, "registered_source.png"), warped)
        r8 = _u8(ref)
        # green = warped source, magenta/purple = reference only; grey = agreement
        overlay = np.dstack([r8, warped, r8])
        cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)
        np.savetxt(os.path.join(out_dir, "transform.txt"), M, fmt="%.8f")
        _draw_matches(src, ref, src_pts[inl], ref_pts[inl], os.path.join(out_dir, "matches.png"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
