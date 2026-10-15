"""
Preprocessing (step 1 of the pipeline).

Goal: make a source and a reference that come from different sensors / sun
angles / units look comparable, so feature detection works on both.

  1. robust normalization   -> everything becomes float 0..1 (units disappear)
  2. local contrast (CLAHE) -> dark, low-sun images show their structure
"""
import os

import cv2
import numpy as np

from io_utils import Img, load_image


def preprocess(img, clahe=True, low=1.0, high=99.0):
    """Return a new Img with pixels in 0..1 (float32); mask is unchanged."""
    f, valid = img.pixels, img.mask
    vals = f[valid]
    lo, hi = (np.percentile(vals, (low, high)) if vals.size else (0.0, 1.0))
    if hi <= lo:
        hi = lo + 1e-6

    n = np.clip((f - lo) / (hi - lo), 0, 1)
    n[~valid] = 0

    if clahe:
        u8 = (n * 255).astype(np.uint8)
        u8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u8)
        n = u8.astype(np.float32) / 255.0
        n[~valid] = 0

    meta = dict(img.meta)
    meta["norm_range"] = (float(lo), float(hi))
    return Img(n.astype(np.float32), valid, meta)


def preprocess_pair(source_path, reference_path, out_dir="outputs", save_debug=True):
    """Load + preprocess both images. Returns (source, reference) as Img."""
    src = preprocess(load_image(source_path, role="source"))
    ref = preprocess(load_image(reference_path, role="reference"))

    for name, im in (("source", src), ("reference", ref)):
        print(f"[{name}] {im.meta['shape'][1]}x{im.meta['shape'][0]} px, "
              f"{100 * (1 - im.mask.mean()):.1f}% invalid, "
              f"raw range used for stretch: {im.meta['norm_range'][0]:.4g}..{im.meta['norm_range'][1]:.4g}")
    m = src.meta
    if m.get("sun_el") is not None:
        print(f"[source] sun elevation {m['sun_el']:.1f} deg, azimuth {m['sun_az']:.1f} deg, "
              f"{m['res_m']} m/px (from XML)")

    if save_debug:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "debug_source_preprocessed.png"), (src.pixels * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_dir, "debug_reference_preprocessed.png"), (ref.pixels * 255).astype(np.uint8))
    return src, ref