"""
Open and inspect a Chandrayaan-2 image (OHRC / TMC / IIRS) or any raster.

Usage:
    python open_ohrc_image.py ../data/OHRXXD18CHO2359602NNNN24342131250969_V1_0_<XX>_reference_at_5m.tif //// Goes upto 04

Examples:
    python src/open_ohrc_image.py OHRXXD18CHO2470502NNNN25067152549847_V2_1_02_reference_at_5m.tif
    python src/open_ohrc_image.py data/data_for_sih_2026/IIRS/some_source.tif --band 10
    python src/open_ohrc_image.py my_image.tif --save preview.png --no-show
    python src/open_ohrc_image.py some_label.xml        # prints metadata (sun angles etc.)

<file_name> can be a full/relative path, or just a file name. If the path does
not exist, the script searches for that name under ./data (recursively) and
the current directory.

The window has zoom/pan from matplotlib's toolbar. The mouse position and the
pixel value are shown in the status bar.

Requires: numpy, matplotlib, tifffile (pip install numpy matplotlib tifffile).
Pillow is used as a fallback for PNG/JPG/BMP.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

TIFF_EXT = {".tif", ".tiff"}
PIL_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

# XML tags worth printing (these exist in the Chandrayaan-2 labels in the dataset)
XML_TAGS = [
    "job_id", "dop", "data_type", "image_width", "Resolution_in_meter",
    "Sun_azimuth_in_degree", "Sun_elevation_in_degree",
    "Solar_incidence_angle_in_degree", "projection",
    "topleft_latitude", "topleft_longitude",
    "topright_latitude", "topright_longitude",
    "bottomleft_latitude", "bottomleft_longitude",
    "bottomright_latitude", "bottomright_longitude",
]


# ---------------------------------------------------------------------------
# Locating the file
# ---------------------------------------------------------------------------
def find_file(name):
    """Return a path for `name`, searching ./data and . if it isn't found directly."""
    if os.path.isfile(name):
        return name
    base = os.path.basename(name)
    for root_dir in ("data", "."):
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _, files in os.walk(root_dir):
            if base in files:
                return os.path.join(dirpath, base)
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_array(path):
    """Load image pixels as a numpy array (2D, or 3D for multi-band)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in TIFF_EXT:
        import tifffile
        return tifffile.imread(path)
    if ext in PIL_EXT:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        return np.array(Image.open(path))
    # Unknown extension: try tifffile, then Pillow.
    try:
        import tifffile
        return tifffile.imread(path)
    except Exception:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        return np.array(Image.open(path))


def select_band(arr, band):
    """Reduce an array to a 2D image. Handles (H,W), (H,W,C) and (C,H,W)."""
    if arr.ndim == 2:
        return arr, None
    if arr.ndim == 3:
        # Guess the band axis: the smallest dimension is usually the bands.
        axis = int(np.argmin(arr.shape))
        n = arr.shape[axis]
        if band is None:
            band = 0
        if not 0 <= band < n:
            sys.exit(f"--band {band} out of range (file has {n} bands: 0..{n - 1})")
        return np.take(arr, band, axis=axis), (band, n)
    sys.exit(f"Unsupported array shape {arr.shape}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def stretch_limits(img, low, high):
    """Percentile contrast limits, ignoring NaN/inf and zero (nodata) pixels."""
    f = img.astype(np.float32, copy=False)
    valid = np.isfinite(f) & (f != 0)
    vals = f[valid]
    if vals.size == 0:
        vals = f[np.isfinite(f)]
    if vals.size == 0:
        return 0.0, 1.0
    # Sample for speed on huge images.
    if vals.size > 5_000_000:
        vals = vals[:: max(1, vals.size // 5_000_000)]
    lo, hi = np.percentile(vals, (low, high))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def print_info(path, arr, img, band_info, lo, hi):
    f = img.astype(np.float32, copy=False)
    finite = f[np.isfinite(f)]
    print(f"File       : {path}")
    print(f"Array shape: {arr.shape}   dtype: {arr.dtype}")
    if band_info:
        print(f"Band shown : {band_info[0]} of {band_info[1]} (use --band N to change)")
    print(f"Image size : {img.shape[1]} wide x {img.shape[0]} tall (pixels)")
    if finite.size:
        print(f"Value range: min={finite.min():.4g}  max={finite.max():.4g}  mean={finite.mean():.4g}")
    zeros = float((f == 0).mean()) * 100
    print(f"Zero pixels: {zeros:.1f}% (treated as nodata for contrast stretch)")
    print(f"Display    : stretched {lo:.4g} .. {hi:.4g}")


def print_geotiff_info(path):
    if os.path.splitext(path)[1].lower() not in TIFF_EXT:
        return
    try:
        import tifffile
        with tifffile.TiffFile(path) as t:
            tags = t.pages[0].tags
            for name in ("ModelPixelScaleTag", "ModelTiepointTag"):
                if name in tags:
                    print(f"{name}: {tags[name].value}")
            if "ModelPixelScaleTag" not in tags and "ModelTiepointTag" not in tags:
                print("GeoTIFF tags: none found (use the XML label for coordinates)")
    except Exception as e:
        print(f"(could not read TIFF tags: {e})")


def show_xml(path):
    """Print the important metadata from a Chandrayaan-2 XML label."""
    root = ET.parse(path).getroot()
    print(f"Metadata from {path}:")
    for tag in XML_TAGS:
        el = root.find(".//" + tag)
        if el is not None and el.text and el.text.strip():
            print(f"  {tag:34s} {el.text.strip()}")
    # Offer the matching image(s) next to the label, if present.
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.dirname(path) or "."
    matches = [f for f in os.listdir(folder)
               if f.startswith(stem) and f.lower().endswith(tuple(TIFF_EXT))]
    if matches:
        print("\nImage file(s) for this label:")
        for m in sorted(matches):
            print("  " + m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Open and inspect a Chandrayaan-2 image.")
    ap.add_argument("file_name", help="image file (path or just the name) or an XML label")
    ap.add_argument("--band", type=int, default=None, help="band index for multi-band files (default 0)")
    ap.add_argument("--low", type=float, default=1.0, help="low percentile for contrast stretch (default 1)")
    ap.add_argument("--high", type=float, default=99.0, help="high percentile for contrast stretch (default 99)")
    ap.add_argument("--no-stretch", action="store_true", help="show raw min..max range instead of a stretch")
    ap.add_argument("--gamma", type=float, default=1.0, help="gamma < 1 brightens dark images (e.g. 0.5)")
    ap.add_argument("--save", metavar="PNG", help="also save the displayed view to this PNG file")
    ap.add_argument("--no-show", action="store_true", help="do not open a window (useful with --save)")
    args = ap.parse_args()

    path = find_file(args.file_name)
    if path is None:
        sys.exit(f"File not found: {args.file_name} (also searched ./data and the current folder)")

    # XML label: just print metadata.
    if path.lower().endswith(".xml"):
        show_xml(path)
        return

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = load_array(path)
    img, band_info = select_band(arr, args.band)

    if args.no_stretch:
        f = img[np.isfinite(img)] if img.dtype.kind == "f" else img
        lo, hi = float(np.min(f)), float(np.max(f))
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = stretch_limits(img, args.low, args.high)

    print_info(path, arr, img, band_info, lo, hi)
    print_geotiff_info(path)

    # Apply stretch (+ optional gamma) explicitly so the gamma works on any dtype.
    disp = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
    if args.gamma != 1.0:
        disp = disp ** args.gamma

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{os.path.basename(path)}\n{img.shape[1]} x {img.shape[0]} px", fontsize=9)

    # Show original pixel values in the status bar when hovering.
    def fmt(x, y):
        c, r = int(round(x)), int(round(y))
        if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
            return f"x={c} y={r} value={img[r, c]:.5g}"
        return f"x={c} y={r}"
    ax.format_coord = fmt

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved view to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()