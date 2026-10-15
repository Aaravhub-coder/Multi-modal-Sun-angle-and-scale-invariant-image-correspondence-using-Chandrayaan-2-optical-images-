"""
Loading images + metadata. Everything downstream works with the Img container.
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import tifffile

# Suffixes the dataset adds to the base product name. The XML label is named
# after the base product, e.g. <base>.xml, and describes the SOURCE image.
_SUFFIXES = (
    "_source_correctly_aligned_at_5m",
    "_reference_at_5m",
    "_source",
    "_reference",
)


@dataclass
class Img:
    pixels: np.ndarray                 # 2D float32
    mask: np.ndarray                   # 2D bool, True = valid pixel
    meta: dict = field(default_factory=dict)


def find_xml(path):
    """Return the XML label that sits next to `path`, or None."""
    folder = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    for suf in _SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    cand = os.path.join(folder, stem + ".xml")
    return cand if os.path.isfile(cand) else None


def read_xml_meta(xml_path):
    root = ET.parse(xml_path).getroot()

    def g(tag, cast=str):
        el = root.find(".//" + tag)
        if el is None or not el.text or not el.text.strip():
            return None
        try:
            return cast(el.text.strip())
        except ValueError:
            return None

    return {
        "xml": xml_path,
        "date": g("dop"),
        "res_m": g("Resolution_in_meter", float),
        "sun_az": g("Sun_azimuth_in_degree", float),
        "sun_el": g("Sun_elevation_in_degree", float),
        "projection": g("projection"),
    }


def load_image(path, role="source"):
    """
    Load a TIFF as a 2D float32 image plus a validity mask.

    role="source": also attach metadata from the XML label (sun angles, ...).
    role="reference": no XML is attached, because the label in the dataset
    describes the source image, not the reference.
    """
    arr = tifffile.imread(path)
    if arr.ndim == 3:                  # multi-band: keep the first band for now
        axis = int(np.argmin(arr.shape))
        arr = np.take(arr, 0, axis=axis)
    pix = arr.astype(np.float32)

    # 0 is used as nodata/border in these products; NaN/inf are never valid.
    mask = np.isfinite(pix) & (pix != 0)
    pix[~np.isfinite(pix)] = 0

    meta = {"path": path, "role": role, "dtype": str(arr.dtype), "shape": pix.shape}
    if role == "source":
        xml = find_xml(path)
        if xml:
            meta.update(read_xml_meta(xml))
    # The "_at_5m" OHRC files were resampled to 5 m/px; the XML still states the
    # original 0.25 m. Record the real pixel size of the file we loaded.
    if "_at_5m" in os.path.basename(path):
        meta["res_m_original"] = meta.get("res_m")
        meta["res_m"] = 5.0
    return Img(pix, mask, meta)