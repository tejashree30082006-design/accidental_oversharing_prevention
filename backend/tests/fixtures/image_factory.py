"""
Test fixture generator for metadata, QR, and barcode service tests.

Generates synthetic test images programmatically so that the test suite
has no dependency on external image files:

  • image_with_gps_exif   – JPEG with GPS + DateTimeOriginal EXIF
  • image_without_exif    – Plain PNG with no EXIF whatsoever
  • image_with_qr         – PNG containing a rendered QR code
  • image_without_qr      – Solid-colour PNG with no QR code
  • image_with_barcode    – PNG containing a Code 128 barcode
  • image_without_barcode – Solid-colour PNG with no barcode

All returned values are raw bytes (ready to pass straight into the
service under test).

IMPORTANT: These helpers write nothing to disk.
"""
from __future__ import annotations

import io
import struct
import zlib
from typing import Optional

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


# ── Helper: reverse the TAGS lookup ───────────────────────────────────────
_NAME_TO_TAG = {v: k for k, v in TAGS.items()}


# ────────────────────────────────────────────────────────────────────────────
# EXIF helpers
# ────────────────────────────────────────────────────────────────────────────

def _rational(numerator: int, denominator: int) -> tuple:
    """Return a rational as a (numerator, denominator) tuple."""
    return (numerator, denominator)


def _make_gps_ifd() -> dict:
    """
    Build a synthetic GPS IFD dict compatible with piexif.

    GPS latitude: 37° 46' 26.4" N  → San Francisco (approximate)
    GPS longitude: 122° 25' 52.8" W
    Altitude: 15 m
    """
    return {
        1: b"N",                                     # GPSLatitudeRef
        2: (                                         # GPSLatitude (DMS rationals)
            _rational(37, 1),
            _rational(46, 1),
            _rational(264, 10),
        ),
        3: b"W",                                     # GPSLongitudeRef
        4: (                                         # GPSLongitude (DMS rationals)
            _rational(122, 1),
            _rational(25, 1),
            _rational(528, 10),
        ),
        5: 0,                                        # GPSAltitudeRef (above sea level)
        6: _rational(15, 1),                         # GPSAltitude
    }


def make_image_with_gps_exif() -> bytes:
    """
    Return JPEG bytes for a small 100×100 image that carries:
      - GPSInfo with synthetic San Francisco coordinates
      - DateTimeOriginal
      - Make (camera make)
      - Model (camera model)
      - Software
      - Artist (author)
      - Copyright

    Uses piexif if available, otherwise falls back to Pillow's built-in
    EXIF writer (limited, but sufficient for GPS presence detection tests).
    """
    img = Image.new("RGB", (100, 100), color=(120, 80, 60))

    try:
        import piexif  # type: ignore

        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLatitude: (
                (37, 1), (46, 1), (264, 10)
            ),
            piexif.GPSIFD.GPSLongitudeRef: b"W",
            piexif.GPSIFD.GPSLongitude: (
                (122, 1), (25, 1), (528, 10)
            ),
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: (15, 1),
        }
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make: b"TestCamera",
                piexif.ImageIFD.Model: b"Model-X",
                piexif.ImageIFD.Software: b"TestSuite 1.0",
                piexif.ImageIFD.Artist: b"Test Author",
                piexif.ImageIFD.Copyright: b"(c) Test 2024",
            },
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: b"2024:01:15 12:00:00",
            },
            "GPS": gps_ifd,
        }
        exif_bytes = piexif.dump(exif_dict)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif_bytes)
        return buf.getvalue()

    except ImportError:
        # piexif not installed: use a minimal hand-crafted EXIF approach
        # that embeds a real GPSInfo tag using Pillow's Exif helper.
        exif = img.getexif()
        # Tag 34853 = GPSInfo IFD pointer — we write a placeholder value.
        # Pillow cannot write a full GPS IFD without piexif, so we inject
        # the GPS IFD via the getexif() interface using the internal
        # _getexif / info mechanism.
        # For test purposes we embed GPS using a known raw EXIF byte sequence.
        exif[34853] = {
            1: "N",
            2: ((37, 1), (46, 1), (264, 10)),
            3: "W",
            4: ((122, 1), (25, 1), (528, 10)),
        }
        exif[306] = "2024:01:15 12:00:00"   # DateTime
        exif[271] = "TestCamera"             # Make
        exif[272] = "Model-X"               # Model
        exif[305] = "TestSuite 1.0"         # Software
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        return buf.getvalue()


def make_image_without_exif() -> bytes:
    """Return PNG bytes for a plain 100×100 image with absolutely no EXIF."""
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ────────────────────────────────────────────────────────────────────────────
# QR code helpers
# ────────────────────────────────────────────────────────────────────────────

def make_image_with_qr(data: str = "https://example.com/test-qr") -> bytes:
    """
    Return PNG bytes containing a rendered QR code encoding ``data``.

    Attempts to use the ``qrcode`` library first; falls back to a
    hard-coded 21×21 QR module pattern that OpenCV can decode.
    """
    # ── Attempt qrcode library ────────────────────────────────────────────
    try:
        import qrcode  # type: ignore

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        return buf.getvalue()

    except ImportError:
        pass

    # ── Fallback: generate QR via OpenCV ─────────────────────────────────
    try:
        import cv2

        # Create a white canvas, draw QR with OpenCV's built-in generator
        encoder = cv2.QRCodeEncoder.create()
        qr_matrix = encoder.encode(data)
        # qr_matrix is a binary image; scale it up
        qr_large = cv2.resize(
            qr_matrix, (210, 210), interpolation=cv2.INTER_NEAREST
        )
        _, png_bytes = cv2.imencode(".png", qr_large)
        return bytes(png_bytes)

    except Exception:
        pass

    # ── Last resort: use segno ────────────────────────────────────────────
    try:
        import segno  # type: ignore

        qr = segno.make_qr(data)
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=10)
        return buf.getvalue()

    except ImportError:
        raise RuntimeError(
            "No QR library found. Install one of: qrcode, segno. "
            "Alternatively install opencv-python."
        )


def make_image_without_qr() -> bytes:
    """Return PNG bytes for a solid-colour image containing no QR code."""
    img = Image.new("RGB", (200, 200), color=(180, 210, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ────────────────────────────────────────────────────────────────────────────
# Barcode helpers
# ────────────────────────────────────────────────────────────────────────────

def make_image_with_barcode(data: str = "1234567890") -> bytes:
    """
    Return PNG bytes containing a Code 128 barcode encoding ``data``.

    Attempts the ``python-barcode`` library first; falls back to a
    minimal synthetic Code 128 rendering that pyzbar can decode.
    """
    # ── Attempt python-barcode library ───────────────────────────────────
    try:
        import barcode  # type: ignore
        from barcode.writer import ImageWriter  # type: ignore

        code128 = barcode.get("code128", data, writer=ImageWriter())
        buf = io.BytesIO()
        code128.write(buf)
        buf.seek(0)
        return buf.getvalue()

    except ImportError:
        pass

    # ── Fallback: attempt treepoem ────────────────────────────────────────
    try:
        import treepoem  # type: ignore

        image = treepoem.generate_barcode(barcode_type="code128", data=data)
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    except ImportError:
        pass

    # ── Last resort: minimal Code 128 renderer ───────────────────────────
    return _render_code128_minimal(data)


def _render_code128_minimal(data: str) -> bytes:
    """
    Render a valid Code 128B barcode to PNG bytes.

    This is a pure-Python implementation sufficient for pyzbar to decode.
    It encodes ASCII printable characters using Code 128 subset B.
    """
    # Code 128B character set: value 0 = space (0x20) … 94 = ~ (0x7E)
    CODE128B_START = 104
    CODE128_STOP = 106
    CODE128_QUIET = 10  # quiet zone bar units

    # Module widths for each Code 128 value (indices 0-106)
    # Each code word is 3 bars + 3 spaces = 6 elements of width 1-4
    _CODE128_PATTERNS = [
        "212222", "222122", "222221", "121223", "121322", "131222",
        "122213", "122312", "132212", "221213", "221312", "231212",
        "112232", "122132", "122231", "113222", "123122", "123221",
        "223211", "221132", "221231", "213212", "223112", "312131",
        "311222", "321122", "321221", "312212", "322112", "322211",
        "212123", "212321", "232121", "111323", "131123", "131321",
        "112313", "132113", "132311", "211313", "231113", "231311",
        "112133", "112331", "132131", "113123", "113321", "133121",
        "313121", "211331", "231131", "213113", "213311", "213131",
        "311123", "311321", "331121", "312113", "312311", "332111",
        "314111", "221411", "431111", "111224", "111422", "121124",
        "121421", "141122", "141221", "112214", "112412", "122114",
        "122411", "142112", "142211", "241211", "221114", "413111",
        "241112", "134111", "111242", "121142", "121241", "114212",
        "124112", "124211", "411212", "421112", "421211", "212141",
        "214121", "412121", "111143", "111341", "131141", "114113",
        "114311", "411113", "411311", "113141", "114131", "311141",
        "411131", "211412", "211214", "211232", "2331112",  # STOP (index 106)
    ]

    # Build symbol list
    symbols = [CODE128B_START]
    for ch in data:
        code_val = ord(ch) - 32  # Code 128B: SP=0, A=33, …
        if not (0 <= code_val <= 94):
            raise ValueError(f"Character {ch!r} outside Code 128B range.")
        symbols.append(code_val)

    # Compute check character
    check = CODE128B_START
    for i, sym in enumerate(symbols[1:], start=1):
        check += i * sym
    check %= 103
    symbols.append(check)
    symbols.append(CODE128_STOP)

    # Build bar/space widths
    bar_units: list = []
    for i, sym in enumerate(symbols):
        pattern = _CODE128_PATTERNS[sym]
        bar_units.extend([int(c) for c in pattern])

    # Render to image
    unit_px = 3       # pixels per module
    height_px = 100
    quiet_px = CODE128_QUIET * unit_px

    total_width = quiet_px + sum(bar_units) * unit_px + quiet_px
    img_array = np.ones((height_px, total_width), dtype=np.uint8) * 255  # white

    x = quiet_px
    for i, width in enumerate(bar_units):
        pixel_width = width * unit_px
        if i % 2 == 0:  # even indices = bar (black)
            img_array[:, x: x + pixel_width] = 0
        x += pixel_width

    pil_img = Image.fromarray(img_array).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def make_image_without_barcode() -> bytes:
    """Return PNG bytes for a solid-colour image containing no barcode."""
    img = Image.new("RGB", (200, 80), color=(240, 230, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
