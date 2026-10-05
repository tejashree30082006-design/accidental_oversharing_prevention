"""Sample Image Generator.

Generates benchmark test images in sample_images/ directory to test:
1. Clean image
2. Image containing phone number
3. Image containing email
4. Image containing QR code
5. Image containing barcode
6. Image containing GPS metadata
7. Image containing multiple risks
"""

import io
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ExifTags

SAMPLE_DIR = BASE_DIR / "sample_images"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def generate_clean_image() -> Path:
    """1. Clean image without any privacy risks."""
    img = Image.new("RGB", (400, 300), color=(70, 130, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 250], outline=(255, 255, 255), width=3)
    path = SAMPLE_DIR / "clean_sample.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


def generate_phone_image() -> Path:
    """2. Image containing phone number text."""
    img = Image.new("RGB", (500, 200), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((30, 80), "Call me: +1 (555) 839-2041", fill=(0, 0, 0))
    path = SAMPLE_DIR / "phone_sample.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


def generate_email_image() -> Path:
    """3. Image containing email text."""
    img = Image.new("RGB", (500, 200), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((30, 80), "Send contract to: finance-dept@company.com", fill=(0, 0, 0))
    path = SAMPLE_DIR / "email_sample.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


def generate_qr_image() -> Path:
    """4. Image containing an authentic QR code."""
    enc = cv2.QRCodeEncoder.create()
    qr = enc.encode("https://company-intranet.local/vpn-token-8921")
    qr_resized = cv2.resize(qr, (180, 180), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
    canvas[60:240, 60:240] = cv2.cvtColor(qr_resized, cv2.COLOR_GRAY2BGR)

    pil_img = Image.fromarray(canvas)
    path = SAMPLE_DIR / "qr_sample.jpg"
    pil_img.save(path, format="JPEG", quality=95)
    return path


def generate_barcode_image() -> Path:
    """5. Image containing a barcode pattern."""
    canvas = np.full((200, 400, 3), 255, dtype=np.uint8)
    # Draw vertical stripes simulating a 1D barcode
    pattern = [2, 4, 1, 3, 2, 5, 2, 1, 4, 2, 3, 1, 2, 4, 3, 2, 1, 5, 2, 3]
    curr_x = 50
    for width in pattern * 3:
        cv2.rectangle(canvas, (curr_x, 40), (curr_x + width, 140), (0, 0, 0), -1)
        curr_x += width + 3

    pil_img = Image.fromarray(canvas)
    path = SAMPLE_DIR / "barcode_sample.jpg"
    pil_img.save(path, format="JPEG", quality=95)
    return path


def generate_gps_image() -> Path:
    """6. Image containing embedded EXIF GPS coordinates."""
    img = Image.new("RGB", (400, 300), color=(100, 180, 120))
    exif = img.getexif()
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_ifd[1] = "N"
    gps_ifd[2] = (37.0, 46.0, 30.0)  # 37° 46' 30" N
    gps_ifd[3] = "W"
    gps_ifd[4] = (122.0, 25.0, 5.0)  # 122° 25' 5" W

    path = SAMPLE_DIR / "gps_sample.jpg"
    img.save(path, format="JPEG", exif=exif, quality=95)
    return path


def generate_multi_risk_image() -> Path:
    """7. Image containing multiple privacy risks (QR code + GPS metadata)."""
    enc = cv2.QRCodeEncoder.create()
    qr = enc.encode("WIFI:T:WPA;S:HomeNetwork;P:SuperSecretPass;;")
    qr_resized = cv2.resize(qr, (160, 160), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((350, 350, 3), 255, dtype=np.uint8)
    canvas[95:255, 95:255] = cv2.cvtColor(qr_resized, cv2.COLOR_GRAY2BGR)

    pil_img = Image.fromarray(canvas)

    # Attach GPS EXIF metadata
    exif = pil_img.getexif()
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_ifd[1] = "N"
    gps_ifd[2] = (40.0, 42.0, 46.0)

    path = SAMPLE_DIR / "multi_risk_sample.jpg"
    pil_img.save(path, format="JPEG", exif=exif, quality=95)
    return path


def generate_all_samples():
    """Generate all 7 benchmark images."""
    print("Generating sample test images in", SAMPLE_DIR)
    generate_clean_image()
    generate_phone_image()
    generate_email_image()
    generate_qr_image()
    generate_barcode_image()
    generate_gps_image()
    generate_multi_risk_image()
    print("All sample images generated successfully!")


if __name__ == "__main__":
    generate_all_samples()
