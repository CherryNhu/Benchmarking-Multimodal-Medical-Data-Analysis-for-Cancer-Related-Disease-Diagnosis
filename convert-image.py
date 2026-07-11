#!/usr/bin/env python3
"""
Convert DICOM (CT/MRI) và SVS (WSI) sang PNG
Output: /mmlab_students/storageStudents/nguyenvd/Nhulnq/RunBaseline/image/
"""

import os
import numpy as np
from pathlib import Path

# ── Cài thư viện nếu chưa có ─────────────────────────────────────────────────
# pip install pydicom pillow openslide-python

try:
    import pydicom
except ImportError:
    os.system("pip install pydicom --quiet")
    import pydicom

try:
    from PIL import Image
except ImportError:
    os.system("pip install pillow --quiet")
    from PIL import Image

try:
    import openslide
except ImportError:
    os.system("pip install openslide-python --quiet")
    import openslide

# ── Thư mục output ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/mmlab_students/storageStudents/nguyenvd/Nhulnq/RunBaseline/image")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Danh sách file CT ─────────────────────────────────────────────────────────
CT_FILES = [
    # CT group 1
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/CT/1.3.6.1.4.1.14519.5.2.1.2692.1975.150408586611782069290121669635/1-01.dcm", "CT_1.1"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/CT/1.3.6.1.4.1.14519.5.2.1.2692.1975.150408586611782069290121669635/1-02.dcm", "CT_1.2"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/CT/1.3.6.1.4.1.14519.5.2.1.2692.1975.150408586611782069290121669635/1-03.dcm", "CT_1.3"),
    # CT group 2
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/CT/1.3.6.1.4.1.14519.5.2.1.2857.3304.239222061534769647580911799179/1-001.dcm", "CT_2.1"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/CT/1.3.6.1.4.1.14519.5.2.1.2857.3304.239222061534769647580911799179/1-002.dcm", "CT_2.2"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/CT/1.3.6.1.4.1.14519.5.2.1.2857.3304.239222061534769647580911799179/1-003.dcm", "CT_2.3"),
    # CT group 3
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/CT/1.3.6.1.4.1.14519.5.2.1.2857.3304.319582964259409452565516739579/1-01.dcm",  "CT_3.1"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/CT/1.3.6.1.4.1.14519.5.2.1.2857.3304.319582964259409452565516739579/1-03.dcm",  "CT_3.2"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/CT/1.3.6.1.4.1.14519.5.2.1.2857.3304.319582964259409452565516739579/1-06.dcm",  "CT_3.3"),
]

# ── Danh sách file MRI ────────────────────────────────────────────────────────
MRI_FILES = [
    # MRI group 1
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/MRI/1.3.6.1.4.1.14519.5.2.1.3320.3273.132642653502802060220279546165/1-01.dcm", "MRI_1.1"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/MRI/1.3.6.1.4.1.14519.5.2.1.3320.3273.132642653502802060220279546165/1-02.dcm", "MRI_1.2"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/MRI/1.3.6.1.4.1.14519.5.2.1.3320.3273.132642653502802060220279546165/1-03.dcm", "MRI_1.3"),
    # MRI group 2
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/MRI/1.3.6.1.4.1.14519.5.2.1.3320.3273.183670223112444579407107960119/1-01.dcm",  "MRI_2.1"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/MRI/1.3.6.1.4.1.14519.5.2.1.3320.3273.183670223112444579407107960119/1-02.dcm",  "MRI_2.2"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/MRI/1.3.6.1.4.1.14519.5.2.1.3320.3273.183670223112444579407107960119/1-03.dcm",  "MRI_2.3"),
    # MRI group 3
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/MRI/1.3.6.1.4.1.14519.5.2.1.4801.5885.189057890864241370752526114465/1-01.dcm",  "MRI_3.1"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/MRI/1.3.6.1.4.1.14519.5.2.1.4801.5885.189057890864241370752526114465/1-03.dcm",  "MRI_3.2"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/MRI_CT/mmist/CPTAC/MRI/1.3.6.1.4.1.14519.5.2.1.4801.5885.189057890864241370752526114465/1-06.dcm",  "MRI_3.3"),
]

# ── Danh sách file WSI ────────────────────────────────────────────────────────
WSI_FILES = [
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/453907.svs", "WSI_1"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/489645.svs", "WSI_2"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/490063.svs", "WSI_3"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/527050.svs", "WSI_4"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/565806.svs", "WSI_5"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/590582.svs", "WSI_6"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/627662.svs", "WSI_7"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/682887.svs", "WSI_8"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/718022.svs", "WSI_9"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/732453.svs", "WSI_10"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/763263.svs", "WSI_11"),
    ("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/ExtractFeature/dataset/WSI/935685.svs", "WSI_12")
]

# ── Hàm convert DICOM → PNG ───────────────────────────────────────────────────
def dicom_to_png(dcm_path, out_name):
    """
    Convert 1 file DICOM sang PNG.
    Áp dụng windowing nếu có WindowCenter/WindowWidth trong metadata.
    """
    out_path = OUTPUT_DIR / f"{out_name}.png"
    if out_path.exists():
        print(f"  [SKIP] {out_name}.png đã tồn tại")
        return

    try:
        ds = pydicom.dcmread(dcm_path)
        pixel = ds.pixel_array.astype(np.float32)

        # Áp dụng windowing nếu có (giúp ảnh CT/MRI trông đúng hơn)
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            wc = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) \
                 else float(ds.WindowCenter[0])
            ww = float(ds.WindowWidth)  if not isinstance(ds.WindowWidth,  pydicom.multival.MultiValue) \
                 else float(ds.WindowWidth[0])
            lo = wc - ww / 2
            hi = wc + ww / 2
            pixel = np.clip(pixel, lo, hi)
            pixel = (pixel - lo) / (hi - lo) * 255.0
        else:
            # Normalize min-max
            pmin, pmax = pixel.min(), pixel.max()
            if pmax > pmin:
                pixel = (pixel - pmin) / (pmax - pmin) * 255.0
            else:
                pixel = np.zeros_like(pixel)

        pixel = pixel.astype(np.uint8)
        img = Image.fromarray(pixel)

        # Đảm bảo mode L (grayscale) để lưu đúng
        if img.mode != 'L':
            img = img.convert('L')

        img.save(out_path)
        print(f"  [OK]   {out_name}.png  ({pixel.shape[1]}x{pixel.shape[0]}px)")

    except Exception as e:
        print(f"  [ERR]  {out_name}: {e}")


# ── Hàm convert SVS → PNG (thumbnail) ────────────────────────────────────────
def svs_to_png(svs_path, out_name, thumb_size=1024):
    """
    Convert WSI .svs sang PNG dạng thumbnail.
    thumb_size: kích thước cạnh dài của thumbnail (px)
    """
    out_path = OUTPUT_DIR / f"{out_name}.png"
    if out_path.exists():
        print(f"  [SKIP] {out_name}.png đã tồn tại")
        return

    try:
        slide = openslide.OpenSlide(svs_path)

        # Lấy kích thước gốc
        w, h = slide.dimensions
        print(f"  Kích thước gốc WSI: {w} x {h} px")

        # Tính thumbnail giữ tỷ lệ
        ratio = thumb_size / max(w, h)
        tw, th = int(w * ratio), int(h * ratio)

        thumbnail = slide.get_thumbnail((tw, th))
        thumbnail = thumbnail.convert("RGB")
        thumbnail.save(out_path)
        print(f"  [OK]   {out_name}.png  ({tw}x{th}px, thumbnail từ {w}x{h})")
        slide.close()

    except Exception as e:
        print(f"  [ERR]  {out_name}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 60)

    # CT
    print("\n[CT] Converting DICOM files...")
    for path, name in CT_FILES:
        if not Path(path).exists():
            print(f"  [NOT FOUND] {path}")
            continue
        dicom_to_png(path, name)

    # MRI
    print("\n[MRI] Converting DICOM files...")
    for path, name in MRI_FILES:
        if not Path(path).exists():
            print(f"  [NOT FOUND] {path}")
            continue
        dicom_to_png(path, name)

    # WSI
    print("\n[WSI] Converting SVS files (thumbnail)...")
    for path, name in WSI_FILES:
        if not Path(path).exists():
            print(f"  [NOT FOUND] {path}")
            continue
        svs_to_png(path, name, thumb_size=1024)

    print("\n" + "=" * 60)
    print("Done! Ảnh được lưu tại:")
    saved = list(OUTPUT_DIR.glob("*.png"))
    for f in sorted(saved):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")
    print(f"\nTổng: {len(saved)} ảnh")