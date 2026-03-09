"""
==========================================================================
PERCOBAAN 14: PADDING DAN BORDER GAMBAR
==========================================================================
Program ini mempelajari cara menambahkan padding/border pada gambar.
Padding berguna untuk: menambah ukuran canvas, menghindari efek tepi
saat filtering, dan membuat frame foto.

Fungsi utama:
- cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
  borderType:
  - cv2.BORDER_CONSTANT  : Warna konstan (butuh parameter value)
  - cv2.BORDER_REFLECT   : Cermin tepi (abcba)
  - cv2.BORDER_REFLECT_101: Cermin tanpa duplikat tepi (abcb)
  - cv2.BORDER_REPLICATE : Duplikasi piksel tepi (aaa|abcd|ddd)
  - cv2.BORDER_WRAP      : Membungkus/repeat (bcd|abcd|abc)
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def _ensure_sample_images():
    fruits = os.path.join(IMAGE_DIR, "fruits.jpg")
    flower = os.path.join(IMAGE_DIR, "bunga.jpg")

    if not os.path.exists(fruits):
        print(f"[INFO] '{fruits}' tidak ditemukan — membuat placeholder.")
        h, w = 240, 320
        img_f = np.full((h, w, 3), (240, 230, 210), dtype=np.uint8)
        # draw some round fruits
        cv2.circle(img_f, (60, 80), 30, (0, 0, 200), -1)   # apple (red)
        cv2.circle(img_f, (160, 70), 28, (0, 200, 200), -1) # orange-ish
        cv2.circle(img_f, (260, 110), 26, (0, 180, 0), -1)  # lime
        cv2.putText(img_f, "Fruits (placeholder)", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 1)
        cv2.imwrite(fruits, img_f)

    if not os.path.exists(flower):
        print(f"[INFO] '{flower}' tidak ditemukan — membuat placeholder.")
        h, w = 240, 240
        img_fl = np.full((h, w, 3), (230, 240, 250), dtype=np.uint8)
        # draw simple flower: center + petals
        center = (w // 2, h // 2 - 10)
        cv2.circle(img_fl, center, 18, (10, 10, 10), -1)
        for angle in range(0, 360, 45):
            x = int(center[0] + 40 * np.cos(np.deg2rad(angle)))
            y = int(center[1] + 40 * np.sin(np.deg2rad(angle)))
            cv2.circle(img_fl, (x, y), 20, (200, 80, 200), -1)
        cv2.putText(img_fl, "Foto Bunga (placeholder)", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 1)
        cv2.imwrite(flower, img_fl)


_ensure_sample_images()

print("=" * 60)
print("PERCOBAAN 14: PADDING DAN BORDER GAMBAR")
print("=" * 60)

# Membaca gambar
img = cv2.imread(os.path.join(IMAGE_DIR, "kuda.jpg"))
if img is None:
    print("[ERROR] Gambar vokasi.jpg tidak ditemukan!")
    exit()

# Perkecil gambar agar padding terlihat jelas
img = cv2.resize(img, (200, 150))
print(f"[INFO] Ukuran gambar: {img.shape[1]}×{img.shape[0]}")

# Ukuran padding (piksel)
pad = 30

# ============================================================
# 1. BORDER_CONSTANT (warna konstan)
# ============================================================
print("\n--- 1. BORDER_CONSTANT ---")

# ★ KODE INTI ★ — Perbedaan tipe border:
# CONSTANT   → piksel tetap (warna solid, cocok untuk frame)
# REFLECT    → efek cermin inkl. tepi (fedcba|abcd|dcba)
# REFLECT_101→ cermin tanpa duplikat tepi (gfedcb|abcd|bcde) — default OpenCV
# REPLICATE  → piksel tepi diulang (aaaa|abcd|dddd)
# WRAP       → gambar di-tile/berulang (bcd|abcd|abc)

# Padding dengan warna hitam
img_const_black = cv2.copyMakeBorder(
    img, pad, pad, pad, pad,
    cv2.BORDER_CONSTANT,
    value=[0, 0, 0]  # Warna border BGR (hitam)
)
print(f"  Hitam: {img_const_black.shape[1]}×{img_const_black.shape[0]}")

# Padding dengan warna merah
img_const_red = cv2.copyMakeBorder(
    img, pad, pad, pad, pad,
    cv2.BORDER_CONSTANT,
    value=[0, 0, 255]  # BGR: merah
)

# Padding dengan warna putih
img_const_white = cv2.copyMakeBorder(
    img, pad, pad, pad, pad,
    cv2.BORDER_CONSTANT,
    value=[255, 255, 255]  # BGR: putih
)
print("  Merah dan putih juga dibuat")

# ============================================================
# 2. BORDER_REFLECT (cermin tepi)
# ============================================================
print("\n--- 2. BORDER_REFLECT ---")

# Padding dengan cermin (mencerminkan piksel tepi)
# Pola: fedcba|abcdefgh|hgfedcb
img_reflect = cv2.copyMakeBorder(
    img, pad, pad, pad, pad,
    cv2.BORDER_REFLECT
)
print("  REFLECT: piksel dicerminkan di tepi")

# ============================================================
# 3. BORDER_REFLECT_101 (cermin tanpa duplikat)
# ============================================================
print("\n--- 3. BORDER_REFLECT_101 ---")

# Sama seperti REFLECT tapi NOT duplikasi piksel tepian
# Pola: gfedcb|abcdefgh|gfedcba
img_reflect101 = cv2.copyMakeBorder(
    img, pad, pad, pad, pad,
    cv2.BORDER_REFLECT_101
)
print("  REFLECT_101: cermin tanpa duplikat tepi")

# ============================================================
# 4. BORDER_REPLICATE (duplikasi tepi)
# ============================================================
print("\n--- 4. BORDER_REPLICATE ---")

# Menduplikasi piksel paling tepi
# Pola: aaaa|abcdefgh|hhhh
img_replicate = cv2.copyMakeBorder(
    img, pad, pad, pad, pad,
    cv2.BORDER_REPLICATE
)
print("  REPLICATE: piksel tepi diduplikasi")

# ============================================================
# 5. BORDER_WRAP (pembungkusan)
# ============================================================
print("\n--- 5. BORDER_WRAP ---")

# Membungkus gambar (opposite side)
# Pola: efgh|abcdefgh|abcd
img_wrap = cv2.copyMakeBorder(
    img, pad, pad, pad, pad,
    cv2.BORDER_WRAP
)
print("  WRAP: gambar dibungkus dari sisi berlawanan")

# ============================================================
# 6. Padding asimetris
# ============================================================
print("\n--- 6. Padding Asimetris ---")

# Padding berbeda untuk setiap sisi: top=10, bottom=50, left=20, right=40
img_asym = cv2.copyMakeBorder(
    img, 10, 50, 20, 40,
    cv2.BORDER_CONSTANT,
    value=[128, 128, 128]  # Abu-abu
)
print(f"  Asimetris (10,50,20,40): {img_asym.shape[1]}×{img_asym.shape[0]}")

# ============================================================
# 7. Aplikasi: Frame foto dekoratif
# ============================================================
print("\n--- 7. Frame Foto Dekoratif ---")

def buat_frame_foto(img, lebar_frame=20, warna_luar=(50, 50, 50), warna_dalam=(200, 200, 200)):
    """Membuat frame foto dekoratif dengan dua lapisan border."""
    # Lapisan dalam: border tipis
    img_frame = cv2.copyMakeBorder(
        img, 3, 3, 3, 3,
        cv2.BORDER_CONSTANT,
        value=warna_dalam
    )
    # Lapisan luar: border tebal
    img_frame = cv2.copyMakeBorder(
        img_frame, lebar_frame, lebar_frame, lebar_frame, lebar_frame,
        cv2.BORDER_CONSTANT,
        value=warna_luar
    )
    return img_frame

# Membuat frame foto dekoratif
img_frame = buat_frame_foto(img, lebar_frame=25,
                            warna_luar=(40, 40, 40),
                            warna_dalam=(220, 220, 220))
print(f"  Frame foto: {img_frame.shape[1]}×{img_frame.shape[0]}")

# ============================================================
# 8. Aplikasi: Padding untuk square crop
# ============================================================
print("\n--- 8. Padding untuk Square ---")

h, w = img.shape[:2]
# Menentukan sisi terbesar
maks = max(h, w)
# Menghitung padding yang diperlukan untuk setiap sisi
pad_top = (maks - h) // 2
pad_bottom = maks - h - pad_top
pad_left = (maks - w) // 2
pad_right = maks - w - pad_left

# Membuat gambar persegi dengan padding hitam
img_square = cv2.copyMakeBorder(
    img, pad_top, pad_bottom, pad_left, pad_right,
    cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
)
print(f"  Square: {img_square.shape[1]}×{img_square.shape[0]}")

# ============================================================
# 9. Visualisasi semua jenis border
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

daftar = [
    (img, "Original"),
    (img_const_black, "CONSTANT (Hitam)"),
    (img_const_red, "CONSTANT (Merah)"),
    (img_reflect, "REFLECT"),
    (img_reflect101, "REFLECT_101"),
    (img_replicate, "REPLICATE"),
    (img_wrap, "WRAP"),
    (img_frame, "Frame Foto")
]

for idx, (gambar, judul) in enumerate(daftar):
    baris = idx // 4
    kolom = idx % 4
    axes[baris, kolom].imshow(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
    axes[baris, kolom].set_title(judul, fontsize=12)
    axes[baris, kolom].axis("off")

plt.suptitle("Percobaan 14: Padding & Border Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "14_padding_border_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")
plt.show()

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 14")
print("=" * 60)
print("  BORDER_CONSTANT    → Warna konstan (value=BGR)")
print("  BORDER_REFLECT     → Cermin di tepi")
print("  BORDER_REFLECT_101 → Cermin tanpa duplikat")
print("  BORDER_REPLICATE   → Duplikasi piksel tepi")
print("  BORDER_WRAP        → Bungkus dari sisi berlawanan")
print("=" * 60)
