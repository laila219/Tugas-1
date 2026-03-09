"""
==========================================================================
PERCOBAAN 16: BLENDING DUA GAMBAR
==========================================================================
Program ini mempelajari cara mencampur/memadukan dua gambar dengan
bobot berbeda. Blending berguna untuk transisi, watermark, overlay.

Fungsi utama:
- cv2.addWeighted(src1, alpha, src2, beta, gamma) → dst
  dst = alpha*src1 + beta*src2 + gamma
  alpha + beta biasanya = 1.0 (tapi tidak wajib)
  gamma = brightness offset
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


def _ensure_blend_images():
    night = os.path.join(IMAGE_DIR, "foto_malam.jpg")
    day = os.path.join(IMAGE_DIR, "foto_siang.jpg")

    if not os.path.exists(night):
        print(f"[INFO] '{night}' tidak ditemukan — membuat placeholder malam.")
        h, w = 360, 480
        img_n = np.zeros((h, w, 3), dtype=np.uint8)
        # gradient night sky
        for y in range(h):
            val = int(20 + (y / h) * 60)
            img_n[y, :] = (val//2, val//3, val)  # bluish dark
        # moon
        cv2.circle(img_n, (w - 80, 80), 30, (220, 220, 200), -1)
        # stars
        for sx in range(30, w, 60):
            for sy in range(10, 120, 35):
                cv2.circle(img_n, (sx + (sy % 7), sy + (sx % 5)), 1, (255, 255, 255), -1)
        cv2.putText(img_n, "Foto Malam (placeholder)", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imwrite(night, img_n)

    if not os.path.exists(day):
        print(f"[INFO] '{day}' tidak ditemukan — membuat placeholder siang.")
        h, w = 360, 480
        img_d = np.full((h, w, 3), (200, 220, 255), dtype=np.uint8)
        # sun
        cv2.circle(img_d, (80, 80), 40, (0, 200, 255), -1)
        # simple hills
        pts = np.array([[0, 260], [120, 160], [250, 260]], np.int32)
        cv2.fillPoly(img_d, [pts], (80, 180, 90))
        pts2 = np.array([[200, 280], [340, 180], [480, 280]], np.int32)
        cv2.fillPoly(img_d, [pts2], (60, 160, 70))
        cv2.putText(img_d, "Foto Siang (placeholder)", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 1)
        cv2.imwrite(day, img_d)


_ensure_blend_images()

print("=" * 60)
print("PERCOBAAN 16: BLENDING DUA GAMBAR")
print("=" * 60)

# Membaca dua gambar
img1 = cv2.imread(os.path.join(IMAGE_DIR, "kotalama.jpeg"))
img2 = cv2.imread(os.path.join(IMAGE_DIR, "jalan.jpg"))

# Fallback jika gambar kedua tidak ada
if img1 is None: 
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

if img2 is None:
    # Membuat gambar kedua dari gradien jika tidak tersedia
    img2 = np.zeros_like(img1)
    h, w = img1.shape[:2]
    # Membuat gradien merah-biru
    for y in range(h):
        rasio = y / h
        img2[y, :, 0] = int(255 * rasio)       # Blue meningkat
        img2[y, :, 2] = int(255 * (1 - rasio))  # Red menurun

# Samakan ukuran kedua gambar
ukuran = (300, 300)
img1 = cv2.resize(img1, ukuran)
img2 = cv2.resize(img2, ukuran)

print(f"[INFO] Gambar 1: {img1.shape}")
print(f"[INFO] Gambar 2: {img2.shape}")

# ============================================================
# 1. Blending dasar dengan cv2.addWeighted
# ============================================================
print("\n--- 1. Blending Dasar ---")

# ★ KODE INTI ★ — Rumus: dst = alpha*img1 + beta*img2 + gamma
# alpha=1.0 → hanya img1 terlihat, alpha=0.0 → hanya img2, alpha=0.5 → 50-50 transparan
# alpha=0.7, beta=0.3 → gambar 1 lebih dominan
# gamma=0 → tanpa brightness offset
blend_70_30 = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
print("  Alpha=0.7, Beta=0.3 → Gambar 1 dominan")

# alpha=0.5, beta=0.5 → sama rata
blend_50_50 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
print("  Alpha=0.5, Beta=0.5 → Setara")

# alpha=0.3, beta=0.7 → gambar 2 lebih dominan
blend_30_70 = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)
print("  Alpha=0.3, Beta=0.7 → Gambar 2 dominan")

# ============================================================
# 2. Variasi alpha blending
# ============================================================
print("\n--- 2. Variasi Alpha ---")

# Membuat beberapa variasi alpha blending
alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
blend_variasi = []

for alpha in alpha_values:
    # Beta = 1 - alpha agar total bobot = 1
    beta = 1.0 - alpha
    # ★ KODE INTI ★ — Loop alpha 0.0→1.0 membuat efek FADE/TRANSISI bertahap
    # Digunakan dalam animasi, slide, atau transisi video
    # cv2.addWeighted: dst = alpha*src1 + beta*src2 + gamma
    blend = cv2.addWeighted(img1, alpha, img2, beta, 0)
    blend_variasi.append(blend)
    print(f"  Alpha={alpha:.2f}, Beta={beta:.2f}")

# ============================================================
# 3. Pengaruh parameter gamma
# ============================================================
print("\n--- 3. Pengaruh Gamma ---")

# Gamma menambahkan brightness konstan ke hasil blending
blend_g0 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
blend_g50 = cv2.addWeighted(img1, 0.5, img2, 0.5, 50)
blend_gn50 = cv2.addWeighted(img1, 0.5, img2, 0.5, -50)

print("  Gamma = 0   → normal")
print("  Gamma = 50  → lebih terang")
print("  Gamma = -50 → lebih gelap")

# ============================================================
# 4. Blending manual menggunakan NumPy
# ============================================================
print("\n--- 4. Blending Manual (NumPy) ---")

alpha_manual = 0.6
# Konversi ke float untuk perhitungan presisi
f1 = img1.astype(np.float64)
f2 = img2.astype(np.float64)

# Hitung blending manual: alpha*img1 + (1-alpha)*img2
blend_manual = (alpha_manual * f1 + (1 - alpha_manual) * f2)
# np.clip untuk memastikan nilai dalam rentang valid
# .astype(np.uint8) untuk konversi kembali ke format gambar
blend_manual = np.clip(blend_manual, 0, 255).astype(np.uint8)

# Bandingkan dengan cv2.addWeighted
blend_cv = cv2.addWeighted(img1, alpha_manual, img2, 1 - alpha_manual, 0)
# Hitung perbedaan (mungkin ada sedikit karena pembulatan)
diff = np.mean(cv2.absdiff(blend_manual, blend_cv))
print(f"  Rata-rata perbedaan manual vs cv2: {diff:.4f}")

# ============================================================
# 5. Gradient blending (transisi halus)
# ============================================================
print("\n--- 5. Gradient Blending ---")

h, w = img1.shape[:2]

# Membuat mask gradien horizontal (kiri ke kanan)
# np.linspace membuat array nilai 0.0 sampai 1.0
gradient_h = np.linspace(0, 1, w).reshape(1, w, 1)
# np.tile menduplikasi array sesuai dimensi yang diinginkan
gradient_h = np.tile(gradient_h, (h, 1, 3))

# Terapkan blending dengan bobot berbeda per kolom
f1 = img1.astype(np.float64)
f2 = img2.astype(np.float64)
blend_gradient = (f1 * (1 - gradient_h) + f2 * gradient_h)
blend_gradient = np.clip(blend_gradient, 0, 255).astype(np.uint8)
print("  Gradien horizontal: kiri=img1, kanan=img2")

# Membuat mask gradien vertikal
gradient_v = np.linspace(0, 1, h).reshape(h, 1, 1)
gradient_v = np.tile(gradient_v, (1, w, 3))
blend_gradient_v = (f1 * (1 - gradient_v) + f2 * gradient_v)
blend_gradient_v = np.clip(blend_gradient_v, 0, 255).astype(np.uint8)
print("  Gradien vertikal: atas=img1, bawah=img2")

# ============================================================
# 6. Circular blending (transisi radial)
# ============================================================
print("\n--- 6. Circular Blending ---")

# Membuat mask radial (lingkaran di tengah)
Y, X = np.ogrid[:h, :w]
# Menghitung jarak setiap piksel dari pusat
cx, cy = w // 2, h // 2
jarak = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
# Normalisasi jarak ke range 0-1
radius_maks = np.sqrt(cx ** 2 + cy ** 2)
mask_radial = np.clip(jarak / radius_maks, 0, 1)
mask_radial = np.stack([mask_radial] * 3, axis=2)

# Terapkan: tengah=img1, pinggir=img2
blend_radial = (f1 * (1 - mask_radial) + f2 * mask_radial)
blend_radial = np.clip(blend_radial, 0, 255).astype(np.uint8)
print("  Radial: tengah=img1, pinggir=img2")

# ============================================================
# 7. Visualisasi
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Sumber + blending dasar
axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar 1")
axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Gambar 2")
axes[0, 2].imshow(cv2.cvtColor(blend_70_30, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("α=0.7 β=0.3")
axes[0, 3].imshow(cv2.cvtColor(blend_50_50, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("α=0.5 β=0.5")

# Baris 2: Variasi alpha
for i, (alpha, blnd) in enumerate(zip(alpha_values[:4], blend_variasi[:4])):
    axes[1, i].imshow(cv2.cvtColor(blnd, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"α={alpha:.2f}")

# Baris 3: Transisi
axes[2, 0].imshow(cv2.cvtColor(blend_g50, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("Gamma=+50")
axes[2, 1].imshow(cv2.cvtColor(blend_gradient, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("Gradien Horizontal")
axes[2, 2].imshow(cv2.cvtColor(blend_gradient_v, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title("Gradien Vertikal")
axes[2, 3].imshow(cv2.cvtColor(blend_radial, cv2.COLOR_BGR2RGB))
axes[2, 3].set_title("Radial Blend")

for ax in axes.flat:
    ax.axis("off")

plt.suptitle("Percobaan 16: Blending Dua Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "16_blending_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")
plt.show()

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 16")
print("=" * 60)
print("  cv2.addWeighted(s1,α,s2,β,γ) → α*s1 + β*s2 + γ")
print("  α + β = 1.0 untuk bobot normal")
print("  γ > 0 lebih terang, γ < 0 lebih gelap")
print("  Gradient mask → transisi halus antar gambar")
print("  Radial mask   → efek vignette/spotlight")
print("=" * 60)
