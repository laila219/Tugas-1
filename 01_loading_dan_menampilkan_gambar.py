"""
konsep dasar pengolahan citra digital menggunakan Python dan OpenCV,
mulai dari proses membaca gambar, manipulasi piksel, hingga analisis histogram 
dan penyimpanan gambar.
==========================================================================
PERCOBAAN 1: LOADING DAN MENAMPILKAN GAMBAR
==========================================================================
Program ini mempelajari cara memuat (load) gambar dari file menggunakan
OpenCV dan menampilkannya di jendela GUI atau menyimpan hasilnya.

Fungsi utama yang dipelajari:
- cv2.imread()    : Membaca/memuat gambar dari file
- cv2.imshow()    : Menampilkan gambar di jendela GUI
- cv2.waitKey()   : Menunggu input keyboard
- cv2.destroyAllWindows() : Menutup semua jendela GUI

Catatan: Jika jendela GUI tidak tersedia (server/remote), gunakan
matplotlib atau simpan hasil langsung ke file.
==========================================================================
"""

# Standard libraries
import os
import sys

# Mengimpor library OpenCV untuk pemrosesan gambar
try:
    import cv2
except Exception:
    print("[ERROR] Modul 'cv2' (OpenCV) tidak ditemukan. Install dengan: pip install opencv-python")
    sys.exit(1)

# Mengimpor library NumPy untuk operasi array
try:
    import numpy as np
except Exception:
    print("[ERROR] Modul 'numpy' tidak ditemukan. Install dengan: pip install numpy")
    sys.exit(1)

# Mengimpor matplotlib untuk alternatif menampilkan gambar
try:
    import matplotlib.pyplot as plt
except Exception:
    print("[ERROR] Modul 'matplotlib' tidak ditemukan. Install dengan: pip install matplotlib")
    sys.exit(1)

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Membuat folder image jika belum ada
os.makedirs(IMAGE_DIR, exist_ok=True)

# Jika gambar contoh tidak ada, buat placeholder sehingga skrip dapat dijalankan
def _ensure_sample_images():
    cat_path = os.path.join(IMAGE_DIR, "kucing.jpg")
    flower_path = os.path.join(IMAGE_DIR, "bunga.jpg")
    # Create a higher-quality synthetic cat (gradient background, textured fur, silhouette)
    if not os.path.exists(cat_path):
        print(f"[INFO] '{cat_path}' tidak ditemukan — membuat placeholder berkualitas.")
        h, w = 480, 640
        # vertical gradient background
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            t = y / (h - 1)
            color = np.array([220 * (1 - t) + 100 * t, 180 * (1 - t) + 70 * t, 160 * (1 - t) + 30 * t])
            bg[y, :] = color

        # textured noise (smoothed) to simulate fur/background texture
        noise = (np.random.randn(h, w) * 25 + 128).astype(np.uint8)
        noise = cv2.GaussianBlur(noise, (31, 31), 0)
        noise_bgr = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        base = cv2.addWeighted(bg, 0.85, noise_bgr, 0.15, 0)

        # draw simple stylized cat silhouette (body, head, ears, tail)
        cat = base.copy()
        # body
        cv2.ellipse(cat, (w // 2, h // 2 + 40), (140, 90), 0, 0, 360, (30, 30, 40), -1)
        # head
        cv2.circle(cat, (w // 2 - 120, h // 2 - 10), 50, (30, 30, 40), -1)
        # ears
        pts1 = np.array([[w//2-160, h//2-40], [w//2-140, h//2-80], [w//2-110, h//2-40]], np.int32)
        pts2 = np.array([[w//2-80, h//2-40], [w//2-60, h//2-80], [w//2-30, h//2-40]], np.int32)
        cv2.fillPoly(cat, [pts1], (30,30,40))
        cv2.fillPoly(cat, [pts2], (30,30,40))
        # tail
        cv2.ellipse(cat, (w//2 + 160, h//2 + 10), (20, 80), -30, 0, 360, (30,30,40), -1)

        # eyes (glow)
        cv2.ellipse(cat, (w//2 - 140, h//2 - 20), (10, 6), 0, 0, 360, (200, 220, 50), -1)
        cv2.ellipse(cat, (w//2 - 100, h//2 - 20), (10, 6), 0, 0, 360, (200, 220, 50), -1)

        # subtle highlight on silhouette
        highlight = cat.copy()
        cv2.ellipse(highlight, (w // 2, h // 2 + 40), (140, 90), 0, 0, 180, (60, 60, 80), 6)
        cat = cv2.addWeighted(cat, 0.95, highlight, 0.05, 0)

        cv2.putText(cat, "Foto Kucing (placeholder)", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
        cv2.imwrite(cat_path, cat)

        # saturated variant
        hsv = cv2.cvtColor(cat, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
        sat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        cv2.putText(sat, "Saturated", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imwrite(os.path.join(IMAGE_DIR, "foto_kucing_sat.jpg"), sat)

        # rotated variant (45 deg)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
        rot = cv2.warpAffine(cat, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        cv2.putText(rot, "Rot45", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imwrite(os.path.join(IMAGE_DIR, "foto_kucing_rot45.jpg"), rot)

    # Create a higher-quality synthetic flower (petals, layered colors)
    if not os.path.exists(flower_path):
        print(f"[INFO] '{flower_path}' tidak ditemukan — membuat placeholder berkualitas.")
        h, w = 480, 640
        # pastel radial background
        y_idxs, x_idxs = np.indices((h, w))
        cx, cy = w // 2, h // 2
        r = np.sqrt((x_idxs - cx) ** 2 + (y_idxs - cy) ** 2)
        maxr = r.max()
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:, :, 0] = np.clip(200 + (r / maxr) * 40, 0, 255)
        bg[:, :, 1] = np.clip(180 + (r / maxr) * 60, 0, 255)
        bg[:, :, 2] = np.clip(220 + (r / maxr) * 20, 0, 255)

        flower = bg.copy()
        # draw layered petals using rotated ellipses
        center = (cx, cy - 20)
        petals = 12
        for i in range(petals):
            ang = i * (360 / petals)
            color = (180 + (i % 3) * 20, 80 + (i % 5) * 15, 200 - (i % 4) * 20)
            axes_len = (40 + (i % 3) * 10, 120 - (i % 4) * 8)
            Mpetal = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.ellipse(Mpetal, center, axes_len, ang, 0, 360, color, -1)
            flower = cv2.addWeighted(flower, 1.0, Mpetal, 0.6, 0)

        # center
        cv2.circle(flower, (cx, cy - 20), 36, (40, 30, 20), -1)
        cv2.circle(flower, (cx, cy - 20), 18, (230, 200, 60), -1)

        cv2.putText(flower, "Foto Bunga (placeholder)", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240,240,240), 2)
        cv2.imwrite(flower_path, flower)

        # tinted variant (hue shift)
        hsvf = cv2.cvtColor(flower, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsvf[:, :, 0] = (hsvf[:, :, 0] + 25) % 180
        tinted = cv2.cvtColor(hsvf.astype(np.uint8), cv2.COLOR_HSV2BGR)
        cv2.putText(tinted, "Tinted", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imwrite(os.path.join(IMAGE_DIR, "foto_bunga2_tint.jpg"), tinted)

        # rotated variant (-30 deg)
        M2 = cv2.getRotationMatrix2D((w // 2, h // 2), -30, 1.0)
        rotf = cv2.warpAffine(flower, M2, (w, h), borderMode=cv2.BORDER_REFLECT)
        cv2.putText(rotf, "Rot-30", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imwrite(os.path.join(IMAGE_DIR, "foto_bunga2_rot-30.jpg"), rotf)

_ensure_sample_images()

# ============================================================
# 1. Membaca gambar dalam mode warna (BGR)
# cv2.imread(path, flag) - flag default = cv2.IMREAD_COLOR
# ============================================================
print("=" * 60)
print("PERCOBAAN 1: LOADING DAN MENAMPILKAN GAMBAR")
print("=" * 60)

# Membaca foto dalam mode warna penuh (3 channel: Blue, Green, Red)
# cv2.IMREAD_COLOR (1) = baca sebagai gambar berwarna, abaikan transparansi
# ★ KODE INTI ★ — Hasil: array NumPy (h, w, 3) dalam format BGR. Jika file tidak ada → None
img_color = cv2.imread(os.path.join(IMAGE_DIR, "alam.jpg"), cv2.IMREAD_COLOR)

# Memeriksa apakah gambar berhasil dimuat (tidak None)
if img_color is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi bahwa gambar berhasil dimuat
print(f"[INFO] Gambar berwarna berhasil dimuat.")
print(f"  - Dimensi: {img_color.shape}")  # (height, width, channels)
print(f"  - Tipe data: {img_color.dtype}")  # uint8 (0-255)

# ============================================================
# 2. Membaca gambar dalam mode grayscale (1 channel)
# cv2.IMREAD_GRAYSCALE (0) = konversi ke abu-abu saat loading
# ============================================================

# Membaca foto yang sama dalam mode grayscale (abu-abu)
img_gray = cv2.imread(os.path.join(IMAGE_DIR, "alam.jpg"), cv2.IMREAD_GRAYSCALE)

# Menampilkan informasi gambar grayscale
print(f"\n[INFO] Gambar grayscale berhasil dimuat.")
print(f"  - Dimensi: {img_gray.shape}")  # (height, width) - tanpa channel
print(f"  - Tipe data: {img_gray.dtype}")

# ============================================================
# 3. Membaca gambar dengan alpha channel (transparansi)
# cv2.IMREAD_UNCHANGED (-1) = baca apa adanya termasuk alpha
# ============================================================

# Membaca foto berwarna (PNG output dari kamera biasanya tidak ada alpha channel)
img_unchanged = cv2.imread(os.path.join(IMAGE_DIR, "kota.jpg"), cv2.IMREAD_UNCHANGED)

# Menampilkan informasi gambar unchanged
print(f"\n[INFO] Gambar unchanged berhasil dimuat.")
print(f"  - Dimensi: {img_unchanged.shape}")
print(f"  - Tipe data: {img_unchanged.dtype}")

# ============================================================
# 4. Menampilkan gambar menggunakan matplotlib (lebih portable)
# Catatan: OpenCV menggunakan BGR, matplotlib menggunakan RGB
# ============================================================

# Membuat figure dengan 3 subplot untuk menampilkan 3 versi gambar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Gambar berwarna (konversi BGR -> RGB untuk matplotlib)
# ★ KODE INTI ★ — WAJIB konversi BGR→RGB sebelum plt.imshow()!
# OpenCV menyimpan warna dalam urutan Blue-Green-Red, matplotlib mengharapkan Red-Green-Blue
# Tanpa konversi ini, warna merah dan biru akan TERTUKAR di tampilan
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
axes[0].imshow(img_rgb)
axes[0].set_title("Gambar Berwarna (RGB)")
# Menghilangkan sumbu/axis agar tampilan lebih bersih
axes[0].axis("off")

# Subplot 2: Gambar grayscale (gunakan colormap 'gray')
axes[1].imshow(img_gray, cmap="gray")
axes[1].set_title("Gambar Grayscale")
axes[1].axis("off")

# Subplot 3: Gambar unchanged
img_unch_rgb = cv2.cvtColor(img_unchanged, cv2.COLOR_BGR2RGB)
axes[2].imshow(img_unch_rgb)
axes[2].set_title("Gambar Unchanged")
axes[2].axis("off")

# Mengatur layout agar tidak saling tumpang tindih
plt.suptitle("Percobaan 1: Tiga Mode Pembacaan Gambar", fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil visualisasi ke folder output
output_path = os.path.join(OUTPUT_DIR, "01_loading_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# Menampilkan gambar (opsional)
plt.show()

# ============================================================
# 5. Menampilkan menggunakan cv2.imshow() (GUI mode)
# ============================================================

# Menampilkan gambar di jendela OpenCV (hanya bekerja jika ada GUI)
try:
    # cv2.imshow(nama_jendela, gambar) - menampilkan gambar di jendela
#     cv2.imshow("Gambar Berwarna", img_color)  # (disabled for batch execution)
#     cv2.imshow("Gambar Grayscale", img_gray)  # (disabled for batch execution)

    # cv2.waitKey(1) - menunggu sampai user menekan tombol apapun
    # Parameter 0 = tunggu tanpa batas waktu
    # Parameter 1000 = tunggu 1000ms (1 detik)
    print("\n[INFO] Tekan tombol apapun pada jendela gambar untuk menutup...")
    cv2.waitKey(0)  # Tunggu sampai tombol ditekan

    # cv2.destroyAllWindows() - menutup semua jendela yang dibuat oleh OpenCV
    cv2.destroyAllWindows()
except Exception as e:
    print(f"[INFO] GUI tidak tersedia: {e}")
    print("[INFO] Hasil sudah disimpan ke folder output/")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 1")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.imread(path, flag)  → Membaca gambar dari file")
print("     - IMREAD_COLOR (1)      → Baca sebagai BGR 3 channel")
print("     - IMREAD_GRAYSCALE (0)  → Baca sebagai grayscale 1 channel")
print("     - IMREAD_UNCHANGED (-1) → Baca apa adanya (termasuk alpha)")
print("  2. cv2.imshow(nama, img)   → Tampilkan gambar di jendela GUI")
print("  3. cv2.waitKey(ms)         → Tunggu input keyboard")
print("  4. cv2.destroyAllWindows() → Tutup semua jendela GUI")
print("  5. cv2.cvtColor(img, code) → Konversi ruang warna (BGR↔RGB)")
print("=" * 60)
