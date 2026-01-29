# train_yolo.py
# Skrip untuk Melatih Model Object Detection YOLOv8 untuk Proyek Deteksi Kantuk

from ultralytics import YOLO
import os
import torch
import warnings

# Mengabaikan warnings untuk tampilan output yang lebih bersih saat training
warnings.filterwarnings("ignore", category=UserWarning)

# --- FUNGSI UTAMA UNTUK PELATIHAN ---
def train_model():
    # --- 1. Konfigurasi Path ---
    DATA_CONFIG_PATH = os.path.join('YOLO_Dataset', 'data.yaml') 

    # --- 2. Cek Status Hardware ---
    print("\n--- Status Hardware ---")
    if torch.cuda.is_available():
        print("Menggunakan GPU: YA")
        device = 0 # Menggunakan GPU pertama (RTX Anda)
    else:
        print("Menggunakan GPU: TIDAK (Menggunakan CPU)")
        device = 'cpu'
        
    # --- 3. Inisialisasi Model ---
    try:
        model = YOLO('yolov8n.pt') 
    except Exception as e:
        print(f"ERROR: Gagal memuat model YOLO. Pastikan Anda sudah menginstal 'ultralytics'. Error: {e}")
        return

    # --- 4. Mulai Pelatihan ---
    print("\n--- Memulai Pelatihan Model YOLOv8 ---")
    print(f"Path data.yaml: {os.path.abspath(DATA_CONFIG_PATH)}")

    try:
        # Latih model
        results = model.train(data=DATA_CONFIG_PATH, 
                              epochs=100, 
                              imgsz=640, 
                              name='drowsiness_final_run', # Ganti nama jika ingin run baru
                              batch=8,
                              device=device,
                              workers=4) # Mengurangi workers (default 8) sering membantu di Windows
        
        print("\n--- Pelatihan Selesai! ---")
        output_path = results.save_dir
        weights_path = os.path.join(output_path, 'weights', 'best.pt')
        
        print(f"MODEL BERHASIL DILATIH. Weights tersimpan di: {weights_path}")
        print("\nLANGKAH SELANJUTNYA: Pindahkan file 'best.pt' ini ke folder utama proyek Anda untuk menjalankan 'yolo_app.py'.")

    except Exception as e:
        print(f"Terjadi error saat training: {e}")
        
# --- STRUKTUR WAJIB WINDOWS (PENTING!) ---
if __name__ == '__main__':
    train_model()

# Jika error Runtime Error: An attempt has been made to start a new process masih muncul:
# 1. Coba kurangi atau set 'workers=0' di fungsi model.train()
# 2. Pastikan file 'train_yolo.py' tidak memiliki kode di luar blok if __name__ == '__main__':