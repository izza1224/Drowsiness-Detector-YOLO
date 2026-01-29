# 🚗 Drowsiness Detector using YOLO (Tugas Besar Visi Komputer)

Proyek ini adalah sistem deteksi kantuk secara real-time yang dirancang untuk membantu mengurangi risiko kecelakaan akibat pengemudi yang mengantuk. Sistem ini menggunakan algoritma **YOLO** (You Only Look Once) untuk mendeteksi mata terbuka dan tertutup melalui webcam.

## 📝 Deskripsi Proyek
Aplikasi ini dikembangkan sebagai bagian dari tugas mata kuliah Visi Komputer di Politeknik Negeri Semarang (Polines). Sistem akan memberikan peringatan berupa suara alarm jika mendeteksi mata tertutup (kantuk) dalam durasi tertentu.

## 🛠️ Fitur Utama
* **Real-time Detection**: Menggunakan model YOLOv8/v11 untuk deteksi instan.
* **Graphical User Interface (GUI)**: Antarmuka modern menggunakan `CustomTkinter`.
* **Audio Warning**: Alarm otomatis menggunakan `Pygame` saat kantuk terdeteksi.

## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat
Pastikan kamu sudah menginstal Python (disarankan v3.9+).

### 2. Instalasi Library
Buka terminal/command prompt dan jalankan perintah berikut untuk menginstal semua kebutuhan:
```bash
pip install ultralytics customtkinter pygame opencv-python pillow

## 3. Jalankan Program
lll:
```bash
python drowsiness_gui.py
