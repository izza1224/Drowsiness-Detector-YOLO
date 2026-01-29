# drowsiness_gui.py
# Deteksi Kantuk YOLOv8 dengan Antarmuka Grafis (CustomTkinter)
# Versi FINAL & OPTIMAL: Tuning Threshold, Smoothing 20 frame, Optimasi FPS, dan UI Responsif

from ultralytics import YOLO
import cv2
import threading
import time
import os
import pygame 
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

# --- 1. KONSTANTA DAN THRESHOLD ---
CLOSED_EYE_CLASS_ID = 0  
OPEN_MOUTH_CLASS_ID = 3  
OPEN_EYE_CLASS_ID = 2    
CLOSED_MOUTH_CLASS_ID = 1

# THRESHOLD TUNING (Disesuaikan untuk Akurasi Terbaik)
OPEN_MOUTH_CONFIDENCE_THRESHOLD = 0.80 
CLOSED_EYE_CONFIDENCE_THRESHOLD = 0.60 

# Logika Kuantitas 
FPS = 30
CLOSED_EYE_CONSEC_FRAMES = 3 * FPS  
YAWN_THRESH = 2                    

# Startup Grace Period (2 Detik)
GRACE_PERIOD_SECONDS = 2
GRACE_PERIOD_FRAMES = GRACE_PERIOD_SECONDS * FPS 

# --- KONSTANTA STABILITAS & OPTIMASI FPS ---
SMOOTHING_FRAMES = 20 # Smoothing Mulut/Mata (Sekitar 0.66 detik)
NMS_IOU_THRESHOLD = 0.6 
MOUTH_EYE_OVERLAP_THRESHOLD = 0.3 

# Optimasi FPS UI: Update UI setiap 33ms (~30 FPS)
UI_REFRESH_RATE_MS = 33 
# --- AKHIR KONSTANTA STABILITAS & OPTIMASI FPS ---

ALARM_FILE_PATH = "alarm.wav"           
VOICE_ALERT_PATH = "voice_alert.wav"    
VOICE_LOOP_COUNT = 3
# --- AKHIR KONSTANTA ---


# --- FUNGSI UTILITY (IoU & NMS) ---

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def apply_nms(boxes_data, iou_threshold):
    if not boxes_data: return []
    boxes_data.sort(key=lambda x: x['confidence'], reverse=True)
    kept_boxes = []
    kept_coords = [] 
    for current_box in boxes_data:
        current_coords = current_box['box']
        should_keep = True
        for kept_coord in kept_coords:
            if calculate_iou(current_coords, kept_coord) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept_boxes.append(current_box)
            kept_coords.append(current_coords)
    return kept_boxes
# --- AKHIR FUNGSI UTILITY ---


class DrowsinessDetector:
    def __init__(self, detector_app):
        self.app = detector_app
        self.running = False
        self.frame = None
        self.status = "IDLE"

        # Inisialisasi Counter
        self.EYE_COUNTER = 0
        self.YAWN_COUNTER = 0
        self.YAWN_STATE = 0
        self.FRAME_COUNT = 0
        self.ALARM_ON = False
        self.last_stable_mouth_box = None
        self.eye_closed_tracker = 0
        self.mouth_open_tracker = 0
        self.cap = None

        pygame.mixer.init() 
        try:
            self.model = YOLO("best.pt")
        except Exception as e:
            print(f"ERROR: Gagal memuat model: {e}")
            self.model = None

    def start_detection(self):
        if self.running or not self.model: return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status = "CAMERA ERROR"
            return
        
        # Optimasi: Set resolusi capture lebih rendah untuk mengurangi beban komputasi
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        # Reset counter saat mulai
        self.EYE_COUNTER = 0
        self.YAWN_COUNTER = 0
        self.YAWN_STATE = 0
        self.FRAME_COUNT = 0
        self.ALARM_ON = False
        self.app.update_counters(0, 0)
        
        threading.Thread(target=self._detection_loop, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.status = "STOPPED"
        if self.cap:
            self.cap.release()
        self._reset_alarm_and_counters(reset_all=True) # Reset total saat stop
        
    def reset_alarm(self):
        # Dipanggil oleh tombol Reset Alarm
        self._reset_alarm_and_counters(reset_all=False) # Reset hanya alarm dan counter
        
    def _reset_alarm_and_counters(self, reset_all=True):
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        
        self.ALARM_ON = False
        self.EYE_COUNTER = 0
        self.YAWN_COUNTER = 0
        self.YAWN_STATE = 0
        self.eye_closed_tracker = 0
        self.mouth_open_tracker = 0
        
        if reset_all:
             self.status = "READY"
             self.app.update_alarm_status("AMAN", "green")
        else:
             self.app.update_alarm_status("RESET", "yellow")
             # Kembalikan status alarm ke AMAN setelah reset (0.5 detik)
             self.app.after(500, lambda: self.app.update_alarm_status("AMAN", "green"))


        print("[INFO] Alarm dihentikan dan counter direset.")

    def _sound_alarm(self, path, loop_count):
        if not os.path.exists(path): 
            print(f"ERROR: File suara tidak ditemukan di: {path}")
            return
        try:
            if not pygame.mixer.get_init(): pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame_loop = -1 if loop_count == -1 else (loop_count - 1)
            pygame.mixer.music.play(pygame_loop)
        except Exception as e:
            print(f"ERROR PYGAME PLAYSOUND: {e}")
            
    def _detection_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            # Optimasi: Resize frame ke resolusi 640x480 (atau yang di set di cap)
            # frame = cv2.resize(frame, (640, 480)) # Jika tidak di set di cap
            
            self.FRAME_COUNT += 1

            results = self.model(frame, verbose=False) 
            
            WARNING_TEXT = ""
            TRIGGER_EYE_ALARM = False
            TRIGGER_YAWN_ALERT = False
            
            found_closed_eye = False
            found_open_eye = False
            found_open_mouth = False
            all_mouth_detections = []
            eye_detections = []

            # Proses deteksi
            for result in results:
                for box_data in result.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, confidence, class_id = box_data[:6]
                    class_id = int(class_id)
                    box = (int(x1), int(y1), int(x2), int(y2))
                    det = {'class_id': class_id, 'confidence': confidence, 'box': box}

                    if class_id == OPEN_MOUTH_CLASS_ID or class_id == CLOSED_MOUTH_CLASS_ID:
                        all_mouth_detections.append(det)
                    elif class_id == CLOSED_EYE_CLASS_ID or class_id == OPEN_EYE_CLASS_ID:
                        eye_detections.append(det)

            # 1. NMS MANUAL UNTUK MULUT (Stabilitas Kotak)
            stable_mouth_detections = apply_nms(all_mouth_detections, NMS_IOU_THRESHOLD)
            best_mouth_box = stable_mouth_detections[0] if stable_mouth_detections else None
            
            # 2. PEMBEBASAN DETEKSI MATA (Hapus False Positive)
            final_eye_detections = []
            for eye_det in eye_detections:
                conf = eye_det['confidence']
                if eye_det['class_id'] == CLOSED_EYE_CLASS_ID and best_mouth_box:
                    iou = calculate_iou(eye_det['box'], best_mouth_box['box'])
                    if iou > MOUTH_EYE_OVERLAP_THRESHOLD: continue # Abaikan
                
                # Filter mata untuk alarm dan smoothing
                if (eye_det['class_id'] == CLOSED_EYE_CLASS_ID and conf >= CLOSED_EYE_CONFIDENCE_THRESHOLD) or \
                   (eye_det['class_id'] == OPEN_EYE_CLASS_ID):
                    final_eye_detections.append(eye_det)
                    
            # Set status found_X_X untuk smoothing
            for det in final_eye_detections:
                if det['class_id'] == CLOSED_EYE_CLASS_ID: found_closed_eye = True
                elif det['class_id'] == OPEN_EYE_CLASS_ID: found_open_eye = True

            # Set status open mouth
            if best_mouth_box:
                if best_mouth_box['class_id'] == OPEN_MOUTH_CLASS_ID and best_mouth_box['confidence'] >= OPEN_MOUTH_CONFIDENCE_THRESHOLD:
                    found_open_mouth = True
                self.last_stable_mouth_box = best_mouth_box

            # 3. Update Smoothing Tracker
            self.eye_closed_tracker = SMOOTHING_FRAMES if found_closed_eye else max(0, self.eye_closed_tracker - 1)
            self.mouth_open_tracker = SMOOTHING_FRAMES if found_open_mouth else max(0, self.mouth_open_tracker - 1)

            is_eye_closed_smoothed = self.eye_closed_tracker > 0
            is_mouth_open_smoothed = self.mouth_open_tracker > 0
            
            # --- LOGIKA KANTUK DAN ALARM ---
            
            if self.FRAME_COUNT > GRACE_PERIOD_FRAMES: # Setelah Grace Period
                
                # a) Logika Mata
                if is_eye_closed_smoothed:
                    self.EYE_COUNTER += 1
                    if self.EYE_COUNTER >= CLOSED_EYE_CONSEC_FRAMES:
                        TRIGGER_EYE_ALARM = True
                        WARNING_TEXT = "!!! KANTUK BERAT (MATA TERTUTUP >3s) !!!"
                else:
                    self.EYE_COUNTER = 0 
                    
                # b) Logika Menguap
                if self.YAWN_STATE == 0 and is_mouth_open_smoothed: self.YAWN_STATE = 1 
                elif self.YAWN_STATE == 1 and not is_mouth_open_smoothed:
                    # HANYA hitung menguap saat transisi OPEN -> CLOSE selesai
                    self.YAWN_COUNTER += 1
                    self.YAWN_STATE = 0 
                
                if self.YAWN_COUNTER >= YAWN_THRESH:
                    TRIGGER_YAWN_ALERT = True
                    if not WARNING_TEXT: WARNING_TEXT = "!!! MENGUAP (SEGERA ISTIRAHAT) !!!"
                    
                # c) Pemicu Alarm
                if not self.ALARM_ON:
                    if TRIGGER_EYE_ALARM:
                        self.ALARM_ON = True
                        self.app.update_alarm_status("ALARM BERBUNYI!", "red")
                        threading.Thread(target=self._sound_alarm, args=(ALARM_FILE_PATH, -1)).start()
                    elif TRIGGER_YAWN_ALERT:
                        # Peringatan Yawn: Reset alarm internal detektor setelah dipicu
                        self.ALARM_ON = True
                        self.app.update_alarm_status("PERINGATAN SUARA", "orange")
                        threading.Thread(target=self._sound_alarm, args=(VOICE_ALERT_PATH, VOICE_LOOP_COUNT)).start()
                        self.YAWN_COUNTER = 0 # Reset counter agar tidak trigger lagi
                        self.YAWN_STATE = 0 
                
                # d) Penanganan Alarm Aktif
                elif self.ALARM_ON and not pygame.mixer.music.get_busy() and not TRIGGER_EYE_ALARM:
                    self.ALARM_ON = False
                    self.app.update_alarm_status("AMAN", "green")
                
                # e) Update Status Label UI
                if not self.ALARM_ON:
                    self.status = "SIAP"
                
                self.app.update_status_labels(is_eye_closed_smoothed, is_mouth_open_smoothed)
                self.app.update_counters(self.EYE_COUNTER, self.YAWN_COUNTER)
            
            else:
                s_left = int(GRACE_PERIOD_SECONDS - self.FRAME_COUNT/FPS)
                self.status = f"STARTING UP... ({s_left}s left)"
                self.app.update_status_labels(False, False, startup_status=self.status)

            # 4. Gambar Bounding Box di Frame
            for item in final_eye_detections:
                if (item['class_id'] == CLOSED_EYE_CLASS_ID and item['confidence'] >= CLOSED_EYE_CONFIDENCE_THRESHOLD) or \
                   (item['class_id'] == OPEN_EYE_CLASS_ID):
                    self._draw_box(frame, item)

            if self.last_stable_mouth_box:
                if is_mouth_open_smoothed:
                    self._draw_box(frame, self.last_stable_mouth_box, forced_label=OPEN_MOUTH_CLASS_ID)
                else:
                    self._draw_box(frame, self.last_stable_mouth_box, forced_label=CLOSED_MOUTH_CLASS_ID)
            
            self.frame = frame

        self.stop_detection()
        
    def _draw_box(self, frame, det, forced_label=None):
        c_id = forced_label if forced_label is not None else det['class_id']
        conf = det['confidence']
        box = det['box']
        
        color = (0, 0, 0)
        # Warna Kotak
        if c_id == CLOSED_EYE_CLASS_ID: color = (0, 0, 255) 
        elif c_id == OPEN_MOUTH_CLASS_ID: color = (255, 0, 0) 
        elif c_id == CLOSED_MOUTH_CLASS_ID: color = (255, 255, 0) 
        elif c_id == OPEN_EYE_CLASS_ID: color = (0, 255, 0) 

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = self.model.names.get(c_id, 'UNKNOWN')
        cv2.putText(frame, f'{label} {conf:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Aplikasi Deteksi Kantuk YOLOv8")
        # Mengatur agar jendela dapat diubah ukurannya dan mengisi layar
        self.geometry("1000x600")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1) # Membuat baris utama responsif
        selfont = ctk.CTkFont(family="Arial", size=14)

        self.detector = DrowsinessDetector(self)
        self.current_frame_ref = None 

        # --- A. UI: Bingkai Video (Mengisi Kolom Kiri) ---
        self.video_frame = ctk.CTkFrame(self, width=680, height=580)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_frame.grid_propagate(False)
        
        # Label Video dibuat responsif di dalam frame
        self.video_label = ctk.CTkLabel(self.video_frame, text="Tekan 'Mulai Kamera'...", font=ctk.CTkFont(size=18))
        self.video_label.place(relx=0.5, rely=0.5, anchor="center") # Posisikan di tengah frame

        # --- B. UI: Bingkai Kontrol & Status (Mengisi Kolom Kanan) ---
        self.control_frame = ctk.CTkFrame(self, width=300, height=580)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.control_frame.grid_columnconfigure(0, weight=1)
        # Membuat ruang di bawah detail counter agar tombol menempel di bawah
        self.control_frame.grid_rowconfigure(10, weight=1) 

        # 1. Judul
        ctk.CTkLabel(self.control_frame, text="STATUS SISTEM", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="n")
        
        # 2. Status Kamera
        ctk.CTkLabel(self.control_frame, text="Status Kamera:", font=selfont).grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.status_kamera_label = ctk.CTkLabel(self.control_frame, text="█ TIDAK AKTIF", fg_color="red", corner_radius=6, font=selfont, width=200)
        self.status_kamera_label.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="w")
        
        # 3. Status Deteksi (Mata & Mulut)
        ctk.CTkLabel(self.control_frame, text="Status Deteksi:", font=selfont).grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.status_deteksi_label = ctk.CTkLabel(self.control_frame, text="IDLE", fg_color="gray", corner_radius=6, font=selfont, width=200)
        self.status_deteksi_label.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="w")

        # 4. Status Alarm
        ctk.CTkLabel(self.control_frame, text="Status Alarm:", font=selfont).grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.status_alarm_label = ctk.CTkLabel(self.control_frame, text="AMAN", fg_color="green", corner_radius=6, font=selfont, width=200)
        self.status_alarm_label.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="w")

        # 5. Detail Counter
        ctk.CTkLabel(self.control_frame, text="Detail Counter:", font=selfont).grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.eye_counter_label = ctk.CTkLabel(self.control_frame, text=f"Mata Tertutup: 0/{CLOSED_EYE_CONSEC_FRAMES} frame", font=selfont)
        self.eye_counter_label.grid(row=8, column=0, padx=20, sticky="w")
        self.yawn_counter_label = ctk.CTkLabel(self.control_frame, text=f"Menguap: 0/{YAWN_THRESH} kali", font=selfont)
        self.yawn_counter_label.grid(row=9, column=0, padx=20, pady=(0, 10), sticky="w")
        
        # 6. Tombol Kontrol (Diletakkan di bagian bawah karena ada weight=1 di row 10)
        self.start_button = ctk.CTkButton(self.control_frame, text="Mulai Kamera", command=self.start_camera)
        self.start_button.grid(row=11, column=0, padx=20, pady=(20, 5), sticky="ew") # Row 11

        self.reset_alarm_button = ctk.CTkButton(self.control_frame, text="Reset Alarm", command=self.detector.reset_alarm, state="disabled", fg_color="orange")
        self.reset_alarm_button.grid(row=12, column=0, padx=20, pady=(0, 5), sticky="ew") # Row 12

        self.stop_button = ctk.CTkButton(self.control_frame, text="Hentikan Kamera", command=self.stop_camera, state="disabled", fg_color="red")
        self.stop_button.grid(row=13, column=0, padx=20, pady=(0, 20), sticky="ew") # Row 13

        # 7. Update Loop
        self.update_video_frame()
    
    def start_camera(self):
        self.detector.start_detection()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.reset_alarm_button.configure(state="normal")
        self.status_kamera_label.configure(text="█ AKTIF", fg_color="green")
        self.status_deteksi_label.configure(text="STARTING UP...", fg_color="gray")
        # PERBAIKAN UI: Hapus teks segera setelah tombol ditekan
        self.video_label.configure(image=None, text="") 

    def stop_camera(self):
        self.detector.stop_detection()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.reset_alarm_button.configure(state="disabled")
        self.status_kamera_label.configure(text="█ TIDAK AKTIF", fg_color="red")
        self.status_deteksi_label.configure(text="IDLE", fg_color="gray")
        self.current_frame_ref = None 
        # PERBAIKAN UI: Tambahkan teks "Tekan Mulai Kamera" saat berhenti
        self.video_label.configure(image=None, text="Tekan 'Mulai Kamera'") 
        self.update_counters(0, 0)
        
    def update_alarm_status(self, text, color):
        self.status_alarm_label.configure(text=text, fg_color=color)

    def update_status_labels(self, is_eye_closed, is_mouth_open, startup_status=None):
        if startup_status:
             self.status_deteksi_label.configure(text=startup_status, fg_color="orange")
             return
             
        if is_eye_closed:
            self.status_deteksi_label.configure(text="MENGANTUK!", fg_color="red")
        elif is_mouth_open:
            self.status_deteksi_label.configure(text="WASPADA (Menguap)", fg_color="orange")
        else:
            self.status_deteksi_label.configure(text="FOKUS", fg_color="green")

    def update_counters(self, eye_count, yawn_count):
        self.eye_counter_label.configure(text=f"Mata Tertutup: {eye_count}/{CLOSED_EYE_CONSEC_FRAMES} frame")
        self.yawn_counter_label.configure(text=f"Menguap: {yawn_count}/{YAWN_THRESH} kali")
        
    def update_video_frame(self):
        # Ambil lebar dan tinggi frame video secara dinamis
        video_frame_width = self.video_frame.winfo_width()
        video_frame_height = self.video_frame.winfo_height()

        if self.detector.running and self.detector.frame is not None:
            # Tetapkan ukuran display agar sesuai dengan frame yang tersedia
            display_width = video_frame_width - 20 # Kurangi padding
            display_height = video_frame_height - 20 # Kurangi padding

            # Pastikan ukuran positif sebelum resize
            if display_width > 0 and display_height > 0:
                frame_resized = cv2.resize(self.detector.frame, (display_width, display_height))
                
                cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.current_frame_ref = imgtk
                
                # Posisikan label video di tengah frame yang responsif
                self.video_label.configure(image=imgtk, width=display_width, height=display_height)
                self.video_label.place(relx=0.5, rely=0.5, anchor="center") 
            
        elif not self.detector.running and self.current_frame_ref is not None:
            self.current_frame_ref = None
            self.video_label.configure(image=None, text="Tekan 'Mulai Kamera'")
            self.video_label.place(relx=0.5, rely=0.5, anchor="center") # Pastikan teks tetap di tengah
        
        # Optimasi: Refresh rate yang lebih rendah untuk mengurangi lag
        self.after(UI_REFRESH_RATE_MS, self.update_video_frame)


if __name__ == "__main__":
    if not os.path.exists("best.pt"):
        print("\nFATAL ERROR: File 'best.pt' tidak ditemukan di direktori ini.")
    else:
        if not pygame.mixer.get_init():
             pygame.mixer.init() 
        
        ctk.set_appearance_mode("Dark") # Tampilan Dark Mode
        app = App()
        app.mainloop()