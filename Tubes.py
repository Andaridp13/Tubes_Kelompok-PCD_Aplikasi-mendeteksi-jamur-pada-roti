import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QFrame,
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QMessageBox, QStatusBar, QMenuBar,
    QMenu
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize
import os
import time

# Import YOLO model dari Ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: Library 'ultralytics' tidak ditemukan. Harap instal dengan: pip install ultralytics")
    print("Fitur deteksi objek akan dinonaktifkan.")

# === FUNGSI PEMROSESAN CITRA ===

def enhance_contrast_stretching(img):
    """
    Meningkatkan kontras gambar menggunakan metode Contrast Stretching tradisional
    pada kanal Y (luminance) dari ruang warna YCrCb.
    Mengembalikan gambar dalam format BGR yang kontrasnya telah ditingkatkan.
    """
    if img is None:
        return None

    # Memastikan gambar dalam format BGR. Jika grayscale, ubah menjadi BGR.
    if len(img.shape) < 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy() # Hindari modifikasi inplace

    # Konversi ke ruang warna YCrCb
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0] # Ambil kanal Y (luminance)

    # Hitung nilai min dan max dari kanal Y
    min_y = np.min(y)
    max_y = np.max(y)

    # Lakukan Contrast Stretching
    if max_y - min_y == 0:
        y_stretched = y # Tidak ada stretching jika min=max (gambar uniform)
    else:
        # Rumus contrast stretching: (piksel - min) * (255 / (max - min))
        y_stretched = ((y - min_y) / (max_y - min_y) * 255).astype(np.uint8)

    # Ganti kanal Y asli dengan yang telah ditingkatkan kontrasnya
    ycrcb[:, :, 0] = y_stretched
    # Konversi kembali ke BGR
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return enhanced

def apply_hsv_threshold(img, lower_hsv, upper_hsv):
    """
    Menerapkan thresholding berbasis HSV pada gambar untuk membuat mask biner.
    Mengembalikan mask biner.
    """
    if img is None:
        print("apply_hsv_threshold: Input gambar kosong.")
        return None

    # Memastikan gambar dalam format BGR sebelum konversi ke HSV
    if len(img.shape) < 3:
        print("apply_hsv_threshold: Mengubah gambar grayscale menjadi BGR.")
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy()

    # Konversi gambar ke ruang warna HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Buat mask menggunakan rentang HSV yang ditentukan
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask

# --- Fungsi Morfologi Baru ---
DEFAULT_KERNEL_SIZE = (5, 5) # Ukuran kernel default untuk operasi morfologi

def apply_morphology_erode(mask, kernel_size=DEFAULT_KERNEL_SIZE):
    """Menerapkan operasi erosi pada mask biner."""
    if mask is None: return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.erode(mask, kernel)

def apply_morphology_dilate(mask, kernel_size=DEFAULT_KERNEL_SIZE):
    """Menerapkan operasi dilasi pada mask biner."""
    if mask is None: return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(mask, kernel)

def apply_morphology_open(mask, kernel_size=DEFAULT_KERNEL_SIZE):
    """Menerapkan operasi opening (erosi lalu dilasi) pada mask biner."""
    if mask is None: return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def apply_morphology_close(mask, kernel_size=DEFAULT_KERNEL_SIZE):
    """Menerapkan operasi closing (dilasi lalu erosi) pada mask biner."""
    if mask is None: return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# --- Akhir Fungsi Morfologi Baru ---

def draw_detections_and_classify(img_original, detection_results, confidence_threshold=0.5):
    
    img_display = img_original.copy()
    kualitas = "SEGAR" # Default kualitas

    mold_detected = False

    # Jika tidak ada hasil deteksi, kembalikan gambar asli dan kualitas default
    if not detection_results:
        return img_display, kualitas

    # Iterasi melalui setiap objek yang terdeteksi
    for obj in detection_results:
        obj_name = obj['name']
        xmin, ymin, xmax, ymax = obj['bbox']
        score = obj.get('score', 1.0) # Ambil skor, default 1.0 jika tidak ada

        # Lewati deteksi yang di bawah ambang kepercayaan (confidence)
        if score < confidence_threshold:
            continue

        box_color = (0, 255, 0) # Warna hijau untuk objek non-jamur
        text_color = (0, 255, 0) # Warna teks hijau

        # Cek apakah objek yang terdeteksi adalah jamur
        if obj_name.lower() in ['mold', 'mould', 'bread_mold', 'jamur']:
            mold_detected = True
            box_color = (0, 0, 255) # Warna merah untuk jamur
            text_color = (0, 0, 255) # Warna teks merah

        # Pastikan koordinat berada dalam batas gambar
        img_h, img_w = img_display.shape[:2]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_w - 1, xmax)
        ymax = min(img_h - 1, ymax)

        # Gambar kotak pembatas jika koordinat valid
        if xmin < xmax and ymin < ymax:
            cv2.rectangle(img_display, (xmin, ymin), (xmax, ymax), box_color, 2)
            display_text = f"{obj_name} ({score:.2f})"
            # Atur posisi teks agar tidak keluar batas atas gambar
            text_x = xmin
            text_y = ymin - 5 if ymin - 5 > 10 else ymin + 15
            cv2.putText(img_display, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    # Ubah kualitas menjadi "BERJAMUR" jika ada jamur terdeteksi
    if mold_detected:
        kualitas = "BERJAMUR"

    return img_display, kualitas

# === KELAS UTAMA APLIKASI GUI ===
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Aktifkan Qt Fusion Style untuk tampilan modern
        QApplication.setStyle("Fusion")

        # Tambahkan palet warna gelap custom untuk tampilan modern
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(40, 44, 52))
        dark_palette.setColor(QtGui.QPalette.WindowText, Qt.white)
        dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 34, 40))
        dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(44, 49, 60))
        dark_palette.setColor(QtGui.QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QtGui.QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QtGui.QPalette.Text, Qt.white)
        dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 132, 228))
        dark_palette.setColor(QtGui.QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QtGui.QPalette.BrightText, Qt.red)
        dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(53, 132, 228))
        dark_palette.setColor(QtGui.QPalette.HighlightedText, Qt.white)
        dark_palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(127, 127, 127))
        dark_palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(127, 127, 127))
        QApplication.setPalette(dark_palette)

        # Gunakan font modern
        font = QtGui.QFont("Segoe UI", 11)
        QApplication.setFont(font)
        # Terapkan tema gelap modern untuk aplikasi
        self.setStyleSheet("""
            QMainWindow {
            background-color: #23272e;
            }
            QWidget {
            background-color: #23272e;
            color: #e0e0e0;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 13px;
            }
            QFrame {
            background-color: #23272e;
            border: 1px solid #353b48;
            border-radius: 10px;
            }
            QLabel {
            color: #e0e0e0;
            }
            QPushButton {
            background-color: #3b82f6;
            color: #fff;
            border: none;
            padding: 12px;
            font-size: 13px;
            font-weight: bold;
            border-radius: 8px;
            margin-bottom: 5px;
            }
            QPushButton:hover {
            background-color: #2563eb;
            }
            QPushButton:pressed {
            background-color: #1d4ed8;
            }
            QPushButton:disabled {
            background-color: #374151;
            color: #9ca3af;
            }
            QMenuBar {
            background-color: #23272e;
            color: #e0e0e0;
            }
            QMenuBar::item:selected {
            background: #3b82f6;
            }
            QMenu {
            background-color: #23272e;
            color: #e0e0e0;
            border: 1px solid #353b48;
            }
            QMenu::item:selected {
            background-color: #3b82f6;
            color: #fff;
            }
            QStatusBar {
            background: #181a20;
            color: #e0e0e0;
            border-top: 1px solid #353b48;
            }
        """)
        
        # --- PERBAIKAN: Inisialisasi SEMUA variabel instance di awal konstruktor ---
        self.current_step = 0 # Melacak langkah aktif untuk mengontrol tombol

        self.current_image_original = None
        self.current_image_processed = None 
        self.current_image_hsv_mask = None  
        self.current_image_morphed_mask = None # Akan menyimpan hasil morfologi terakhir
        self.current_image_detection = None 
        self.current_image_path = None
        self.detection_model = None # Inisialisasi di sini juga
        # --- END PERBAIKAN ---

        self.initUI() # initUI() akan memanggil set_button_states(self.current_step)

        # Pastikan model deteksi YOLO dimuat saat aplikasi dijalankan
        self.load_detection_model()
        
        self.clear_image_labels()
        self.label_hasil_klasifikasi.setText("Kualitas: Siap Dimulai")
        self.label_hasil_klasifikasi.setAlignment(Qt.AlignCenter)
        self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: black;")

        # Memperluas jendela aplikasi saat pertama kali ditampilkan
        self.showMaximized()


    def initUI(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle("Aplikasi Deteksi Kualitas Roti Tawar")
        self.setWindowIcon(QtGui.QIcon())

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # HEADER SECTION
        header_frame = QFrame()
        # header_frame.setFrameRule(QFrame.Box) # Dihapus: Ini menyebabkan AttributeError
        # header_frame.setLineWidth(2)         # Dihapus: Ditangani oleh stylesheet
        header_frame.setFrameStyle(QFrame.Box | QFrame.Raised) # Menggunakan setFrameStyle yang benar
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)

        self.label_title = QLabel("SISTEM DETEKSI KUALITAS ROTI TAWAR")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                background: transparent;
                border: none;
            }
        """)
        header_layout.addWidget(self.label_title)

        self.label_subtitle = QLabel("Aplikasi Mendeteksi Kualitas Roti Tawar Berdasarkan Citra")
        self.label_subtitle.setAlignment(Qt.AlignCenter)
        self.label_subtitle.setStyleSheet("""
            QLabel {
                color: #bdc3c7;
                font-size: 14px;
                font-style: italic;
                padding: 5px;
                background: transparent;
                border: none;
            }
        """)
        header_layout.addWidget(self.label_subtitle)
        main_layout.addWidget(header_frame)

        # CONTENT SECTION
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # LEFT PANEL - Controls
        left_panel = QFrame()
        # left_panel.setFrameRule(QFrame.Box) # Dihapus
        # left_panel.setLineWidth(2)          # Dihapus
        left_panel.setFrameStyle(QFrame.Box | QFrame.Raised) # Menggunakan setFrameStyle yang benar
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        control_title = QLabel("PANEL KONTROL")
        control_title.setAlignment(Qt.AlignCenter)
        control_title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #3498db;
                color: white;
                border-radius: 5px;
            }
        """)
        left_layout.addWidget(control_title)

        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 15px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 8px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #7f8c8d;
            }
        """
        
        self.btn_pilih_gambar = QPushButton("1. Pilih Gambar Roti")
        self.btn_pilih_gambar.setMinimumHeight(50)
        self.btn_pilih_gambar.setStyleSheet(button_style)
        left_layout.addWidget(self.btn_pilih_gambar)

        self.btn_proses_contrast = QPushButton("2. Perbaiki Kualitas Citra")
        self.btn_proses_contrast.setMinimumHeight(50)
        self.btn_proses_contrast.setStyleSheet(button_style)
        left_layout.addWidget(self.btn_proses_contrast)

        self.btn_proses_hsv = QPushButton("3. Terapkan HSV Threshold")
        self.btn_proses_hsv.setMinimumHeight(50)
        self.btn_proses_hsv.setStyleSheet(button_style)
        left_layout.addWidget(self.btn_proses_hsv)
        
        # --- Tombol Morfologi dengan QMenu ---
        self.btn_proses_morphology = QPushButton("4. Terapkan Morfologi")
        self.btn_proses_morphology.setMinimumHeight(50)
        self.btn_proses_morphology.setStyleSheet(button_style)
        left_layout.addWidget(self.btn_proses_morphology)

        # Buat QMenu untuk operasi morfologi
        self.morphology_menu = QMenu(self)
        self.morphology_menu.setStyleSheet("""
            QMenu {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
            }
            QMenu::item {
                padding: 10px 20px;
                background-color: transparent;
                color: #2c3e50;
            }
            QMenu::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)

        # Tambahkan aksi ke menu
        self.action_erode = self.morphology_menu.addAction("Erosi")
        self.action_dilate = self.morphology_menu.addAction("Dilasi")
        self.action_open = self.morphology_menu.addAction("Opening")
        self.action_close = self.morphology_menu.addAction("Closing")
        
        # Hubungkan aksi dengan fungsi pemrosesan
        self.action_erode.triggered.connect(self.action_morphology_erode)
        self.action_dilate.triggered.connect(self.action_morphology_dilate)
        self.action_open.triggered.connect(self.action_morphology_opening)
        self.action_close.triggered.connect(self.action_morphology_closing)

        # Atur tombol untuk menampilkan menu saat diklik
        self.btn_proses_morphology.setMenu(self.morphology_menu)
        # --- Akhir Tombol Morfologi dengan QMenu ---


        self.btn_deteksi_jamur = QPushButton("5. Deteksi & Klasifikasi") 
        self.btn_deteksi_jamur.setMinimumHeight(50)
        self.btn_deteksi_jamur.setStyleSheet(button_style)
        left_layout.addWidget(self.btn_deteksi_jamur)

        self.btn_reset_aplikasi = QPushButton("6. Reset Aplikasi") 
        self.btn_reset_aplikasi.setMinimumHeight(50)
        self.btn_reset_aplikasi.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 15px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 8px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #7f8c8d;
            }
        """)
        left_layout.addWidget(self.btn_reset_aplikasi)

        # Status Section
        status_frame = QFrame()
        # status_frame.setFrameRule(QFrame.Box) # Dihapus
        # status_frame.setLineWidth(2)          # Dihapus
        status_frame.setFrameStyle(QFrame.Box | QFrame.Raised) # Menggunakan setFrameStyle yang benar
        status_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        status_layout = QVBoxLayout(status_frame)

        status_label = QLabel("STATUS SISTEM")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        status_layout.addWidget(status_label)

        self.label_hasil_klasifikasi = QLabel("Kualitas: Siap Dimulai")
        self.label_hasil_klasifikasi.setAlignment(Qt.AlignCenter)
        self.label_hasil_klasifikasi.setWordWrap(True)
        self.label_hasil_klasifikasi.setMinimumHeight(80)
        self.label_hasil_klasifikasi.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
            }
        """)
        status_layout.addWidget(self.label_hasil_klasifikasi)

        left_layout.addWidget(status_frame)
        left_layout.addStretch()

        # RIGHT PANEL - Display Images
        right_panel = QFrame()
        # right_panel.setFrameRule(QFrame.Box) # Dihapus
        # right_panel.setLineWidth(2)          # Dihapus
        right_panel.setFrameStyle(QFrame.Box | QFrame.Raised) # Menggunakan setFrameStyle yang benar
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)

        # Menggunakan QHBoxLayout untuk menampilkan gambar berdampingan
        image_display_layout = QHBoxLayout()
        image_display_layout.setSpacing(15) # Spasi antara dua label gambar

        self.label_gambar_asli = QLabel("Gambar belum dipilih\nKlik 'Pilih Gambar Roti'")
        self.label_gambar_asli.setAlignment(Qt.AlignCenter)
        # Menentukan ukuran minimum yang lebih fleksibel dan kebijakan ukuran untuk ekspansi
        # Minimum size diatur cukup rendah agar bisa fleksibel, expanding policy yang akan membuat besar
        self.label_gambar_asli.setMinimumSize(QSize(200, 150))
        self.label_gambar_asli.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.label_gambar_asli.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                font-size: 14px;
                color: #7f8c8d;
                padding: 10px;
            }
        """)
        image_display_layout.addWidget(self.label_gambar_asli, 1) # Tambahkan stretch factor

        self.label_gambar_proses = QLabel("Hasil pemrosesan akan\nditampilkan di sini")
        self.label_gambar_proses.setAlignment(Qt.AlignCenter)
        # Menentukan ukuran minimum yang lebih fleksibel dan kebijakan ukuran untuk ekspansi
        self.label_gambar_proses.setMinimumSize(QSize(200, 150))
        self.label_gambar_proses.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.label_gambar_proses.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                font-size: 14px;
                color: #7f8c8d;
                padding: 10px;
            }
        """)
        image_display_layout.addWidget(self.label_gambar_proses, 1) # Tambahkan stretch factor

        # Tambahkan layout horizontal gambar ke layout vertikal panel kanan
        right_layout.addLayout(image_display_layout)
        right_layout.addStretch() # Pastikan stretch tetap ada di bagian bawah

        content_layout.addWidget(left_panel)
        content_layout.addWidget(right_panel, 1) # Tambahkan stretch factor untuk panel kanan
        main_layout.addLayout(content_layout)

        # Setup Menu Bar and Status Bar
        self.menubar = QMenuBar(self)
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Siap untuk memulai deteksi kualitas roti tawar")

        # Connect buttons

        # Step 1: Pilih gambar
        self.btn_pilih_gambar.clicked.connect(self.action_step1_pilih_gambar)
        # Step 2: Perbaiki kualitas citra (Contrast Stretching)
        self.btn_proses_contrast.clicked.connect(self.action_step2_perbaiki_kualitas)
        # Step 3: HSV Threshold
        self.btn_proses_hsv.clicked.connect(self.action_step3_hsv_threshold)
        # Step 4: Morfologi (masing-masing operasi sudah dihubungkan ke menu di atas)

        # Step 5: Deteksi & Klasifikasi
        self.btn_deteksi_jamur.clicked.connect(self.action_step5_deteksi_dan_klasifikasi)
        # Step 6: Reset aplikasi
        self.btn_reset_aplikasi.clicked.connect(self.action_step6_reset_aplikasi)

        # Set initial button states
        self.set_button_states(self.current_step)

    def load_detection_model(self):
        """
        Memuat model deteksi objek YOLOv8 dari path yang ditentukan.
        Menampilkan status di status bar dan label klasifikasi.
        """
        self.statusbar.showMessage("Inisialisasi: Memuat model deteksi objek...")
        QApplication.processEvents() # Perbarui GUI agar pesan status terlihat

        if YOLO_AVAILABLE:
            try:
                # Path ke model YOLOv8 (misal: 'best.pt' setelah melatih model custom)
                model_path = os.path.join(os.getcwd(), "models", "best.pt")
                if os.path.exists(model_path):
                    self.detection_model = YOLO(model_path)
                    self.statusbar.showMessage("Model deteksi objek YOLOv8 berhasil dimuat.")
                    self.label_hasil_klasifikasi.setText("Kualitas: Model Siap!")
                    self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: black;")
                else:
                    self.statusbar.showMessage("ERROR: Model deteksi objek tidak ditemukan. Fitur deteksi tidak aktif.", 5000)
                    self.label_hasil_klasifikasi.setText("Kualitas: Model Belum Siap!")
                    self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")
                    QMessageBox.warning(self, "Model Not Found",
                                        f"Model deteksi objek tidak ditemukan di:\n{model_path}\n"
                                        "Deteksi jamur tidak akan berfungsi. Harap latih model terlebih dahulu dan tempatkan di folder 'models'.")
                    print(f"ERROR: Model deteksi objek tidak ditemukan di {model_path}. Pastikan Anda telah melatih model dan menempatkannya dengan benar.")
            except Exception as e:
                self.statusbar.showMessage(f"ERROR memuat model: {e}", 5000)
                self.label_hasil_klasifikasi.setText("Kualitas: Gagal Memuat Model!")
                self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")
                QMessageBox.critical(self, "Error Loading Model", f"Gagal memuat model deteksi objek: {e}\n"
                                                                  "Pastikan instalasi 'ultralytics' dan file model benar.")
                print(f"Error during detection: {e}")
        else:
            self.statusbar.showMessage("ERROR: 'ultralytics' tidak tersedia. Deteksi objek tidak aktif.", 5000)
            self.label_hasil_klasifikasi.setText("Kualitas: Library Tidak Ada!")
            self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")

    def set_button_states(self, step):
        """
        Mengatur status (enabled/disabled) tombol berdasarkan langkah proses saat ini.
        """
        self.btn_pilih_gambar.setEnabled(step == 0)
        self.btn_proses_contrast.setEnabled(step == 1)
        self.btn_proses_hsv.setEnabled(step == 2)
        
        # Tombol Morfologi utama aktif setelah HSV (step 3) DAN mask HSV tidak kosong
        can_do_morph = (step >= 3 and self.current_image_hsv_mask is not None and np.count_nonzero(self.current_image_hsv_mask) > 0)
        self.btn_proses_morphology.setEnabled(can_do_morph)
        
        # Deteksi jamur aktif setelah minimal HSV (step 3)
        # ATAU setelah salah satu operasi morfologi telah dilakukan (current_image_morphed_mask tidak None)
        # DAN pastikan model deteksi tersedia
        self.btn_deteksi_jamur.setEnabled(
            (step >= 3 or self.current_image_morphed_mask is not None) and self.detection_model is not None
        )
        
        self.btn_reset_aplikasi.setEnabled(step >= 1) # Bisa reset kapan saja setelah pilih gambar

    def clear_image_labels(self):
        """
        Mengosongkan label gambar dan mengatur teks placeholder.
        """
        self.label_gambar_asli.clear()
        self.label_gambar_asli.setText("Gambar belum dipilih\nKlik 'Pilih Gambar Roti'")
        self.label_gambar_proses.clear()
        self.label_gambar_proses.setText("Hasil pemrosesan akan\nditampilkan di sini")

    def display_image_on_label(self, img, label_widget):
        """
        Menampilkan gambar OpenCV (numpy array) pada QLabel.
        Mengatasi gambar grayscale dan menyesuaikan ukuran agar sesuai dengan label.
        """
        if img is None:
            label_widget.clear()
            label_widget.setText("Gambar kosong")
            return
        if not isinstance(img, np.ndarray):
            label_widget.clear()
            label_widget.setText("Input bukan gambar valid")
            return

        # Pastikan array kontigu (contiguous) untuk QImage
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)

        # Ubah gambar grayscale ke BGR untuk tampilan yang konsisten (jika mask)
        if len(img.shape) == 2:
            img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_display = img.copy() 

        # Konversi BGR ke RGB karena QImage menggunakan format RGB
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Skala pixmap berdasarkan ukuran label_widget saat ini
        # Ini akan membuat gambar merespons perubahan ukuran label
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            label_widget.width(), label_widget.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label_widget.setPixmap(scaled_pixmap)
        label_widget.setAlignment(Qt.AlignCenter)

    # Step 1: Pilih gambar
    def action_step1_pilih_gambar(self):
        """
        Membuka dialog file untuk memilih gambar roti tawar.
        Memuat gambar dan menampilkannya di label gambar asli.
        """
        self.statusbar.showMessage("Langkah 1: Memilih gambar...")
        self.label_hasil_klasifikasi.setText("Kualitas: Menunggu Input...")
        self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: black;")
        QApplication.processEvents()

        initial_dir = os.getcwd() # Direktori awal adalah direktori kerja saat ini
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih Gambar Roti Tawar",
            initial_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if file_name:
            self.current_image_path = file_name
            self.current_image_original = cv2.imread(file_name)
            if self.current_image_original is not None:
                # Pastikan gambar asli selalu dalam format BGR
                if len(self.current_image_original.shape) == 2:
                    self.current_image_original = cv2.cvtColor(self.current_image_original, cv2.COLOR_GRAY2BGR)
                
                self.clear_image_labels() # Hapus konten label sebelumnya
                self.display_image_on_label(self.current_image_original, self.label_gambar_asli)
                
                # Reset semua hasil pemrosesan setelah gambar baru dipilih
                self.current_image_processed = None
                self.current_image_hsv_mask = None
                self.current_image_morphed_mask = None
                self.current_image_detection = None
                self.display_image_on_label(self.current_image_processed, self.label_gambar_proses) # Hapus tampilan proses

                self.label_hasil_klasifikasi.setText("Kualitas: Gambar Dimuat")
                self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: black;")
                self.statusbar.showMessage(f"Langkah 1 Selesai: Gambar '{os.path.basename(file_name)}' berhasil dimuat.")
                self.current_step = 1 # Pindah ke langkah berikutnya
                self.set_button_states(self.current_step)
            else:
                QMessageBox.warning(self, "Gagal Memuat Gambar", "Tidak dapat memuat gambar. Pastikan file valid atau tidak korup.")
                self.label_hasil_klasifikasi.setText("Kualitas: Gagal Memuat Gambar")
                self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")
                self.statusbar.showMessage("Langkah 1 Gagal: Gambar tidak dapat dimuat.")
                self.current_step = 0 # Kembali ke langkah awal jika gagal
                self.set_button_states(self.current_step)
        else:
            self.label_hasil_klasifikasi.setText("Kualitas: Pemilihan Gambar Dibatalkan")
            self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: orange;")
            self.statusbar.showMessage("Langkah 1 Dibatalkan: Pemilihan gambar dibatalkan.")
            # Tidak mengubah current_step jika pemilihan dibatalkan, agar state sebelumnya tetap ada
            if self.current_image_original is None: # Hanya reset step jika belum ada gambar sama sekali
                self.current_step = 0
                self.set_button_states(self.current_step)

    # Step 2: Perbaiki kualitas citra dengan Contrast Stretching
    def action_step2_perbaiki_kualitas(self):
        """
        Menerapkan Contrast Stretching pada gambar asli atau gambar yang terakhir diproses.
        Menampilkan hasilnya di label gambar proses.
        """
        if self.current_image_original is None:
            QMessageBox.information(self, "Informasi", "Harap pilih gambar terlebih dahulu (Langkah 1).")
            return

        self.statusbar.showMessage("Langkah 2: Memperbaiki kualitas citra dengan Contrast Stretching...")
        self.label_hasil_klasifikasi.setText("Kualitas: Memproses Contrast Stretching...")
        self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: blue;")
        QApplication.processEvents()

        start_time = time.time()
        # Menerapkan contrast stretching pada gambar asli (sekarang menggunakan CLAHE)
        enhanced_img = enhance_contrast_stretching(self.current_image_original)
        end_time = time.time()
        process_time = end_time - start_time

        if enhanced_img is not None:
            self.current_image_processed = enhanced_img # Simpan hasil contrast stretching
            self.display_image_on_label(enhanced_img, self.label_gambar_proses)
            self.label_hasil_klasifikasi.setText(f"Langkah 2 Selesai: Contrast Stretching ({process_time:.2f} detik)")
            self.statusbar.showMessage(f"Langkah 2 Selesai: Contrast Stretching selesai dalam {process_time:.2f} detik.")
            self.current_step = 2 # Pindah ke langkah berikutnya
            self.set_button_states(self.current_step)
            # Reset downstream images (mask dan deteksi) karena gambar dasar berubah
            self.current_image_hsv_mask = None
            self.current_image_morphed_mask = None
            self.current_image_detection = None
        else:
            QMessageBox.critical(self, "Error", "Gagal memproses Contrast Stretching.")
            self.label_hasil_klasifikasi.setText("Kualitas: Gagal Memproses")
            self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")

    # Step 3: Terapkan HSV Threshold
    def action_step3_hsv_threshold(self):
        """
        Menerapkan thresholding HSV pada gambar yang sudah diproses (Contrast Stretching).
        Menampilkan mask biner hasilnya.
        """
        if self.current_image_processed is None:
            QMessageBox.information(self, "Informasi", "Harap lakukan step 2 (Perbaiki Kualitas Citra) terlebih dahulu.")
            return

        self.statusbar.showMessage("Langkah 3: Menerapkan HSV Threshold...")
        self.label_hasil_klasifikasi.setText("Kualitas: Memproses HSV Threshold...")
        self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: blue;")
        QApplication.processEvents()

        start_time = time.time()
        # === PENTING: SESUAIKAN NILAI HSV INI SESUAI OBJEK YANG INGIN DIDESEKSI ===
        # Rentang HSV yang cocok untuk jamur/warna tertentu pada roti
        # Contoh: Warna hijau-kekuningan atau coklat muda bisa di sini
        lower_hsv = np.array([40, 50, 30])  
        upper_hsv = np.array([100, 255, 200]) 

        mask = apply_hsv_threshold(self.current_image_processed, lower_hsv, upper_hsv)
        end_time = time.time()
        process_time = end_time - start_time

        if mask is not None:
            self.current_image_hsv_mask = mask # Simpan mask HSV yang dihasilkan
            self.current_image_morphed_mask = None # Reset morfologi setelah HSV baru diterapkan
            
            # Tampilkan mask (perlu diubah ke BGR agar bisa ditampilkan oleh QLabel)
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.display_image_on_label(mask_display, self.label_gambar_proses)
            self.label_hasil_klasifikasi.setText(f"Langkah 3 Selesai: HSV Threshold ({process_time:.2f} detik)")
            self.statusbar.showMessage(f"Langkah 3 Selesai: HSV Threshold selesai dalam {process_time:.2f} detik.")
            self.current_step = 3 # Pindah ke langkah berikutnya
            self.set_button_states(self.current_step)
            self.current_image_detection = None # Reset deteksi
        else:
            QMessageBox.critical(self, "Error", "Gagal memproses HSV Threshold. Mask yang dihasilkan kosong.")
            self.label_hasil_klasifikasi.setText("Kualitas: Gagal Memproses HSV Threshold")
            self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")

    # --- Operasi Morfologi Individual (Dipanggil dari QMenu) ---
    def _apply_and_display_morphology(self, morph_func, op_name):
        """
        Fungsi helper untuk menerapkan dan menampilkan operasi morfologi tertentu.
        """
        if self.current_image_hsv_mask is None:
            QMessageBox.information(self, "Informasi", "Harap lakukan step 3 (HSV Threshold) terlebih dahulu.")
            return

        # Cek apakah mask HSV kosong sebelum mencoba operasi morfologi
        if np.count_nonzero(self.current_image_hsv_mask) == 0:
            QMessageBox.warning(self, "Peringatan", "Mask HSV kosong. Operasi morfologi tidak akan efektif. Silakan sesuaikan rentang HSV.")
            self.statusbar.showMessage(f"Morfologi Gagal: Mask HSV kosong. Silakan sesuaikan HSV threshold.", 5000)
            return

        self.statusbar.showMessage(f"Langkah 4: Menerapkan {op_name}...")
        self.label_hasil_klasifikasi.setText(f"Kualitas: Memproses {op_name}...")
        self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: blue;")
        QApplication.processEvents()

        start_time = time.time()
        # Setiap operasi morfologi mengambil mask HSV sebagai input dasar
        # Ini penting agar setiap tombol morfologi menunjukkan efeknya sendiri dari mask HSV awal.
        morphed_mask = morph_func(self.current_image_hsv_mask) 
        end_time = time.time()
        process_time = end_time - start_time

        if morphed_mask is not None:
            self.current_image_morphed_mask = morphed_mask # Simpan hasil morfologi terakhir
            # Tampilkan mask morfologi (perlu diubah ke BGR)
            morphed_display = cv2.cvtColor(morphed_mask, cv2.COLOR_GRAY2BGR)
            self.display_image_on_label(morphed_display, self.label_gambar_proses)
            self.label_hasil_klasifikasi.setText(f"Langkah 4 Selesai: {op_name} ({process_time:.2f} detik)")
            self.statusbar.showMessage(f"Langkah 4 Selesai: {op_name} selesai dalam {process_time:.2f} detik.")
            # current_step tetap 3, karena ini adalah sub-operasi dari langkah 4.
            # Namun, kita set ulang button states agar tombol deteksi aktif karena current_image_morphed_mask sudah terisi.
            self.set_button_states(self.current_step) 
            self.current_image_detection = None # Reset deteksi
        else:
            QMessageBox.critical(self, "Error", f"Gagal memproses operasi {op_name}.")
            self.label_hasil_klasifikasi.setText(f"Kualitas: Gagal Memproses {op_name}")
            self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")

    def action_morphology_erode(self):
        """Action untuk menerapkan operasi erosi."""
        self._apply_and_display_morphology(apply_morphology_erode, "Erosi")

    def action_morphology_dilate(self):
        """Action untuk menerapkan operasi dilasi."""
        self._apply_and_display_morphology(apply_morphology_dilate, "Dilasi")

    def action_morphology_opening(self):
        """Action untuk menerapkan operasi opening."""
        self._apply_and_display_morphology(apply_morphology_open, "Opening")

    def action_morphology_closing(self):
        """Action untuk menerapkan operasi closing."""
        self._apply_and_display_morphology(apply_morphology_close, "Closing")
    # --- Akhir Operasi Morfologi Individual ---

    # Step 5: Deteksi dan klasifikasi (Nomor kembali 5)
    def action_step5_deteksi_dan_klasifikasi(self):
        """
        Melakukan deteksi objek (jamur) menggunakan model YOLOv8 pada gambar yang diproses.
        Menggambar bounding box dan mengklasifikasikan kualitas roti.
        """
        # Deteksi YOLO sebaiknya dilakukan pada gambar asli atau hasil contrast stretching,
        # karena model YOLO butuh RGB/BGR citra asli.
        img_to_process_for_yolo = None
        if self.current_image_processed is not None:
            img_to_process_for_yolo = self.current_image_processed
        elif self.current_image_original is not None:
            img_to_process_for_yolo = self.current_image_original
        
        if img_to_process_for_yolo is None:
            QMessageBox.information(self, "Informasi", "Harap pilih dan proses gambar hingga Contrast Stretching terlebih dahulu (Langkah 2).")
            return

        if self.detection_model is None:
            QMessageBox.warning(self, "Model Tidak Siap", "Model deteksi belum dimuat atau ada kesalahan. Fitur ini dinonaktifkan.")
            return

        self.statusbar.showMessage("Langkah 5: Memulai deteksi jamur dengan YOLOv8...")
        self.label_hasil_klasifikasi.setText("Kualitas: Menganalisis Citra...")
        self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: black;")
        QApplication.processEvents()

        start_time = time.time()
        detected_objects = []
        try:
            results = self.detection_model(img_to_process_for_yolo, verbose=False)
            class_names_from_model = self.detection_model.names

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        class_name = class_names_from_model[cls_id]
                        xmin, ymin, xmax, ymax = map(int, xyxy)
                        detected_objects.append({
                            'name': class_name,
                            'bbox': (xmin, ymin, xmax, ymax),
                            'score': float(conf)
                        })
                else:
                    print("Tidak ada kotak deteksi ditemukan untuk gambar ini.")

            end_time = time.time()
            detection_time = end_time - start_time

            img_with_detection, kualitas_roti = draw_detections_and_classify(
                img_to_process_for_yolo, detected_objects, confidence_threshold=0.5
            )
            self.current_image_detection = img_with_detection
            self.display_image_on_label(img_with_detection, self.label_gambar_proses)

            if kualitas_roti == "SEGAR":
                self.label_hasil_klasifikasi.setText(f"Kualitas: {kualitas_roti}!")
                self.label_hasil_klasifikasi.setStyleSheet("color: green; font-weight: bold; font-size: 24px;")
                self.statusbar.showMessage(f"Langkah 5 Selesai: Deteksi & Klasifikasi dalam {detection_time:.2f} detik. Roti SEGAR.")
            else:
                self.label_hasil_klasifikasi.setText(f"Kualitas: {kualitas_roti}!")
                self.label_hasil_klasifikasi.setStyleSheet("color: red; font-weight: bold; font-size: 24px;")
                self.statusbar.showMessage(f"Langkah 5 Selesai: Deteksi & Klasifikasi dalam {detection_time:.2f} detik. Roti BERJAMUR!")

            self.current_step = 5 # Update current_step
            self.set_button_states(self.current_step)

        except Exception as e:
            QMessageBox.critical(self, "Error Deteksi", f"Terjadi kesalahan saat deteksi: {e}\n"
                                                         "Pastikan model telah dilatih dengan benar dan kelas cocok.")
            self.label_hasil_klasifikasi.setText("Kualitas: Gagal Deteksi!")
            self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")
            self.statusbar.showMessage(f"Langkah 5 Gagal: Terjadi kesalahan saat deteksi: {e}", 5000)
            print(f"Error during detection: {e}")

    # Step 6: Reset aplikasi (Nomor kembali 6)
    def action_step6_reset_aplikasi(self):
        """
        Mereset semua variabel gambar dan status aplikasi ke kondisi awal.
        """
        self.current_image_original = None
        self.current_image_processed = None
        self.current_image_hsv_mask = None
        self.current_image_morphed_mask = None
        self.current_image_detection = None
        self.current_image_path = None
        self.clear_image_labels()
        self.label_hasil_klasifikasi.setText("Kualitas: Menunggu Gambar...")
        self.label_hasil_klasifikasi.setStyleSheet("font-weight: bold; font-size: 18px; color: black;")
        self.statusbar.showMessage("Langkah 6 Selesai: Aplikasi telah direset. Siap untuk analisis baru.")
        self.current_step = 0 # Kembali ke langkah awal
        self.set_button_states(self.current_step)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
