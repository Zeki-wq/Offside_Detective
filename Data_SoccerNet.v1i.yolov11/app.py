import sys
import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QFileDialog,
    QFrame, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from ultralytics import YOLO

# =====================
# SABİT AYARLAR (eğim1.py ile BİREBİR AYNI)
# =====================
DEFAULT_MODEL_PATH = r"C:\Users\Zeki\OneDrive\Masaüstü\9kfoto\OfsaytProjesi\v11_egitim\weights\best.pt"
FRAME_W, FRAME_H = 640, 360
CONF_TH = 0.3
EPS = 10          # 🔥 VAR toleransı (eğim1.py ile birebir)
FINAL_MARGIN = 2  # 🔥 karar marjı

# =====================
# ARKA PLAN ANALİZ İŞ PARÇACIĞI (UI DONMAZ)
# =====================
class VarAnalysisThread(QThread):
    # Sinyaller (UI güncellemeleri için)
    progress_signal = pyqtSignal(int, int)          # (işlenen kare, toplam kare)
    stats_signal = pyqtSignal(int, int, int, int)   # (işlenen, geçerli, offside, onside)
    log_signal = pyqtSignal(str)                    # terminal log satırı
    finished_signal = pyqtSignal(dict)              # özet sonuç verisi
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, model_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            if not os.path.exists(self.model_path):
                self.error_signal.emit(f"Model dosyası bulunamadı:\n{self.model_path}")
                return

            self.log_signal.emit(">> YOLO Modeli yükleniyor...")
            model = YOLO(self.model_path)
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_signal.emit("Video dosyası açılamadı!")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.log_signal.emit(f">> Video açıldı. Toplam kare sayısı: {total_frames}")
            self.log_signal.emit(">> VAR Analizi başladı...\n" + "="*50)

            # Sayaçlar (eğim1.py ile birebir)
            offside_frames = 0
            onside_frames = 0
            valid_frames = 0
            processed_frames = 0
            prev_ball_x = None

            start_time = time.time()

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frames += 1

                # Çözünürlük ve Skor Tablosu Maskelemesi (eğim1.py birebir)
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                frame[0:90, 480:640] = 0

                results = model(frame, verbose=False)

                ball = None
                goalkeeper = None
                players = []

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    clss = results[0].boxes.cls.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()

                    for i in range(len(boxes)):
                        if confs[i] < CONF_TH:
                            continue
                        x1, y1, x2, y2 = boxes[i]
                        cx = (x1 + x2) / 2
                        foot = (cx, y2)
                        cid = int(clss[i])

                        if cid == 0:
                            ball = foot
                        elif cid == 1:
                            goalkeeper = foot
                        elif cid == 3:
                            players.append(foot)

                if ball is None or len(players) < 2:
                    if processed_frames % 5 == 0:
                        self.progress_signal.emit(processed_frames, total_frames)
                        self.stats_signal.emit(processed_frames, valid_frames, offside_frames, onside_frames)
                    continue

                ball_x = ball[0]

                # =====================
                # PAS ANI FİLTRESİ
                # =====================
                if prev_ball_x is not None and abs(ball_x - prev_ball_x) > 6:
                    prev_ball_x = ball_x
                    continue
                prev_ball_x = ball_x

                # =====================
                # HÜCUM YÖNÜ
                # =====================
                if goalkeeper is not None:
                    hucum_yonu = 'r' if goalkeeper[0] < FRAME_W / 2 else 'l'
                else:
                    hucum_yonu = 'r' if ball_x < FRAME_W / 2 else 'l'

                # =====================
                # SAVUNMA HATTI
                # =====================
                if hucum_yonu == 'r':
                    defenders = sorted(players, key=lambda p: p[0])[:2]
                    second_def_x = max(d[0] for d in defenders)
                else:
                    defenders = sorted(players, key=lambda p: p[0], reverse=True)[:2]
                    second_def_x = min(d[0] for d in defenders)

                # =====================
                # HÜCUMCU
                # =====================
                attackers = [p for p in players if p not in defenders]
                if not attackers:
                    continue

                attackers.sort(key=lambda p: abs(p[0] - ball_x))
                attacker_x = attackers[0][0]

                # =====================
                # GERÇEK VAR HESABI (eğim1.py birebir)
                # =====================
                if hucum_yonu == 'r':
                    offside_line = max(ball_x, second_def_x)
                    offside = attacker_x > offside_line + EPS
                else:
                    offside_line = min(ball_x, second_def_x)
                    offside = attacker_x < offside_line - EPS

                valid_frames += 1
                if offside:
                    offside_frames += 1
                else:
                    onside_frames += 1

                # 🔁 KULLANICININ ÖZEL ÇIKTI ŞEMASI (Ters yazdırma kuralı korunuyor)
                log_text = (
                    f"Frame {processed_frames:04d} | Ball:{ball_x:.1f} Att:{attacker_x:.1f} "
                    f"Def:{second_def_x:.1f} Line:{offside_line:.1f} Karar:{'ONSIDE' if offside else 'OFFSIDE'}"
                )
                self.log_signal.emit(log_text)

                if processed_frames % 2 == 0:
                    self.progress_signal.emit(processed_frames, total_frames)
                    self.stats_signal.emit(processed_frames, valid_frames, offside_frames, onside_frames)

            cap.release()
            elapsed = time.time() - start_time

            # Son durum güncellemesi
            self.progress_signal.emit(processed_frames, total_frames)
            self.stats_signal.emit(processed_frames, valid_frames, offside_frames, onside_frames)

            # =====================
            # SON KARAR (eğim1.py birebir şeması)
            # =====================
            if valid_frames == 0:
                final_decision = "UNKNOWN"
                decision_text = "❓ KARAR VERİLEMEDİ"
            elif offside_frames > onside_frames + FINAL_MARGIN:
                final_decision = "ONSIDE"
                decision_text = "✅ SON KARAR: ONSIDE"
            else:
                final_decision = "OFFSIDE"
                decision_text = "🚩 SON KARAR: OFFSIDE"

            result = {
                "decision": final_decision,
                "decision_text": decision_text,
                "valid_frames": valid_frames,
                "offside_frames": offside_frames,
                "onside_frames": onside_frames,
                "processed_frames": processed_frames,
                "elapsed": elapsed
            }
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(f"Hata oluştu: {str(e)}")


# =====================
# SÜRÜKLE - BIRAK ALANI
# =====================
class DropArea(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #3b82f6;
                border-radius: 12px;
                background-color: #111827;
                padding: 20px;
            }
            QFrame:hover {
                border-color: #60a5fa;
                background-color: #1e293b;
            }
        """)
        layout = QVBoxLayout()
        self.label = QLabel("📁 Videoyu Buraya Sürükleyip Bırakın\nveya Seçmek İçin Tıklayın")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #93c5fd; font-size: 15px; font-weight: bold;")
        layout.addWidget(self.label)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Video Seç", "", "Video Dosyaları (*.mp4 *.avi *.mov *.mkv);;Tüm Dosyalar (*.*)"
        )
        if file_path:
            self.file_dropped.emit(file_path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)

    def set_file_name(self, file_path):
        name = os.path.basename(file_path)
        self.label.setText(f"🎬 Seçilen Video:\n{name}")
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #10b981;
                border-radius: 12px;
                background-color: #064e3b;
                padding: 20px;
            }
        """)
        self.label.setStyleSheet("color: #a7f3d0; font-size: 15px; font-weight: bold;")


# =====================
# ANA PENCERE
# =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI VAR Sistemi - Ofsayt Tespit Arayüzü")
        self.resize(1000, 720)
        self.selected_video_path = None
        self.thread = None

        self.init_ui()

    def init_ui(self):
        # Koyu ve modern tema
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0f19;
            }
            QLabel {
                color: #f3f4f6;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #374151;
                color: #9ca3af;
            }
            QProgressBar {
                border: 1px solid #374151;
                border-radius: 6px;
                text-align: center;
                color: white;
                background-color: #1f2937;
                height: 22px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 5px;
            }
            QTextEdit {
                background-color: #030712;
                color: #38bdf8;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 8px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 1. Başlık Kartı
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e3a8a, stop:1 #0f172a);
                border-radius: 10px;
                border: 1px solid #3b82f6;
                padding: 12px;
            }
        """)
        h_layout = QVBoxLayout()
        title_label = QLabel("⚽ AI VAR OFSAYT TESPİT SİSTEMİ")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: white; font-size: 20px; font-weight: 800; letter-spacing: 1px;")
        sub_label = QLabel("YOLOv11 & Bilgisayarlı Görü Tabanlı Masaüstü Pozisyon Analizi")
        sub_label.setAlignment(Qt.AlignCenter)
        sub_label.setStyleSheet("color: #93c5fd; font-size: 12px;")
        h_layout.addWidget(title_label)
        h_layout.addWidget(sub_label)
        header_frame.setLayout(h_layout)
        main_layout.addWidget(header_frame)

        # 2. Sürükle Bırak ve Başlat Butonu
        top_h_layout = QHBoxLayout()
        
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self.on_file_selected)
        top_h_layout.addWidget(self.drop_area, 3)

        btn_vbox = QVBoxLayout()
        self.btn_start = QPushButton("🚀 VAR Analizini Başlat")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_analysis)
        btn_vbox.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹️ Durdur")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #dc2626;")
        self.btn_stop.clicked.connect(self.stop_analysis)
        btn_vbox.addWidget(self.btn_stop)

        top_h_layout.addLayout(btn_vbox, 1)
        main_layout.addLayout(top_h_layout)

        # 3. İlerleme Çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Beklemede... %p%")
        main_layout.addWidget(self.progress_bar)

        # 4. Sayaç Kartları (Grid)
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #111827;
                border-radius: 8px;
                border: 1px solid #1f2937;
                padding: 6px;
            }
        """)
        stats_layout = QGridLayout()

        self.lbl_processed = QLabel("0")
        self.lbl_valid = QLabel("0")
        self.lbl_offside = QLabel("0")
        self.lbl_onside = QLabel("0")

        for lbl in [self.lbl_processed, self.lbl_valid, self.lbl_offside, self.lbl_onside]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #60a5fa;")

        self.lbl_offside.setStyleSheet("font-size: 18px; font-weight: bold; color: #f87171;")
        self.lbl_onside.setStyleSheet("font-size: 18px; font-weight: bold; color: #4ade80;")

        def make_title(t):
            l = QLabel(t)
            l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet("color: #9ca3af; font-size: 11px;")
            return l

        stats_layout.addWidget(make_title("İşlenen Kare"), 0, 0)
        stats_layout.addWidget(self.lbl_processed, 1, 0)

        stats_layout.addWidget(make_title("Geçerli Frame"), 0, 1)
        stats_layout.addWidget(self.lbl_valid, 1, 1)

        stats_layout.addWidget(make_title("OFFSIDE Frame"), 0, 2)
        stats_layout.addWidget(self.lbl_offside, 1, 2)

        stats_layout.addWidget(make_title("ONSIDE Frame"), 0, 3)
        stats_layout.addWidget(self.lbl_onside, 1, 3)

        stats_frame.setLayout(stats_layout)
        main_layout.addWidget(stats_frame)

        # 5. Büyük VAR Karar Rozeti
        self.decision_card = QLabel("Pozisyon Analizi Bekleniyor...")
        self.decision_card.setAlignment(Qt.AlignCenter)
        self.decision_card.setFixedHeight(60)
        self.decision_card.setStyleSheet("""
            background-color: #1f2937;
            color: #9ca3af;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            border: 1px solid #374151;
        """)
        main_layout.addWidget(self.decision_card)

        # 6. Canlı Terminal Konsolu
        log_label = QLabel("📜 Canlı VAR Terminal Çıktıları:")
        log_label.setStyleSheet("color: #cbd5e1; font-size: 12px; font-weight: bold;")
        main_layout.addWidget(log_label)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        main_layout.addWidget(self.log_console)

        central_widget.setLayout(main_layout)

    def on_file_selected(self, file_path):
        self.selected_video_path = file_path
        self.drop_area.set_file_name(file_path)
        self.btn_start.setEnabled(True)
        self.log_console.append(f">> Video seçildi: {file_path}")

    def start_analysis(self):
        if not self.selected_video_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir video dosyası seçin!")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_console.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Analiz Ediliyor... %p%")

        self.decision_card.setText("⏳ VAR Analizi Yapılıyor...")
        self.decision_card.setStyleSheet("""
            background-color: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            border: 2px solid #3b82f6;
        """)

        # Thread başlat
        self.thread = VarAnalysisThread(self.selected_video_path, DEFAULT_MODEL_PATH)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.stats_signal.connect(self.update_stats)
        self.thread.log_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.error_signal.connect(self.on_error)
        self.thread.start()

    def stop_analysis(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.append_log(">> Analiz kullanıcı tarafından durduruldu.")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def update_progress(self, current, total):
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"Analiz: %{pct} ({current}/{total} kare)")

    def update_stats(self, processed, valid, offside, onside):
        self.lbl_processed.setText(str(processed))
        self.lbl_valid.setText(str(valid))
        self.lbl_offside.setText(str(offside))
        self.lbl_onside.setText(str(onside))

    def append_log(self, text):
        self.log_console.append(text)
        # Otomatik en alta kaydır
        self.log_console.verticalScrollBar().setValue(
            self.log_console.verticalScrollBar().maximum()
        )

    def on_finished(self, res):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Analiz Tamamlandı! (%p%)")

        decision = res["decision"]
        text = res["decision_text"]

        if decision == "ONSIDE":
            self.decision_card.setText(text)
            self.decision_card.setStyleSheet("""
                background-color: rgba(34, 197, 94, 0.2);
                color: #22c55e;
                font-size: 22px;
                font-weight: bold;
                border-radius: 8px;
                border: 2px solid #22c55e;
            """)
        elif decision == "OFFSIDE":
            self.decision_card.setText(text)
            self.decision_card.setStyleSheet("""
                background-color: rgba(239, 68, 68, 0.2);
                color: #ef4444;
                font-size: 22px;
                font-weight: bold;
                border-radius: 8px;
                border: 2px solid #ef4444;
            """)
        else:
            self.decision_card.setText(text)
            self.decision_card.setStyleSheet("""
                background-color: rgba(234, 179, 8, 0.2);
                color: #eab308;
                font-size: 22px;
                font-weight: bold;
                border-radius: 8px;
                border: 2px solid #eab308;
            """)

        self.append_log("\n" + "="*50)
        self.append_log(f">> ANALİZ BİTTİ (Süre: {res['elapsed']:.2f} sn)")
        self.append_log(f">> {text}")
        self.append_log("="*50)

    def on_error(self, err_msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Hata", err_msg)
        self.append_log(f">> HATA: {err_msg}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
