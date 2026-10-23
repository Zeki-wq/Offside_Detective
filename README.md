# ⚽ AI VAR - Automated Offside Detection System

An automated Computer Vision & Deep Learning system for football (soccer) offside detection using fine-tuned **YOLOv11** object detection and geometric spatial analysis.

---

## 📌 Overview

This project implements an intelligent Video Assistant Referee (VAR) offside analysis tool. It processes match video feeds frame-by-frame to accurately detect key entities (ball, goalkeeper, outfield players), establishes dynamic defensive and attacking reference lines, and determines offside positions in real time.

```
       [ Match Video Frame ]
                 │
                 ▼
        [ YOLOv11 Detector ]
     (Ball, Goalkeeper, Players)
                 │
                 ▼
    [ Spatial & Vector Analysis ]
  • Dynamic Attack Direction Estimation
  • Second-to-last Defender Line
  • Offside Plane Calculation
                 │
                 ▼
      [ Automated VAR Decision ]
          (ONSIDE / OFFSIDE)
```

---

## 🚀 Key Features

- **Fine-Tuned YOLOv11 Architecture:** Specialized detector trained on football broadcast datasets to track players, goalkeepers, and footballs across varying camera angles.
- **Dynamic Orientation Tracking:** Automatically detects field orientation and attack directions based on positional vectors and goalkeeper coordinates.
- **Pass Moment Filtering:** High-speed ball displacement filters to eliminate transitional noise during passes.
- **Desktop GUI (PyQt5):** Native, zero-terminal desktop application with drag-and-drop video loading, live progress tracking, and instantaneous VAR decision cards.
- **Terminal Batch Analyzer:** Standalone terminal engine for rapid frame-by-frame quantitative analysis.

---

## 🛠️ Project Structure

```text
├── Data_SoccerNet.v1i.yolov11/
│   ├── app.py                # Native PyQt5 Desktop GUI application
│   ├── eğim1.py              # Quantitative terminal VAR analyzer
│   ├── eğitim2.py            # Homography & perspective module
│   └── data.yaml             # Dataset class configurations
├── OfsaytProjesi/
│   └── v11_egitim/
│       └── weights/
│           └── best.pt       # Fine-tuned YOLOv11 weights (best checkpoint)
├── AI_VAR_Ofsayt.lnk         # Direct Desktop Launcher (Windowless / EXE style)
├── VAR_Uygulamasi.vbs        # Zero-console launcher script
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ai-var-offside-detection.git
   cd ai-var-offside-detection
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 How to Run

### Option 1: Native Desktop Application (Recommended)
Double-click `AI_VAR_Ofsayt.lnk` or run:
```bash
python Data_SoccerNet.v1i.yolov11/app.py
```
- Drag and drop your match video into the application window.
- Click **🚀 VAR Analizini Başlat**.
- View real-time frame telemetry and the final VAR decision.

### Option 2: Terminal Batch Engine
```bash
python Data_SoccerNet.v1i.yolov11/eğim1.py
```

---

## 📊 Detection Classes

The underlying YOLOv11 model is configured to detect 5 distinct pitch entities:
1. `ball`
2. `goalkeeper`
3. `player`
4. `referee`
5. `other`

---

## ⚖️ License & Disclaimer

This project is developed for educational, academic, and sports analytics research purposes.
