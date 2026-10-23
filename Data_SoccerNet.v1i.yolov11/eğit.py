from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Ekran kartının (CUDA) aktif olup olmadığını kontrol et
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Eğitim şunun üzerinde başlıyor: {device}")

    # YOLOv11 Nano modeli (RTX 2050 için en ideali)
    model = YOLO('yolo11n.pt') 

    # Eğitimi başlat
    model.train(
        data=r'C:\Users\Zeki\OneDrive\Masaüstü\9kfoto\Data_SoccerNet.v1i.yolov11\data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,      # Hafıza hatası alırsan burayı 4 yap kanka
        device=device,
        workers=2,
        project='OfsaytProjesi',
        name='v11_egitim'
    )