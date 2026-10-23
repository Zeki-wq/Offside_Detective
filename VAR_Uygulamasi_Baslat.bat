@echo off
chcp 65001 > nul
title AI VAR Ofsayt Tespit Sistemi (Masaüstü)
echo ========================================================
echo       AI VAR OFSAYT TESPIT MASAUSTU UYGULAMASI
echo ========================================================
echo.
cd /d "%~dp0Data_SoccerNet.v1i.yolov11"
python app.py
if errorlevel 1 (
    echo.
    echo Bir hata olustu. Eger Python yolu bulunamadiysa tam yol ile calistirmayi deneyin:
    "C:\Users\Zeki\AppData\Local\Programs\Python\Python312\python.exe" app.py
)
pause
