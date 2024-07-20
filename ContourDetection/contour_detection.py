import cv2
import numpy as np

# Web kamerasını aç
cap = cv2.VideoCapture(0)

while True:
    # Bir kare yakala
    ret, goruntu = cap.read()

    # Gri tonlamaya dönüştür
    gri_ton = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)

    # Kenarları algıla
    kenarlar = cv2.Canny(gri_ton, 50, 150)

    # Konturları bul
    konturlar, _ = cv2.findContours(kenarlar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Her kontur için
    for i, kontur in enumerate(konturlar):
        # Konturun alanını hesapla
        alan = cv2.contourArea(kontur)

        # Alanı belirli bir eşik değerin üzerindeyse çiz
        if alan > 500:
            # Dikdörtgen çizin
            x, y, w, h = cv2.boundingRect(kontur)
            cv2.rectangle(goruntu, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Sonucu göster
    cv2.imshow('Canlı Kontur Algılama', goruntu)

    # Q tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
cap.release()
cv2.destroyAllWindows()
