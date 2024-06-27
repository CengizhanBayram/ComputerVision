import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt

# Giriş ve çıkış yolları
pathIn = r"img1"
pathOut = "deneme.mp4"

# Dosyaları listele ve sıralama
files = sorted([f for f in os.listdir(pathIn) if isfile(join(pathIn, f))])

# Dosya sayısını kontrol et
"""if len(files) > 44:
    # Dosya yolunu doğru şekilde birleştir
    img_path = join(pathIn, files[44])
    
    # Dosya var mı kontrol et
    if not isfile(img_path):
        print(f"Dosya mevcut değil: {img_path}")
        exit()

    # Görüntüyü okuma
    img = cv2.imread(img_path)

    if img is None:
        print(f"Görüntü dosyası okunamadı: {img_path}")
        exit()

    # BGR'den RGB'ye dönüştürme
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Görüntüyü gösterme
    plt.imshow(img_rgb)
    plt.axis('off')  # Eksenleri gizle
    plt.show()
else:
    print("Yeterli dosya yok.")
"""
fps = 25 
size = (1920,1080)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for i in files:
    print(i)
    filename = pathIn + "\\" + i
    img = cv2.imread(filename)
    out.write(img)

out.release()    
