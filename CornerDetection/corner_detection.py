import cv2 
import matplotlib.pyplot as plt
import numpy as np

# Görüntüyü okuyun
img = cv2.imread(r"C:\Users\cengh\Desktop\ComputerVison\CornerDetection\img.jpg")

# Görüntünün başarıyla yüklenip yüklenmediğini kontrol edin
if img is None:
    print("Görüntü yüklenemedi. Lütfen dosya yolunu kontrol edin.")
else:
    # Görüntüyü BGR'den RGB'ye dönüştürün
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Görüntü boyutunu yazdırın
    print("Orijinal görüntü boyutu:", img.shape)
    
    # Görüntüyü gösterin
    plt.figure()
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Orijinal Görüntü")
    plt.show()

    # Görüntüyü gri tonlamalıya çevirin
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # float32'ye çevirin
    gray = np.float32(gray)

    # Harris köşe algılama
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Köşeleri daha belirgin hale getirin (köşe noktalarını büyütmek için)
    dst = cv2.dilate(dst, None)

    # Eşik belirleme: tespit edilen köşeleri orijinal görüntüde işaretleme
    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Kırmızı renk

    # Köşeleri gösterin
    img_rgb_corners = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img_rgb_corners)
    plt.axis("off")
    plt.title("Harris Köşe Algılama")
    plt.show()
