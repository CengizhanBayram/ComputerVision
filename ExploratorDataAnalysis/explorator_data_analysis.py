import pandas as pd
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sütun isimlerini belirleme
col_list = ["frame_number", "identity_number", "left", "top", "width", "height", "score", "class", "visibility"]

# CSV dosyasını okuma
data_path = r"C:\Users\cengh\Desktop\ComputerVison\ExploratorDataAnalysis\gt.txt"
data = pd.read_csv(data_path, names=col_list)

plt.figure()
sns.countplot(data["class"])

# Sınıfı 3 olan verileri filtreleme (Araba sınıfı)
car = data[data["class"] == 3]

video_path = r'C:\Users\cengh\Desktop\ComputerVison\ExploratorDataAnalysis\deneme.mp4'
cap = cv2.VideoCapture(video_path)
id1 = 29
number_of_image = np.max(data["frame_number"])
fps = 25
bound_box_list = []

for i in range(number_of_image - 1):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, dsize=(960, 540))
        filter_id1 = np.logical_and(car["frame_number"] == i + 1, car["identity_number"] == id1)
        if len(car[filter_id1]) != 0:
            x = int(car[filter_id1].left.values[0] / 2)
            y = int(car[filter_id1].top.values[0] / 2)
            w = int(car[filter_id1].width.values[0] / 2)
            h = int(car[filter_id1].height.values[0] / 2)
            
            # Nesnenin bounding box'ını çizin
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Nesnenin merkez noktasını çizin
            cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 0, 255), -1)
            
            # Bounding box bilgilerini listeye ekleyin
            bound_box_list.append([i, x, y, w, h, int(x + w / 2), int(y + h / 2)])
            
            # Kare numarasını ekranda gösterin
            cv2.putText(frame, "frame num:" + str(i + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Kareyi göster
        cv2.imshow('Frame', frame)
        
        # 'q' tuşuna basarak çıkış yapma imkanı tanır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
