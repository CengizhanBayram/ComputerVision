import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Input

# Dosya yolunu düzeltin
data_dir = r"C:\Users\cengh\Desktop\dataset\asl_dataset"

labels = []
images = []

# Tüm resimleri ve etiketleri yükleyin.
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        img = cv2.resize(img, (64, 64))  # Tüm resimleri aynı boyuta getirin.
        images.append(img)
        labels.append(folder)

images = np.array(images)
labels = np.array(labels)

# Orijinal etiket listesini kaydedin
unique_labels = list(np.unique(labels))

# Verileri normalleştirin ve etiketleri one-hot encode edin.
images = images / 255.0
labels = pd.get_dummies(labels).values

# Veriyi Eğitim ve Test Setlerine Bölme
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# TensorFlow/Keras kullanıyorsanız, verileri 4 boyutlu hale getirin (num_samples, height, width, num_channels)
X_train = X_train.reshape(-1, 64, 64, 1)
X_test = X_test.reshape(-1, 64, 64, 1)

# Modeli Oluşturma ve Eğitim
model = Sequential([
    Input(shape=(64, 64, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')  # Çıkış katmanındaki nöron sayısını etiket sayısına göre ayarlayın
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Modelin Değerlendirilmesi
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Eğitim ve doğrulama kayıplarını ve doğruluklarını görselleştirin.
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Modelin Kullanımı
def predict_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64, 64, 1) / 255.0
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    return predicted_label

# Canlı video akışı ile modelin kullanımı
cap = cv2.VideoCapture(0)
labels_dict = {i: label for i, label in enumerate(unique_labels)}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Görüntüyü tahmin etmek için kullanın
    prediction = predict_image(frame)
    label = labels_dict[prediction]
    
    # Tahmini görüntüde gösterin
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('ASL Translation', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
