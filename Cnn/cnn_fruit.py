from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
import json

# Dataset paths
train_path = r"C:\Users\cengh\Desktop\ComputerVison\Cnn\fruits-360\Training/"
test_path = r"C:\Users\cengh\Desktop\ComputerVison\Cnn\fruits-360\Test/"

# Load and display an example image
img = load_img(r'C:\Users\cengh\Desktop\ComputerVison\Cnn\fruits-360\Training\Apple Braeburn\0_100.jpg')
plt.imshow(img)
plt.axis("off")
plt.show()

# Convert image to array and print its shape
x = img_to_array(img)
print("Image shape:", x.shape)

# Get class names
class_names = glob(train_path + '/*' )
num_classes = len(class_names)
print("Number of classes:", num_classes)

#%% CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))  # Output layer
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

batch_size = 32

#%% Data Generation - Train - Test
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.3,
                                   horizontal_flip=True,
                                   zoom_range=0.3)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, 
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
    test_path, 
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")

# Train the model using a while loop to ensure it continues indefinitely
epochs = 100
steps_per_epoch = len(train_generator)

history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_loss = []
    epoch_acc = []
    val_loss = []
    val_acc = []

    # Training loop
    train_generator.reset()  # Reset generator for each epoch
    for i in range(steps_per_epoch):
        X_batch, y_batch = train_generator.next()
        train_step = model.train_on_batch(X_batch, y_batch)
        epoch_loss.append(train_step[0])
        epoch_acc.append(train_step[1])

    # Validation loop
    test_generator.reset()
    val_steps = len(test_generator)
    for i in range(val_steps):
        X_val_batch, y_val_batch = test_generator.next()
        val_step = model.test_on_batch(X_val_batch, y_val_batch)
        val_loss.append(val_step[0])
        val_acc.append(val_step[1])

    # Compute epoch metrics
    history["loss"].append(sum(epoch_loss) / len(epoch_loss))
    history["accuracy"].append(sum(epoch_acc) / len(epoch_acc))
    history["val_loss"].append(sum(val_loss) / len(val_loss))
    history["val_accuracy"].append(sum(val_acc) / len(val_acc))

    print(f"Loss: {history['loss'][-1]}, Accuracy: {history['accuracy'][-1]}")
    print(f"Validation Loss: {history['val_loss'][-1]}, Validation Accuracy: {history['val_accuracy'][-1]}")

# Save model weights
model.save_weights("deneme.h5")

# Plot training history
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Save history to JSON
with open("deneme.json", "w") as f:
    json.dump(history, f)

# Load history and plot
with open("deneme.json", "r") as f:
    h = json.load(f)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(h["accuracy"], label="Train Accuracy")
plt.plot(h["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
