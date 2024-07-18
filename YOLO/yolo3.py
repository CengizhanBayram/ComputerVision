import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Paths
LabelsPath = r'C:\Users\cengh\Desktop\ComputerVison\YOLO\coco.names'
LABELS = open(LabelsPath).read().strip().split("\n")

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weight_path = r'C:\Users\cengh\Desktop\ComputerVison\YOLO\yolov3.weights'
cfg_path = r'C:\Users\cengh\Desktop\ComputerVison\YOLO\yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)

# Set our backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get all layer names
ln = net.getLayerNames()

# Print unconnected out layers to debug
unconnected_layers = net.getUnconnectedOutLayers()
print("Unconnected Out Layers:", unconnected_layers)

# Get output layers only
ln = [ln[i - 1] for i in unconnected_layers.flatten()]

my_path = r'C:\Users\cengh\Desktop\YOLO\images'
file_names = [f for f in listdir(my_path) if isfile(join(my_path, f))]

for file in file_names:
    # Full path
    full_path = join(my_path, file)
    # Load input image and grab its spatial dimensions
    image = cv2.imread(full_path)
    
    if image is None:
        print(f"Error loading image: {full_path}")
        continue

    (H, W) = image.shape[:2]

    # Construct a blob from our input image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # Set the input to our image blob
    net.setInput(blob)
    # Perform a forward pass through the network
    layerOutputs = net.forward(ln)

    # Initialize our lists of detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    IDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions by ensuring the detected probability is greater than a minimum threshold
            if confidence > 0.75:
                # Scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y) coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                IDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[IDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            text = "{}: {:.4f}".format(LABELS[IDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output image
    cv2.imshow("YOLO Detections", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
