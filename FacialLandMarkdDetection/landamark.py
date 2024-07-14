import cv2
import dlib
import numpy as np

PREDICTOR_PATH = r'C:\Users\cengh\Desktop\ComputerVison\FacialLandMarkdDetection\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(255, 0, 255))
    return im


cap = cv2.VideoCapture(0)

# If you want to use the webcam, use 0 as the argument
# cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        landmarks = get_landmarks(frame)
        frame_with_landmarks = annotate_landmarks(frame, landmarks)
        
        cv2.imshow('Video with Landmarks', frame_with_landmarks)
    except TooManyFaces:
        print("Too many faces detected.")
    except NoFaces:
        print("No faces detected.")
        cv2.imshow('Video with Landmarks', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
