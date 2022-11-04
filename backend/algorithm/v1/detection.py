import cv2

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar_face_cascade = cv2.CascadeClassifier('algorithm/haarcascade_frontalface_default.xml')
    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w], faces[0]

def detect_faces(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    haar_face_cascade = cv2.CascadeClassifier('algorithm/haarcascade_frontalface_default.xml')
    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_copy