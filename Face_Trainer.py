import cv2
import numpy as np
import os
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.face.createLBPHFaceRecognizer()

Face_ID = -1
prev_person_name = ""
y_ID = []
x_train = []

Face_Images = os.path.join(os.getcwd(), "Face_Images")
print(Face_Images)

for root, dirs, files in os.walk(Face_Images):
    for file in files:
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            person_name = os.path.basename(root)
            print(path, person_name)

            if prev_person_name != person_name:
                Face_ID += 1
                prev_person_name = person_name

            gray_image = Image.open(path).convert("L")
            crop_image = gray_image.resize((800, 800), Image.ANTIALIAS)
            final_image = np.array(crop_image, "uint8")
            faces = face_cascade.detectMultiScale(final_image, scaleFactor=1.5, minNeighbors=5)
            print(Face_ID, faces)

            for (x, y, w, h) in faces:
                roi = final_image[y:y + h, x:x + w]
                x_train.append(roi)
                y_ID.append(Face_ID)

recognizer.train(x_train, np.array(y_ID))
recognizer.write("face-trainer.yml")