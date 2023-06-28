import cv2
import numpy as np
from datetime import datetime

labels = ["suraj", "harsh", "srinivas"]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainer.yml")

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX  # Define the font outside the loop

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    name = ""  # Initialize name as an empty string

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)

        if conf >= 80:
            name = labels[id_]
            cv2.putText(img, name, (x, y), font, 1, (0, 0, 255), 2)
            print("Detected:", name)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Preview', img)

    # Create a new window and display the welcome message
    name_window = np.zeros((100, 400, 3), dtype=np.uint8)  # Create a black window

    # Get the current time and determine the greeting
    now = datetime.now()
    current_time = now.time()

    greeting = ""
    if current_time.hour < 12:
        greeting = "Good Morning"
    else:
        greeting = "Good Evening"

    welcome_message = greeting + ", " + name
    cv2.putText(name_window, welcome_message, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Display the welcome message
    cv2.imshow('Welcome', name_window)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
