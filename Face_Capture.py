import cv2
import os

directory = '/Users/harsh/Development/Face-Detection-on-Raspberry-Pi/Face_Images/suraj'
image_count = 1

if not os.path.exists(directory):
    os.makedirs(directory)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        image_path = os.path.join(directory, f"{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")
        image_count += 1

cap.release()
cv2.destroyAllWindows()
