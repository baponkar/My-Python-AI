import cv2

import numpy as np

# Load the cascade

face_cascade = cv2.CascadeClassifier('./data/haarcascade/haarcascade_frontalface_default.xml')

# Load mood classifier

mood_classifier = cv2.dnn.readNetFromCaffe('mood_classifier.prototxt', 
'mood_classifier.caffemodel')

# Define mood labels

mood_labels = ["neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful"]

# Define the detection function

def detect_mood(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi = frame[y:y+h, x:x+w]

        blob = cv2.dnn.blobFromImage(cv2.resize(roi, (48, 48)), 1.0, (48, 48), (104.0, 177.0, 
123.0))

        mood_classifier.setInput(blob)

        mood_predictions = mood_classifier.forward()

        mood_index = np.argmax(mood_predictions[0])

        mood = mood_labels[mood_index]

        cv2.putText(frame, mood, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

# Capture video from the camera

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    frame = detect_mood(frame)

    cv2.imshow('Mood Detection', frame)

    if cv2.waitKey(1) == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()
