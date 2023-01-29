"""
This program uses OpenCV to detect faces in a video stream from the camera and then 
uses a pre-trained mood model to predict the mood of the person in the detected face.
 The mood model is loaded using the load_model function from the Keras library and it
  should be a file that you have trained or downloaded.

It starts by opening the default camera (0) and captures frames continuously in a
 while loop. In each iteration of the loop, the program converts the current frame to 
 grayscale, applies the face detector to the grayscale image, and then draws rectangles
  around the detected faces. Then for each face, it gets the region of interest(ROI) of
   the face, resizes it to 48x48 pixels, and then uses the mood model to predict the mood 
   of the person in the face. Finally, it displays the mood on the top of the face in the
    video stream. The output is displayed in a window named "Face and Mood Detection", 
    and the loop continues until the 'q' key is pressed.

Note: This is just a starting code and It might not work in your case if your
 haarcascade_frontalface_default.xml file or mood_model.h5 file is in different location,
  Also you might want to add
"""

import cv2
import numpy as np
from keras.models import load_model

# Load the cascade
face_cascade = cv2.CascadeClassifier("data/haarcascade/haarcascade_frontalface_default.xml")

# Load the mood model
#pip install keras-models
model = load_model("data/model/EmotionDetectionModel.h5")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        # Predict the mood
        mood = model.predict(roi_gray.reshape(1, 48, 48, 1))
        mood = np.argmax(mood)

        # Display the mood
        cv2.putText(img, str(mood), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the output
    cv2.imshow("Face and Mood Detection", img)

    # Stop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()
