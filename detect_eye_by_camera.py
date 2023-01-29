"""
This code will open the default camera and detect eyes in real-time video feed.
 It will show the live video feed with rectangle around the eyes.

Note: This is just an example and you need to provide the path of haarcascade_eye.
xml file.

You can adjust the parameters of the detectMultiScale function to adjust the
 sensitivity of the eye detection.
Also you can change the camera by changing the index passed in cv2.VideoCapture(0)
 to the desired camera
"""


import cv2

# Load the cascade classifiers for detecting eyes
eye_cascade = cv2.CascadeClassifier("data/haarcascade/haarcascade_eye.xml")

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the output frame
    cv2.imshow("Eyes", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
