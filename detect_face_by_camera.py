'''In this example, the program loads the "haarcascade_frontalface_default.xml" 
file which contains a pre-trained classifier for detecting faces. Then the program
 opens the default camera (0) and captures frames continuously in a while loop. 
 In each iteration of the loop, the program converts the current frame to grayscale, 
 applies the face detector to the grayscale image, and then draws rectangles around
  the detected faces. The output is displayed in a window named "Face Detection", 
  and the loop continues until the 'q' key is pressed.

Note: This is just a starting code and It might not work in your case if your
 haarcascade_frontalface_default.xml file is in different location, 
 
 Also you might want to add error handling and different checks as per requirement.
'''

import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier("data/haarcascade/haarcascade_frontalface_default.xml")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the output
    cv2.imshow("Face Detection", img)

    # Stop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()
