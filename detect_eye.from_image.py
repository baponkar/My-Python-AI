"""
This code will detect eyes in a image and it will show the
 image with rectangle around the eyes.

Note: This is just an example and you need to provide
 the path of image and haarcascade_eye.xml file.
"""

import cv2

# Load the cascade classifiers for detecting eyes
eye_cascade = cv2.CascadeClassifier("data/haarcascade/haarcascade_eye.xml")

# Load the input image
image = cv2.imread("data/image/group_of_people.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect eyes in the image
eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

# Draw a rectangle around the eyes
for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the output image
cv2.imshow("Eyes", image)
cv2.waitKey()
cv2.destroyAllWindows()

