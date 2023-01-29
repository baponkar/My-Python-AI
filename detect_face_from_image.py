'''
In this example, the program loads the "haarcascade_frontalface_default.
xml" file which contains a pre-trained classifier for detecting faces. 
Then the program reads an image, converts it to grayscale, and applies
 the face detector to the grayscale image. The faces are then outlined 
 by rectangles in the original image and the image is displayed.

Note: This is just a starting code and It might not work in your case if 
your haarcascade_frontalface_default.xml file is in different location, 
Also you might want to add error handling and different checks as per requirement.
'''

import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier("data/haarcascade/haarcascade_frontalface_default.xml")

# Read the input image
img = cv2.imread("data/image/group_of_people.jpg")

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
#https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
# Change height only
width = 600
height = 400
new_img = cv2.resize(img, (width, height))
cv2.imshow("img", new_img)
cv2.waitKey()
