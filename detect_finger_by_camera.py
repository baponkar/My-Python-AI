'''
This code will open the default camera and detect fingers in real-time video feed. It will show the live video feed with green color contour around the finger.

You can adjust the color range of skin color in HSV by changing the value of lower_range and upper_range variable.
Also you can change the camera by changing the index passed in cv2.VideoCapture(0) to the desired camera

Note: This is just an example and you may need to adjust the parameters and threshold values to improve the finger detection depending on the lighting conditions and skin color of the user.
'''

import cv2
import numpy as np

# Define the region of interest (ROI) where the fingers will be detected
x1, y1, x2, y2 = (0, 0, 0, 0)

# Define the color range of skin color in HSV
lower_range = np.array([0,20,70], dtype=np.uint8)
upper_range = np.array([20,255,255], dtype=np.uint8)

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get only skin color
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Apply morphological transformations to remove noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the contours in the thresholded frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Draw the largest contour on the frame
    if largest_contour is not None:
        cv2.drawContours(frame, [largest_contour], -1, (0,255,0), 3)

    # Display the output frame
    cv2.imshow("Fingers", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
