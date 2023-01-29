"""
This program captures video from the default camera (camera index 0) using the cv2.VideoCapture class. It then applies a background subtractor (cv2.createBackgroundSubtractorMOG2) to the frames to create a foreground mask. The program counts the non-zero pixels in the foreground mask, and if the count is above a certain threshold, it prints "Motion detected!". The frames and foreground mask are displayed in separate windows. The program exits when the 'q' key is pressed.

You can adjust the threshold and other parameters to fine-tune the motion detection.

Note: You will need OpenCV library installed in order to run this program.

"""



import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Create a background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Apply the background subtractor
    fgmask = fgbg.apply(frame)

    # Count non-zero pixels in the foreground mask
    count = cv2.countNonZero(fgmask)

 
    if count > 1000:
        print("Motion detected!")

    # Display the frame
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
