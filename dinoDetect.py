import numpy as np
import cv2
import math
import pyautogui
import sys

def blur_image(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def morphological_close(image):
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    return erosion

def get_angle(start, end, far):
    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
    return angle

# Open Camera
capture = cv2.VideoCapture(0)

while capture.isOpened():
    # Capture frames from the camera
    ret, frame = capture.read()

    #Create a rectangular box to detect hand
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)

    #Get image inside rectangle
    image_frame = frame[100:300, 100:300]

    #Convert image to HSV color Space after Gaussian Blur with a 3x3 Kernel
    image = cv2.cvtColor(blur_image(image_frame), cv2.COLOR_BGR2HSV)

    #Create a binary Image of Hand
    binary_image = cv2.inRange(image, np.array([2, 0, 0]), np.array([20, 255, 255]))
    
    #Filter background noise with a closing morphological transformation
    denoised_image = morphological_close(binary_image)

    # Apply Gaussian Blur and Threshold
    filtered = blur_image(denoised_image)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    #Find contours
    contours, hierachy = cv2.findContours(denoised_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        #Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 0, 255), 0)

        #Find convex hull
        hull = cv2.convexHull(contour)

        #Draw contour
        drawing = np.zeros(image_frame.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        #Fill convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        #Find fingertip points using cosine rule
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            angle = get_angle(start, end, far)

            # if angle >= 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(image_frame, far, 1, [0, 0, 255], -1)

            cv2.line(image_frame, start, end, [0, 255, 0], 2)
        
        #If hand is present, then press SPACE
        if count_defects >= 4:
                pyautogui.press('space')
                cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    except Exception as e: 
        print("An exception has occurred")
        print(e)
        sys.exit(1)

    # Show required images
    cv2.imshow("Gesture", frame)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()