# HEATMAP APP

import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression

# Parameters : image size
# Smaller image means faster processing
# Larger image may mean better detection ???
height = 480
width = 640

# Initialize the detector
# This is the HOG detector, but any other TF based detector can be used
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

heatMap = np.zeros((height,width,3), np.uint8)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    # Get image data in gray-scale
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # optional resize
    image = imutils.resize(image, width=width, height=height)

    # detect people in the image (Consider TF based more advanced detection)
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # optional non-max suppression
    # eliminate redundant/overlapping rectangles for the same object
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    sumMap = np.zeros((height,width,3), np.uint8)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        rectMap = np.zeros((height,width,3), np.uint8)
        rectMap[yA:yB,xA:xB,2] = 255
        sumMap = cv2.add(sumMap, rectMap)

    heatMap = cv2.addWeighted(heatMap, 0.99, sumMap, 0.01, 0)

    # show the output image
    cv2.imshow("Detect", image)
    cv2.imshow("HeatMap", heatMap)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()