"""
License Plate Recognition using OpenCV and Tesseract OCR
"""

import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

image = cv2.imread("car.jpeg")

resized_image = imutils.resize(image, width=500)

cv2.imshow("Original Image", resized_image)

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# To smoothen the image, Apply bilateral filter 
filtered_gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

edged_image = cv2.Canny(filtered_gray_image, 170, 200)

(contours, _) = cv2.findContours(
    edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
)

# Sort contours by area, keeping the largest 30
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

license_plate_contour = None

# Loop over contours to find the one that resembles a license plate (with 4 sides)
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx_corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # If the approximated contour has 4 corners, we assume it is the license plate
    if len(approx_corners) == 4:
        license_plate_contour = approx_corners
        break

# Create a mask with the same dimensions as the grayscale image
mask = np.zeros(gray_image.shape, np.uint8)

masked_license_plate = cv2.drawContours(mask, [license_plate_contour], 0, 255, -1)

isolated_license_plate = cv2.bitwise_and(resized_image, resized_image, mask=mask)

cv2.namedWindow("Final Image", cv2.WINDOW_NORMAL)
cv2.imshow("Final Image", isolated_license_plate)

# Configuration for Tesseract OCR
tesseract_config = "-l eng --oem 1 --psm 3"

license_plate_text = pytesseract.image_to_string(
    isolated_license_plate, config=tesseract_config
)

data = {
    "date": [time.asctime(time.localtime(time.time()))],
    "license_plate_number": [license_plate_text],
}

data_frame = pd.DataFrame(data, columns=["date", "license_plate_number"])
data_frame.to_csv("license_plate_data.csv")

# Print the recognized text
print(license_plate_text)

cv2.waitKey(0)
