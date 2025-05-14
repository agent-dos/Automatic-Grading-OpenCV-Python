import numpy as np
import cv2
from grade_paper import ProcessPage
from transform_image import transform_paper_image

cv2.namedWindow('Original Image')
cv2.namedWindow('Scanned Paper')

# ret, image = cap.read()
image = cv2.imread("images\enhanced_answer_sheet (1).png")

image, paper, biggestContour = transform_paper_image(image)

answers = 1
answers, paper, codes = ProcessPage(paper)

cv2.imshow("Scanned Paper", paper)
cv2.imwrite("images/scanned_output.jpg", paper)

# draw the contour
if biggestContour is not None:
    if answers != -1:
        cv2.drawContours(image, [biggestContour], -1, (0, 255, 0), 3)
        print(answers)
        if codes is not None:
            print(codes)
    else:
        cv2.drawContours(image, [biggestContour], -1, (0, 0, 255), 3)

cv2.imshow("Original Image", cv2.resize(image, (0, 0), fx=0.7, fy=0.7))

cv2.waitKey(0)
