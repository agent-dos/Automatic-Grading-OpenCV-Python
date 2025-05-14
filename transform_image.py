
import cv2
import numpy as np


def transform_paper_image(image):
    # algorithm for sorting points clockwise
    def clockwise_sort(x):
        return (np.arctan2(x[0] - mx, x[1] - my) + 0.5 * np.pi) % (2*np.pi)

    ratio = len(image[0]) / 500.0  # used for resizing the image
    original_image = image.copy()  # make a copy of the original image

    # find contours on the smaller image because it's faster
    image = cv2.resize(image, (0, 0), fx=1/ratio, fy=1/ratio)

    # gray and filter the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # bilateral filtering removes noise and preserves edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # find the edges
    edged = cv2.Canny(gray, 250, 300)

    # find the contours (OpenCV 4.x+ returns only 2 values)
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # find the biggest contour
    biggestContour = None

    # loop over our contours
    for contour in contours:
        # approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # return the biggest 4 sided approximated contour
        if len(approx) == 4:
            biggestContour = approx
            break

    # used for the perspective transform
    points = []
    desired_points = [[0, 0], [425, 0], [425, 550],
                    [0, 550]]  # 8.5in by 11in. paper

    # convert to np.float32
    desired_points = np.float32(desired_points)

    # extract points from contour
    if biggestContour is not None:
        for i in range(0, 4):
            points.append(biggestContour[i][0])

    # find midpoint of all the contour points for sorting algorithm
    mx = sum(point[0] for point in points) / 4
    my = sum(point[1] for point in points) / 4

    # sort points
    points.sort(key=clockwise_sort, reverse=True)

    # convert points to np.float32
    points = np.float32(points)

    # resize points so we can take the persepctive transform from the
    # original image giving us the maximum resolution
    paper = []
    points *= ratio
    if biggestContour is not None:
        # create persepctive matrix
        M = cv2.getPerspectiveTransform(points, desired_points)
        # warp persepctive
        paper = cv2.warpPerspective(original_image, M, (425, 550))
    
    return image, paper, biggestContour
