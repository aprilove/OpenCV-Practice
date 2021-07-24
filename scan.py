from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse as ap
import cv2 as cv
import imutils

ag = ap.ArgumentParser()
ag.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ag.parse_args())

image = cv.imread(args["image"])
ratio = image.shape[0] / 500
orig = image.copy()
image = imutils.resize(image, height = 500)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
edges = cv.Canny(gray, 75, 200)

# cv.imshow("Original Image", image)
# cv.imshow("Image Edges", edges)
# cv.waitKey(0)
# cv.destroyAllWindows()

cnts = cv.findContours(edges.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

for c in cnts:
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * peri, True)

	if len(approx) == 4:
		screenCnt = approx
		break

# cv.drawContours(image, [screenCnt], -1, (0,255,0), 2)
# cv.imshow("Outlines", image)
# cv.waitKey(0)
# cv.destroyAllWindows()

warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)

# T = threshold_local(warped, 11, method='gaussian',offset=10)
# warped = (warped > T).astype("uint8") * 255

cv.imshow("Original Image", imutils.resize(orig, height=650))
cv.imshow("Scanned", imutils.resize(warped, height=650))
cv.waitKey(0)
cv.destroyAllWindows()