from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse as ap
import imutils as im
import cv2

ag = ap.ArgumentParser()
ag.add_argument("-i", "--image", required=True, help="Image Path")
ag.add_argument("-w", "--width", type=float, required=True,
                help="Width of left obj in img")
args = vars(ag.parse_args())


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

edge = cv2.Canny(gray, 50, 100)
edge = cv2.dilate(edge, None, iterations=1)
edge = cv2.erode(edge, None, iterations=1)

cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = im.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

for c in cnts:
    if cv2.contourArea(c) < 100:
        continue

    og = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if im.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)
    cv2.drawContours(og, [box.astype("int")], -1, (0, 255, 0), 2)

    for x, y in box:
        cv2.circle(og, (int(x), int(y)), 5, (0, 255, 0), -1)

    (tl, tr, br, bl) = box
    (tmX, tmY) = midpoint(tl, tr)
    (bmX, bmY) = midpoint(bl, br)
    (lmX, lmY) = midpoint(tl, bl)
    (rmX, rmY) = midpoint(tr, br)

    cv2.circle(og, (int(tmX), int(tmY)), 5, (255, 0, 0), -1)
    cv2.circle(og, (int(bmX), int(bmY)), 5, (255, 0, 0), -1)
    cv2.circle(og, (int(lmX), int(lmY)), 5, (255, 0, 0), -1)
    cv2.circle(og, (int(rmX), int(rmY)), 5, (255, 0, 0), -1)

    cv2.line(og, (int(tmX), int(tmY)), (int(bmX), int(bmY)), (255, 0, 255), 2)
    cv2.line(og, (int(rmX), int(rmY)), (int(lmX), int(lmY)), (255, 0, 255), 2)

    dv = dist.euclidean((tmX, tmY), (bmX, bmY))
    dh = dist.euclidean((rmX, rmY), (lmX, lmY))

    if pixelsPerMetric is None:
        pixelsPerMetric = dh / args["width"]

    dimv = 2.54 * dv / pixelsPerMetric
    dimh = 2.54 * dh / pixelsPerMetric

    cv2.putText(og, "{:.1f}cm".format(dimv),
                (int(tmX - 15), int(tmY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)
    
    cv2.putText(og, "{:.1f}cm".format(dimh),
                (int(rmX + 10), int(rmY)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)
    
    cv2.imshow("image", og)
    cv2.waitKey(0)

    """
    Explanation of how it works: 
    1. Find contours of the shape
    2. Draw a rectangle around the shape
    3. Find the distances between the adjacent midpoints of the edges
    
    """