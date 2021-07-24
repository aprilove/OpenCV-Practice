from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse as ap
import imutils as im
import cv2 as cv
from scipy.spatial.distance import correlation

ag = ap.ArgumentParser()
ag.add_argument("-i", "--image", required=True,
                help="path to the image")
args = vars(ag.parse_args())

# Dictionary
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

image = cv.imread(args["image"])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
edge = cv.Canny(blur, 75, 200)

cnts = cv.findContours(edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = im.grab_contours(cnts)
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            docCnt = approx
            break
        
paper = four_point_transform(image, docCnt.reshape(4,2))
warp = four_point_transform(gray, docCnt.reshape(4,2))

thresh = cv.threshold(warp, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = im.grab_contours(cnts)
questionCnts = []

for c in cnts:
    (x, y, w, h) = cv.boundingRect(c)
    ar = w / float(h)
    
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
    
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None
    
    for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv.drawContours(mask, [c], -1, 255, -1)
            
            mask = cv.bitwise_and(thresh, thresh, mask=mask)
            total = cv.countNonZero(mask)
            
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
                
    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    
    if k == bubbled[1]:
        color = (0,255,0)
        correct += 1
    
    cv.drawContours(paper, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv.imshow("Original", image)
cv.imshow("Exam", paper)
cv.waitKey(0)