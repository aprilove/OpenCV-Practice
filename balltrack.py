from collections import deque
from imutils.video import VideoStream, videostream
import numpy as np
import argparse as ap
import cv2 as cv
import imutils as im
import time

ag = ap.ArgumentParser()
ag.add_argument("-v", "--video", help="path to the (optional) video file")
ag.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ag.parse_args())

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv.VideoCapture(args["video"])

time.sleep(2.0)

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    
    if frame is None:
        break
    
    frame = im.resize(frame, width=600)
    blur = cv.GaussianBlur(frame, (11,11), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    
    mask = cv.inRange(hsv, greenLower, greenUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = im.grab_contours(cnts)
    center = None
    
    if len(cnts) > 0:
        c = max(cnts, key=cv.contourArea)
        ((x,y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if radius > 10:
            cv.circle(frame, (int(x), int(y)), int(radius), (0,255,255),2)
            cv.circle(frame, center, 5, (0,0,255), -1)
    
    pts.appendleft(center)
    
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
if not args.get("video", False):
    vs.stop()

else:
    vs.release()

cv.destroyAllWindows()