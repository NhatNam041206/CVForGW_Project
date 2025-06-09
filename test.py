import cv2
import numpy as np
import sys # For debug output
from helpers import *
#
#
"""
This file contains codes for testing new methods including ROI, bird view, line clustering, angle estimating and hough transform (lines detection).
Folowing steps:
1. Capture frame from camera
2. Rotate frame if needed
3. Resize frame to target size
4. ROI selection by clicking four corners
5. Bird-eye view transformation based on selected ROI
6. Canny edge detection
7. Hough transform to detect lines
8. Cluster detected lines, estimate angle and distance
9. Display results in multiple windows
"""
#
#

# ───────── tunables ─────────────────────────
CANNY_T1, CANNY_T2 = 40, 120
CANNY_APER   = 3
BLUR_KSIZE   = 5
ROTATE_CW_DEG = 0
HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH = 1, np.pi/180, 140
ANGLE_BIAS, RHO_BIAS = 0.3, 20          # cluster tolerances
W_TARGET, H_TARGET   = 640, 480         # warp size & preview size
nextRunning = True
# ───────── mouse callback ───────────────────
corner_pts = []       # holds ≤4 [x,y]
roi_pts = []      # holds ≤4 clicks  (AOI corners)

def on_click_roi(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_pts) < 4:
        corner_pts.append([x, y])

def on_click_bird_view(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(corner_pts) < 4:
        corner_pts.append([x, y])

# ───────── main loop ────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")


get_bird = False      # becomes True after 4 clicks
H = None

# ───────── ROI selecting ───────────────────
get_roi=False
print("Q quit | R reset points | C confirm (ROI)")

cv2.namedWindow('raw')
cv2.namedWindow('inputted_ROI')
cv2.setMouseCallback('inputted_ROI', on_click_roi)
cv2.namedWindow('ROI')

while not get_roi:
    ok, frame = cap.read()
    if not ok: 
        nextRunning=False
        break
    frame = rotate(frame,ROTATE_CW_DEG)
    frame = cv2.resize(frame, (W_TARGET, H_TARGET))
    cv2.imshow('raw', frame)
    key=cv2.waitKey(1) & 0xFF

    # —— choose / reset four points ——
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('r'), ord('R')):
        roi_pts.clear(); get_bird = False
    if key in (ord('q'), ord('Q')):
        break
    if key in (ord('c'), ord('C')) and len(corner_pts) == 4:
        get_roi = True
    if not get_roi:
        raw_disp = frame.copy()
        for p in roi_pts:
            cv2.circle(raw_disp, tuple(p), 5, (255,0,0), -1)
        if len(roi_pts) == 4:
            cv2.polylines(raw_disp, [np.int32(corner_pts)], True, (0,255,0), 2)
        cv2.imshow('inputted', raw_disp)


sys.exit() # Debug exit point, remove later

print("Q quit | R reset points | C confirm (bird view)")

cv2.namedWindow('inputted_bird_view')
cv2.setMouseCallback('inputted_bird_view', on_click_bird_view)
cv2.namedWindow('bird')
cv2.namedWindow('edges')
cv2.namedWindow('lines')
while nextRunning:
    ok, frame = cap.read()
    if not ok: break
    frame = rotate(frame,ROTATE_CW_DEG)
    frame = cv2.resize(frame, (W_TARGET, H_TARGET))
    cv2.imshow('raw', frame)

    # —— choose / reset four points ——
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('r'), ord('R')):
        corner_pts.clear(); get_bird = False
    if key in (ord('q'), ord('Q')):
        break
    if key in (ord('c'), ord('C')) and len(corner_pts) == 4:
        get_bird = True
    if not get_bird:
        raw_disp = frame.copy()
        for p in corner_pts:
            cv2.circle(raw_disp, tuple(p), 5, (0,0,255), -1)
        if len(corner_pts) == 4:
            cv2.polylines(raw_disp, [np.int32(corner_pts)], True, (0,255,0), 2)
        cv2.imshow('inputted_bird_view', raw_disp)
        continue

    # —— bird-eye warp once we have H ——
    adjusted_matrix, (output_width, output_height)=bird_view(corner_pts)
    bird = cv2.warpPerspective(frame, adjusted_matrix, (output_width, output_height),borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    # bird=cv2.resize(bird,(H_TARGET, W_TARGET), interpolation=cv2.INTER_AREA) #New comment, tempt
    cv2.imshow('bird', bird)
    # cv2.imwrite('bird_view.jpg', bird) # Uncomment to save bird view
    
    # —— blur → edges → Hough —— 
    gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0) # blur is not in used 
    edges = cv2.Canny(gray, CANNY_T1, CANNY_T2, apertureSize=CANNY_APER)
    raw_lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH)
    flines   = cluster_lines(raw_lines,RHO_BIAS,ANGLE_BIAS)

    vis = bird.copy()
    if raw_lines is None:
        print("[cam-tester] no lines found")
        cv2.imshow('cam-tester', vis)
        continue

    else: 
        # Uncomment to see raw lines
        # for r, t in raw_lines[:,0]:
        #     a,b = np.cos(t), np.sin(t); x0,y0 = a*r, b*r
        #     p1 = (int(x0+1000*(-b)), int(y0+1000*a))
        #     p2 = (int(x0-1000*(-b)), int(y0-1000*a))
        #     cv2.line(vis, p1, p2, (180, 180, 180), 1)
        pass
    for r, t in flines[:,0]:
        a,b = np.cos(t), np.sin(t); x0,y0 = a*r, b*r
        p1 = (int(x0+1000*(-b)), int(y0+1000*a))
        p2 = (int(x0-1000*(-b)), int(y0-1000*a))
        cv2.line(vis, p1, p2, (0,255,0), 2)

    # blur_v  = cv2.cvtColor(blur,  cv2.COLOR_GRAY2BGR)
    edge_v  = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('edges', edge_v)
    cv2.imshow('lines', vis)

cap.release()
cv2.destroyAllWindows()
