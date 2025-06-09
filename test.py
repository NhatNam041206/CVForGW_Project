import cv2
import numpy as np
from helpers import *
# ───────── tunables ─────────────────────────
CANNY_T1, CANNY_T2 = 40, 120
CANNY_APER   = 3
BLUR_KSIZE   = 5
ROTATE_CW_DEG = 0
HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH = 1, np.pi/180, 140
ANGLE_BIAS, RHO_BIAS = 0.3, 20          # cluster tolerances
W_TARGET, H_TARGET   = 640, 480         # warp size & preview size
roi_pts = []      # holds ≤2 clicks  (AOI corners)
use_roi = False   # True when roi_pts complete
# ───────── mouse callback ───────────────────
corner_pts = []       # holds ≤4 [x,y]

def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(corner_pts) < 4:
        corner_pts.append([x, y])

# ───────── main loop ────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")

cv2.namedWindow('inputted')
cv2.setMouseCallback('inputted', on_click)
cv2.namedWindow('raw')
cv2.namedWindow('bird')
cv2.namedWindow('edges')
cv2.namedWindow('lines')

get_bird = False      # becomes True after 4 clicks
H = None

print("Q quit | R reset points | C confirm (bird view)")

while True:
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
        H = get_homography(corner_pts); get_bird = True
    if not get_bird:
        raw_disp = frame.copy()
        for p in corner_pts:
            cv2.circle(raw_disp, tuple(p), 5, (0,0,255), -1)
        if len(corner_pts) == 4:
            cv2.polylines(raw_disp, [np.int32(corner_pts)], True, (0,255,0), 2)
        cv2.imshow('inputted', raw_disp)
        continue

    # —— bird-eye warp once we have H ——
    adjusted_matrix, (output_width, output_height)=bird_view(corner_pts)
    bird = cv2.warpPerspective(frame, adjusted_matrix, (output_width, output_height),borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    bird=cv2.resize(bird,(H_TARGET, W_TARGET), interpolation=cv2.INTER_AREA)
    cv2.imshow('bird', bird)
    # cv2.imwrite('bird_view.jpg', bird)
    
    # —— blur → edges → Hough —— 
    gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
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
