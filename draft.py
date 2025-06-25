import cv2
import numpy as np
from helpers import *
# ───────── tunables ───────────────────────────
CANNY_T1   = 40
CANNY_T2   = 120
CANNY_APER = 3
BLUR_KSIZE = 5
ROTATE_CW_DEG    = 90                 # 0, 90, 180, 270; clockwise
HOUGH_RHO      = 1
HOUGH_THETA    = np.pi / 180   # 1°
HOUGH_THRESH   = 140           # votes

# clustering tolerances (same as your script)
ANGLE_BIAS = 0.3               # rad  (~17°)
RHO_BIAS   = 20                # px
W,H=480,640
# K    = np.load('K.npy')        # từ Calibrate Rect Board
# dist = np.load('dist.npy')

# ───────── main loop ─────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")

print("[cam_tester] press Q to exit …")
print('Left lines: RED; Right lines: BLUE')
cv2.namedWindow('cam-tester')
ret, frm0 = cap.read()
# map1, map2, roi, newK = build_maps(frm0.shape[1::-1])
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame=rotate(frame,ROTATE_CW_DEG)
    # frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    frame=cv2.resize(frame,(480,640))
    # 1. blur + gray
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur  = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    # 2. edges
    edges = cv2.Canny(gray, CANNY_T1, CANNY_T2, apertureSize=CANNY_APER)

    # 3. Hough standard (ρ,θ)
    lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH)

    # create overlay copy
    hough_vis = frame.copy()

    # draw raw lines (light gray) if any
    if lines is not None: #Checking if there are any lines
        pass
    else:
        cv2.imshow('cam-tester',frame)
        continue
    flines = cluster_lines(lines,RHO_BIAS,ANGLE_BIAS)
    flines=flines.reshape(len(flines),2)
    # Angle estimation
    angle_est,idx_left,idx_right=angle_detecting(flines)

    # draw filtered lines (green) thicker
    if not angle_est:
        cv2.imshow("cam_tester", hough_vis)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break
        continue

    for index in range(len(flines)):
        rho=flines[index][0]
        theta=flines[index][1]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        if index in list(idx_left):
            cv2.line(hough_vis, pt1, pt2, (255, 0, 0), 2)
        if index in list(idx_right):
            cv2.line(hough_vis, pt1, pt2, (0, 0, 255), 2)
        else:
            cv2.line(hough_vis, pt1, pt2, (0, 255, 0), 2)

    # print(flines)

    print(f'Angle: {math.degrees(angle_est)}')
    
    # Drawing the result
    # Draw the heading line and central line
    start = (W // 2, H - 1)
    end_c = (W // 2, int(H - H*0.4))

    dx = int(H*0.4 * np.cos(angle_est))
    dy = int(H*0.4 * np.sin(angle_est))

    end_h = (start[0] + dx, start[1] - dy)      # minus because y down
    cv2.arrowedLine(hough_vis, start, end_h, (0, 255, 255), 2, tipLength=0.15)  # yellow
    cv2.line(hough_vis, start, end_c, (255, 255, 0), 3)  # cyan
    cv2.imshow("cam_tester", hough_vis)
    
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()