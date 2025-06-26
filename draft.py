import cv2
import numpy as np
from helpers import *
# ───────── tunables ───────────────────────────
ANGLE_TRIANGLE = math.radians(70)
ACCEPT=25
CROP_SIZE=100
FLIPCODE=1 #0 → flip vertically (upside down)   1 → flip horizontally (left to right)   -1 → flip both vertically and horizontally (180° rotation)
CANNY_T1   = 40
CANNY_T2   = 120
CANNY_APER = 3
BLUR_KSIZE = 5
ROTATE_CW_DEG    = 270                 # 0, 90, 180, 270; clockwise
HOUGH_RHO      = 1
HOUGH_THETA    = np.pi / 180   # 1°
HOUGH_THRESH   = 140           # votes
#CLUSTER
ANGLE_BIAS = 0.3               # rad  (~17°)
RHO_BIAS   = 20                # px
W,H=480,640
# K    = np.load('K.npy')        # từ Calibrate Rect Board
# dist = np.load('dist.npy')
corner_points=[]
def on_click_roi(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(corner_points) < 2:
        if len(corner_points)==0: # Top-left inputted
            corner_points.append([x, y])
        if len(corner_points)==1: # Top-right inputted
            if x > corner_points[0][0]:
                corner_points.append([x, corner_points[0][1]])
# ───────── main loop ─────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")

print("[cam_tester] press Q to exit …")
cv2.namedWindow('inputted')
cv2.namedWindow('raw')
cv2.setMouseCallback('inputted', on_click_roi)

# Get ROI
roi_c=False
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = rotate(frame, ROTATE_CW_DEG)
    frame = cv2.flip(frame, FLIPCODE)
    frame = cv2.resize(frame, (W, H))

    raw = frame.copy()

    for p in corner_points:
        cv2.circle(raw, tuple(p), 5, (0,0,255), -1)
        cv2.putText(raw, f"{p[0]},{p[1]}", tuple(p),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    if len(corner_points) == 2:
        p1, p2 = corner_points
        distance = math.hypot(p2[0]-p1[0], p2[1]-p1[1])

        x_3 = int(round((p1[0] + p2[0]) / 2))
        height = (distance / 2) * math.tan(ANGLE_TRIANGLE)
        y_3 = int(round(p1[1] - height))   # up is negative y

        apex = [x_3, y_3]
        corner_points.append(apex)

        cv2.circle(raw, (x_3, p1[1]), 5, (0,0,255), -1)
        cv2.circle(raw, (x_3, y_3), 5, (0,0,255), -1)

    # once we have the triangle, draw its sides
    if len(corner_points) == 3:
        cv2.line(raw, corner_points[0], corner_points[1], (0,255,255), 2)
        cv2.line(raw, corner_points[0], corner_points[2], (0,255,255), 2)
        cv2.line(raw, corner_points[1], corner_points[2], (0,255,255), 2)

    cv2.imshow('raw', frame)
    cv2.imshow('inputted', raw)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), ord('Q')): break
    if k in (ord('c'), ord('C')) and len(corner_points)==3: 
        roi_c=True
        break
    if k in (ord('r'), ord('R')):
        corner_points.clear()


cap.release()
cv2.destroyAllWindows()
if roi_c:
    print(f"Left point: {corner_points[0]}\nRight point: {corner_points[1]}\nTop point: {corner_points[2]}")
else: print("QUITTING")

# Create ROI mask
white_bg = np.full((H,W), 255, dtype=np.uint8)
mask = create_binary_quad(corner_points, img_size=(H, W))
roi=apply_roi(mask)

print("Moving to lines detection")

cv2.namedWindow('cam-tester')
cap = cv2.VideoCapture(0)

while True and roi_c:
    ret, frame = cap.read()
    if not ret:
        break
    frame=rotate(frame,ROTATE_CW_DEG)
    frame = cv2.flip(frame, FLIPCODE)
    frame = cv2.resize(frame, (W, H))

    # Cắt frame phía dưới đáy tam giác để tránh nhiễu
    y_bottom = max(corner_points[0][1], corner_points[1][1])
    y_crop = max(0, y_bottom - CROP_SIZE)  # Crop lên trên 100px từ đáy
    
    mask_3ch = cv2.merge([roi]*3)

    frame = np.where(mask_3ch == 255, frame, 255).astype(np.uint8)  
    frame = frame[:y_crop, :]
    # 1. blur + gray
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. edges
    edges = cv2.Canny(gray, CANNY_T1, CANNY_T2, apertureSize=CANNY_APER)

    # 3. Hough standard (ρ,θ)
    lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH)

    # create overlay copy
    hough_vis = frame.copy()

    if lines is not None: #Checking if there are any lines
        pass
    else:
        cv2.imshow('cam-tester',hough_vis)
        continue
    flines = cluster_lines(lines,RHO_BIAS,ANGLE_BIAS)
    flines=flines.reshape(len(flines),2)

    min_ang=180 # Temporary setting the taken horizontal angle to 180 degrees
    for index in range(len(flines)):
        
        rho=flines[index][0]
        theta=flines[index][1]
        angle_x_axis=math.degrees(np.pi/2 - theta)
        if - ACCEPT <= angle_x_axis and angle_x_axis <= ACCEPT: #Only horizontal line with angle of lines smaller than 45 degree
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            
            cv2.line(hough_vis, pt1, pt2, (255, 0, 0), 2)
            if min_ang>angle_x_axis: min_ang=angle_x_axis
    
    if 90-min_ang>0:
        angle=90-min_ang
        print(f'Vertical Angle: {angle}')
        # min_ang=math.radians(min_ang)
        # Drawing the result
        # Draw the heading line and central line
        H_CROP=H-CROP_SIZE
        start = (W // 2, H_CROP - 20)
        end_c = (W // 2, int(H_CROP - H_CROP*0.4))
        radius = 100
        angle_start = 0
        angle_end = -int(angle)
        dx = int(H_CROP*0.4 * np.cos(math.radians(angle)))
        dy = int(H_CROP*0.4 * np.sin(math.radians(angle)))

        end_h = (start[0] + dx, start[1] - dy)      # minus because y down
        cv2.arrowedLine(hough_vis, start, end_h, (0, 0, 255), 2, tipLength=0.15)  # yellow

        cv2.ellipse(hough_vis, start, (radius, radius), 0, angle_start, angle_end, (0, 0, 255), 2)
        text_angle = f"{angle :.2f}"
        text_x = int(start[0] + radius * math.cos(math.radians(angle / 2)))
        text_y = int(start[1] - radius * math.sin(math.radians(angle / 2)))
        cv2.putText(hough_vis, text_angle, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


        cv2.line(hough_vis, start, end_c, (255, 255, 0), 3)  # cyan
    
    else: print('Vertical Angle: Not detected!')
    cv2.imshow("cam_tester", hough_vis)
    
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()