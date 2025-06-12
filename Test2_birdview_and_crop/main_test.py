import cv2
import numpy as np
from transform import *
import time
# ───────── tunables ─────────────────────────
ROTATE_CW_DEG = 0
CANNY_T1, CANNY_T2 = 10, 200
CANNY_APER   = 3
BLUR_KSIZE   = 5
HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH = 1, np.pi/180, 140
ANGLE_BIAS, RHO_BIAS = 0.3, 20          # cluster tolerances
W_TARGET, H_TARGET   = 640, 480         # warp size & preview size
roi_pts = []      # holds ≤2 clicks  (AOI corners)
use_roi = False   # True when roi_pts complete
# ───────── helpers ─────────────────────────
def rotate(img):
    d = ROTATE_CW_DEG % 360
    if d == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def cluster_lines(lines):
    if lines is None: return []
    segs = [tuple(l[0]) for l in lines]
    segs.sort(key=lambda p: p[0])                   # sort by ρ
    visited, out = set(), []
    for i, (rho_i, th_i) in enumerate(segs):
        if i in visited: continue
        rs, ts = [rho_i], [th_i]; visited.add(i)
        j = i + 1
        while j < len(segs) and abs(segs[j][0]-rho_i) < RHO_BIAS:
            rho_j, th_j = segs[j]
            if abs(th_j - th_i) < ANGLE_BIAS:
                visited.add(j); rs.append(rho_j); ts.append(th_j)
            j += 1
        out.append((float(np.mean(rs)), float(np.mean(ts))))
    return np.array(out, np.float32).reshape(-1,1,2)


# ───────── mouse callback ───────────────────
corner_pts = []       # holds ≤4 [x,y]
def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(corner_pts) < 3:
        if len(corner_pts)==0: # Top-left inputted
            corner_pts.append([x, y])
        if len(corner_pts)==1: # Top-right inputted
            if x > corner_pts[0][0]:
                corner_pts.append([x, corner_pts[0][1]])

            else: print("Top-left point must be to the left of the top-right point.")

        if len(corner_pts)==2: # Bottom left inputted
            if (x < corner_pts[0][0]) and (y> corner_pts[0][1] or y > corner_pts[1][1]):
                corner_pts.append([x, y])
            else: print("Bottom point must be below the top points and having x greater or smaller than top points.")            

# ───────── main loop ────────────────────────
cap = cv2.VideoCapture(1)
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

# Holding cofficients and b constant for top and tangent lines.
cof=[] # Top, tangent
b=[]

print("Q quit | R reset points | C confirm (bird view)")
cap = cv2.VideoCapture('vids/3.mp4')
c1=False
c2=False
while cap.isOpened():
    time.sleep(0.1)
    ok, frame = cap.read()

    frame = cv2.resize(frame, (W_TARGET, H_TARGET))
    cv2.imshow('raw', frame)

    # —— choose / reset four points ——
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('r'), ord('R')):
        corner_pts.clear(); get_bird = False
        cof=[]
        b=[]
        c1=False
        c2=False
    if key in (ord('q'), ord('Q')):
        break
    if key in (ord('c'), ord('C')) and len(corner_pts) == 4:
        get_bird = True
    if not get_bird:
        raw_disp = frame.copy()
        for p in corner_pts:
            cv2.circle(raw_disp, tuple(p), 5, (0,0,255), -1)
            cv2.putText(raw_disp, f"{p[0]},{p[1]}", tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            if len(corner_pts) == 1: 
                print('Top left point inputted.')
                # Draw a horizontal line from the left point to the right side of the frame.
                cv2.line(raw_disp, tuple(p), (W_TARGET-1, p[1]), (0,255,255), 2)
            
            if len(corner_pts) == 2:
                print('Top right point inputted.')
                '''
                if not c1:
                    print('Line calculated!')
                    # Calculate the slope and b constant for the top line.
                    slope = (corner_pts[1][1] - corner_pts[0][1]) / (corner_pts[1][0] - corner_pts[0][0])
                    b_top = corner_pts[0][1] - slope * corner_pts[0][0]

                    # Create cofficients and b_constants matrices
                    cof.append([slope,-1])
                    b.append([-b_top])
                    c1=True'''
            if len(corner_pts) == 3:
                print('Bottom left inputted.')

                if not c2:
                    slope = (corner_pts[2][1] - corner_pts[0][1]) / (corner_pts[2][0] - corner_pts[0][0])
                    b_tangent = corner_pts[1][1] + slope * corner_pts[1][0]

                    cof.append([-slope,-1])
                    b.append([-b_tangent])

                    slope = (corner_pts[1][1] - corner_pts[0][1]) / (corner_pts[1][0] - corner_pts[0][0])
                    b_tangent = corner_pts[2][1] + slope * corner_pts[2][0]

                    cof.append([-slope,-1])
                    b.append([-b_tangent])

                    print(f'Lines cofficients: {cof}')
                    print(f'Lines b constants: {b}')
                    cof,b= np.array(cof), np.array(b)
                    temp=corner_pts[-1]
                    corner_pts[-1]=np.array(np.linalg.solve(cof,b),dtype=int).reshape(-1).tolist() # Solve the linear equations to get the intersection point (Last bottom point).
                    corner_pts.append(temp)
                    print(f'Inputted points: {corner_pts}')
                    print('Inputted sucessfully!')
                    c2=True
            if len(corner_pts) == 4:
                cv2.polylines(raw_disp, [np.int32(corner_pts)], True, (0,255,0), 4)

        cv2.imshow('inputted', raw_disp)
        continue
    # —— bird-eye warp once we have H ——
    bird=four_point_transform(frame, np.array(corner_pts, dtype = "float32"))
    
    rect=corner_pts
    
    print(f"Top left:{rect[0]} Top right:{rect[1]} Bottom right:{rect[2]} Bottom left:{rect[3]}")

    bird=rotate(bird)
    cv2.imshow('bird', bird)
    # cv2.imwrite('bird_view.jpg', bird)
    # —— blur → edges → Hough —— 
    gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    edges = cv2.Canny(gray, CANNY_T1, CANNY_T2, apertureSize=CANNY_APER)
    raw_lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH)
    flines   = cluster_lines(raw_lines)

    vis = bird.copy()
    if raw_lines is None:
        # print("[cam-tester] no lines found")
        cv2.imshow('cam-tester', vis)
        continue

    else: 
        # Uncomment to see raw lines
        for r, t in raw_lines[:,0]:
            a,b = np.cos(t), np.sin(t); x0,y0 = a*r, b*r
            p1 = (int(x0+1000*(-b)), int(y0+1000*a))
            p2 = (int(x0-1000*(-b)), int(y0-1000*a))
            cv2.line(vis, p1, p2, (180, 180, 180), 1)
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