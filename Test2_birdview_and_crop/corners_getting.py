import cv2
import numpy as np
from transform import *
import time

W_TARGET, H_TARGET   = 640, 480         # warp size & preview size
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

get_bird = False      # becomes True after 4 clicks
H = None
space=False
# Holding cofficients and b constant for top and tangent lines.
cof=[] # Top, tangent
b=[]

print("Q quit | R reset points | C confirm (bird view)")
cap = cv2.VideoCapture('vids/3.mp4')
c1=False
c2=False
while cap.isOpened():
    time.sleep(0.1)

    if not space:
        ok, frame = cap.read()

    frame = cv2.resize(frame, (W_TARGET, H_TARGET))
    cv2.imshow('raw', frame)

    # —— choose / reset four points ——
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('r'), ord('R')):
        corner_pts.clear(); get_bird = False
        cof=[]
        b=[]
        c2=False
    if key==32:
        if not c1:
            space=True
            c1=True
            time.sleep(10)
        if c1: 
            space=False
            c1=False
    if key in (ord('q'), ord('Q')):
        break
    if key in (ord('c'), ord('C')) and len(corner_pts) == 4:
        break
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

f=open('points.txt','w+')
f.write(f"tl {str(corner_pts[0]).replace(' ','')};tr {str(corner_pts[1]).replace(' ','')};br {str(corner_pts[2]).replace(' ','')};bl {str(corner_pts[3]).replace(' ','')}")
f.close()

cap.release()
cv2.destroyAllWindows()