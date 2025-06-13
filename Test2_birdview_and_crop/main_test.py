
import cv2, numpy as np, time
from transform import four_point_transform   # <- your own helper

# ─── config ───
VIDEO     = 'vids/3.mp4'      # or 0 / 1 for a webcam
SIZE      = (640, 480)
CANNY_T1, CANNY_T2 = 10, 200
HOUGH_RHO, HOUGH_THETA, HOUGH_T = 1, np.pi/180, 140
ANGLE_BIAS, RHO_BIAS = 0.3, 20

# read warp points
with open('points.txt') as f:
    pts = [p.split(' ')[1][1:-1] for p in f.readline().split(';')]
corner_pts = np.array([[int(x), int(y)] for x, y in (s.split(',') for s in pts)],
                      dtype=np.float32)

def rotate(img, deg):
    d = deg % 360
    if d == 90:   return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:  return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270:  return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def cluster(lines):
    if lines is None:                       # nothing found
        return np.empty((0,1,2), np.float32)
    segs = [tuple(l[0]) for l in lines]
    segs.sort(key=lambda p: p[0])          # by rho
    visited, out = set(), []
    for i, (rho_i, th_i) in enumerate(segs):
        if i in visited: continue
        rs, ts = [rho_i], [th_i]; visited.add(i)
        for j in range(i+1, len(segs)):
            rho_j, th_j = segs[j]
            if abs(rho_j-rho_i) < RHO_BIAS and abs(th_j-th_i) < ANGLE_BIAS:
                visited.add(j);  rs.append(rho_j);  ts.append(th_j)
        out.append((np.mean(rs), np.mean(ts)))
    return np.asarray(out, np.float32).reshape(-1,1,2)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise IOError(f"Cannot open {VIDEO}")

print("Press q to quit")
while True:
    time.sleep(0.1)
    ok, frame = cap.read()
    if not ok:
        print("End of stream")
        break

    frame = cv2.resize(frame, SIZE)
    cv2.imshow('raw', frame)

    bird = four_point_transform(frame, corner_pts)
    bird = rotate(bird, 0)
    cv2.imshow('bird', bird)

    edges = cv2.Canny(cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY),
                      CANNY_T1, CANNY_T2, apertureSize=3)
    raw_lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_T)
    flines = cluster(raw_lines)

    vis = bird.copy()
    if raw_lines is not None:
        for r,t in raw_lines[:,0]:
            a,b = np.cos(t), np.sin(t); x0,y0 = a*r, b*r
            p1 = (int(x0+1000*(-b)), int(y0+1000*a))
            p2 = (int(x0-1000*(-b)), int(y0-1000*a))
            cv2.line(vis, p1, p2, (180,180,180), 1)

    if flines.size:
        for r,t in flines[:,0]:
            a,b = np.cos(t), np.sin(t); x0,y0 = a*r, b*r
            p1 = (int(x0+1000*(-b)), int(y0+1000*a))
            p2 = (int(x0-1000*(-b)), int(y0-1000*a))
            cv2.line(vis, p1, p2, (0,255,0), 2)

    cv2.imshow('edges', cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    cv2.imshow('lines', vis)

    # *** event loop – crucial ***
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
