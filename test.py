import cv2
import numpy as np

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

def get_homography(pts):
    dst = np.float32([[0,0],[W_TARGET,0],[W_TARGET,H_TARGET],[0,H_TARGET]])
    return cv2.getPerspectiveTransform(np.float32(pts), dst)

def bird_view(frame, corner_pts):
    src_points = np.float32(corner_pts)
    
    # Define the destination points (to a square, but this will be adjusted later)
    dst_points = np.float32([[0, 0], [350, 0], [350, 400], [0, 400]])

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Transform the corners of the original image to determine output size
    h, w = H_TARGET, W_TARGET
    original_corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    transformed_corners = cv2.perspectiveTransform(original_corners, matrix)

    # Calculate the bounding box of the transformed corners
    x_coords = transformed_corners[:,0,0]
    y_coords = transformed_corners[:,0,1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Compute required output size and adjust the matrix to avoid cropping
    output_width = max(w, int(np.ceil(max_x - min_x)))
    output_height = max(h, int(np.ceil(max_y - min_y)))
    adjustment_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
    adjusted_matrix = adjustment_matrix @ matrix
    return adjusted_matrix, (output_width, output_height)
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
    frame = rotate(frame)
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
    adjusted_matrix, (output_width, output_height)=bird_view(raw_disp, corner_pts)
    bird = cv2.warpPerspective(frame, adjusted_matrix, (output_width, output_height),borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    bird=cv2.resize(bird,(H_TARGET, W_TARGET), interpolation=cv2.INTER_AREA)
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
