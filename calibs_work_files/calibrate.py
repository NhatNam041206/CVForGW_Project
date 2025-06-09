import cv2
import glob
import numpy as np

"""
calibrate_rect_board.py
––––––––––––––––––––––––––––––––––––––
Read all images in a folder (default *calib/*) that were captured
with **Calib Capture** and compute intrinsic matrix + distortion for a
**rectangular chessboard** (standard black/white squares, not circles).

Usage:
    python calibrate_rect_board.py  [calib_folder]

• BOARD_SIZE   = (columns, rows) **inner** corners.
• SQUARE_SIZE  = physical size of a square (mm).  Set to 1.0 if you only
  need pixel‑accurate undistortion (scale‑less).
• Outputs:   K.npy   dist.npy   calib_report.txt
"""

import os
import sys

# ───────── configurable ─────────
BOARD_SIZE   = (9, 6)      # inner corners 9×6 ⇢ 10×7 squares
SQUARE_SIZE  = 8.0        # mm (change if your square is different)
CALIB_DIR    = sys.argv[1] if len(sys.argv) > 1 else "calib"

# collect image paths
images = sorted(glob.glob(os.path.join(CALIB_DIR, "calib_*.jpg")))
if not images:
    raise FileNotFoundError(f"No images like 'calib_###.jpg' in {CALIB_DIR}")
print(f"[calibrate] found {len(images)} images in '{CALIB_DIR}'")

# prepare 3‑D object points (Z=0 plane)
objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_pts = []   # 3‑D points in real world
img_pts = []   # 2‑D points in image plane

for fn in images:
    img  = cv2.imread(fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH |
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        print(f"  [skip] {fn}: grid not found")
        continue
    # refine corner locations
    cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                     (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    obj_pts.append(objp)
    img_pts.append(corners)
    print(f"  [ok]   {fn}")

if len(obj_pts) < 10:
    raise RuntimeError("Need at least 10 good images for reliable calibration.")

h, w = gray.shape[:2]
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)

print("\n=== Calibration result ===")
print(f"RMS reprojection error : {ret:.4f} px")
print("Camera matrix K:\n", K)
print("Distortion coeffs:\n", dist.ravel())

# save to disk
np.save("K.npy", K)
np.save("dist.npy", dist)
with open("calib_report.txt", "w") as f:
    f.write(f"RMS error: {ret}\n\nK:\n{K}\n\nDistortion:\n{dist}\n")

print("Files saved: K.npy  dist.npy  calib_report.txt")
