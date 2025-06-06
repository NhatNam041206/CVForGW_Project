import cv2
import os
import numpy as np

"""
calib_capture.py
––––––––––––––––––––––––––––––––
A tiny utility to record images for **camera calibration**.

• Live preview from your webcam.
• Press **Space** or **C** to capture a frame.
• Frame is optionally validated: saves only if a chessboard corner grid
  of size BOARD_SIZE is detected (recommended for OpenCV calibration).
• Files are named  calib_000.jpg, calib_001.jpg … in SAVE_DIR.
• Press **Q** to quit.

Tune BOARD_SIZE to match the **inner** grid corners of your printed
chessboard, and adjust the resolution section if your camera supports
it.
"""

SAVE_DIR   = "calib"
BOARD_SIZE = (9, 6)   # (columns, rows) inner corners
CHECK_GRID = True     # True = save only if grid found

# create folder if needed
os.makedirs(SAVE_DIR, exist_ok=True)

# start camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open camera")

# (optional) set a fixed resolution; comment out if not supported
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[calib_capture]  Q = quit  |  C / Space = capture")

frame_id = len([f for f in os.listdir(SAVE_DIR) if f.startswith("calib_")])
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # draw existing count
    cv2.putText(display, f"saved: {frame_id}", (10, 30), font, 1,
                (50, 255, 50), 2, cv2.LINE_AA)

    cv2.imshow("calib capture", display)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), ord('Q')):
        break

    if key in (ord('c'), ord('C'), 32):  # space or c
        save_frame = True
        if CHECK_GRID:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE,
                                                       cv2.CALIB_CB_ADAPTIVE_THRESH |
                                                       cv2.CALIB_CB_FAST_CHECK |
                                                       cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found:
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, found)
                cv2.imshow("calib capture", display)
                cv2.waitKey(500)  # brief flash to confirm
            else:
                print("\t[!] grid NOT found – frame skipped")
                save_frame = False

        if save_frame:
            fname = os.path.join(SAVE_DIR, f"calib_{frame_id:03d}.jpg")
            cv2.imwrite(fname, frame)
            print(f"\tsaved {fname}")
            frame_id += 1

cap.release()
cv2.destroyAllWindows()
