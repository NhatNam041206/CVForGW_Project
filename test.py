# test.py – clean rewrite (v3) with proper crop sizing
# -------------------------------------------------------------
# Phases:
#   1. ROI   – click 4 pts (pink)  → binary mask
#   2. Warp  – click 4 pts (red)   → homography H
#   3. Run   – warp frame & mask, crop exact area, analyse lines
# -------------------------------------------------------------

import cv2
import numpy as np
from helpers import rotate, cluster_lines, create_binary_quad

# ───────── configuration ─────────────────────────────────────
ROTATE_CW_DEG      = 90
W_TARGET, H_TARGET = 640, 480   # base canvas size after resize & warp

CANNY_T1, CANNY_T2 = 40, 120
CANNY_APER         = 3
HOUGH_RHO          = 1
HOUGH_THETA        = np.pi / 180
HOUGH_THRESH       = 140
ANGLE_BIAS, RHO_BIAS = 0.3, 20

# ───────── internal state ────────────────────────────────────
phase        = "roi"  # roi → warp → run
roi_pts:  list[list[int]] = []
warp_pts: list[list[int]] = []
roi_mask: np.ndarray | None = None
H:        np.ndarray | None = None

# ───────── helper functions ──────────────────────────────────

def order_pts(pts: list[list[int]]) -> np.ndarray:
    """Return points in TL, TR, BR, BL order."""
    pts = np.asarray(pts, dtype=np.float32)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1)[:, 0]
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def build_H(src_pts: list[list[int]]) -> np.ndarray:
    src = order_pts(src_pts)
    dst = np.float32([[0, 0], [W_TARGET, 0],
                      [W_TARGET, H_TARGET], [0, H_TARGET]])
    return cv2.getPerspectiveTransform(src, dst)


def crop_by_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop image to the bounding box of non-zero mask region."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img  # nothing – return as is
    x, y, w, h = cv2.boundingRect(coords)
    return img[y : y + h, x : x + w]

# ───────── mouse callbacks ──────────────────────────────────

def cb_roi(event, x, y, *_):
    global phase, roi_pts, roi_mask
    if phase != "roi" or event != cv2.EVENT_LBUTTONDOWN:
        return
    roi_pts.append([x, y])
    if len(roi_pts) == 4:
        roi_mask = create_binary_quad(roi_pts, img_size=(H_TARGET, W_TARGET))
        roi_mask = (roi_mask > 0).astype(np.uint8) * 255
        phase = "warp"
        cv2.setMouseCallback("raw", cb_warp)


def cb_warp(event, x, y, *_):
    global phase, warp_pts, H
    if phase != "warp" or event != cv2.EVENT_LBUTTONDOWN:
        return
    warp_pts.append([x, y])
    if len(warp_pts) == 4:
        H = build_H(warp_pts)
        phase = "run"
        cv2.setMouseCallback("raw", lambda *args: None)

# ───────── main loop ────────────────────────────────────────

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    cv2.namedWindow("raw")
    cv2.setMouseCallback("raw", cb_roi)
    cv2.namedWindow("bird")
    cv2.namedWindow("crop")
    cv2.namedWindow("edges")
    cv2.namedWindow("lines")

    print("Instructions:\n"
          "  ROI  : click 4 pts (pink) → R reset → Q quit\n"
          "  Warp : click 4 pts (red)  → R reset → Q quit\n")

    global phase, roi_pts, warp_pts, roi_mask, H
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = rotate(frame, ROTATE_CW_DEG)
        frame = cv2.resize(frame, (W_TARGET, H_TARGET))
        raw   = frame.copy()

        # keyboard
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            if phase == "roi":
                roi_pts.clear(); roi_mask = None
            elif phase == "warp":
                warp_pts.clear(); H = None
            elif phase == "run":
                phase = "warp"; warp_pts.clear(); H = None
                cv2.setMouseCallback("raw", cb_warp)
            continue

        # ---------- Phase visuals ----------
        if phase == "roi":
            vis = raw.copy()
            for p in roi_pts:
                cv2.circle(vis, tuple(p), 4, (255, 0, 255), -1)
            if len(roi_pts) == 4:
                cv2.polylines(vis, [np.int32(roi_pts)], True, (255, 255, 0), 2)
            cv2.imshow("raw", vis)
            continue

        if phase == "warp":
            vis = cv2.bitwise_and(raw, raw, mask=roi_mask)
            for p in warp_pts:
                cv2.circle(vis, tuple(p), 4, (0, 0, 255), -1)
            if len(warp_pts) == 4:
                cv2.polylines(vis, [np.int32(warp_pts)], True, (0, 255, 0), 2)
            cv2.imshow("raw", vis)
            continue

        # ---------- Run phase ----------
        bird      = cv2.warpPerspective(raw,      H, (W_TARGET, H_TARGET))
        roi_warp  = cv2.warpPerspective(roi_mask, H, (W_TARGET, H_TARGET),
                                        flags=cv2.INTER_NEAREST)
        bird_roi  = cv2.bitwise_and(bird, bird, mask=roi_warp)
        crop_img  = crop_by_mask(bird_roi, roi_warp)

        # — edge & Hough —
        gray   = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray, CANNY_T1, CANNY_T2, apertureSize=CANNY_APER)
        lines  = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH)
        lines_c_raw = cluster_lines(lines, RHO_BIAS, ANGLE_BIAS)
        lines_c = np.asarray(lines_c_raw, np.float32).reshape(-1, 1, 2)

        vis = crop_img.copy()
        for r, t in lines_c[:, 0]:
            a, b = np.cos(t), np.sin(t)
            x0, y0 = a * r, b * r
            p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(vis, p1, p2, (0, 255, 0), 2)

        # show
        cv2.imshow("raw", crop_img)
        cv2.imshow("bird", bird)
        cv2.imshow("crop", crop_img)
        cv2.imshow("edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        cv2.imshow("lines", vis)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
