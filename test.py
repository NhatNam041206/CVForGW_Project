# test.py – v7: use ROI points directly for bird‑eye (no extra warp phase)
# -------------------------------------------------------------
# Flow
#   1. ROI phase  – click exactly 4 points of the area of interest.
#   2. Run phase  – homography built from those 4 points, warp frame & mask.
# -------------------------------------------------------------

import cv2
import numpy as np
from helpers import rotate, cluster_lines, create_binary_quad

# ───────── parameters ───────────────────────────────────────
ROTATE_CW_DEG = 90
PREVIEW_W, PREVIEW_H = 640, 480   # resize camera frame for preview

CANNY_T1, CANNY_T2 = 40, 120
CANNY_APER         = 3
HOUGH_RHO          = 1
HOUGH_THETA        = np.pi / 180
HOUGH_THRESH       = 140
ANGLE_BIAS, RHO_BIAS = 0.3, 20

# ───────── runtime state ────────────────────────────────────
phase = "roi"           # "roi" → "run"
roi_pts = []            # 4 clicked points
roi_mask = None         # binary mask in preview space
H = None                # homography matrix
DST_W = PREVIEW_W       # will be overwritten when H built
DST_H = PREVIEW_H

# ───────── helpers ─────────────────────────────────────────

def order_pts(pts):
    """Return points in TL, TR, BR, BL order."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)[:, 0]
    return np.array([
        pts[np.argmin(s)], pts[np.argmin(d)],
        pts[np.argmax(s)], pts[np.argmax(d)]
    ], np.float32)


def build_H(src_pts):
    src = order_pts(src_pts)
    # destination size computed from ROI dimensions
    w_top  = np.hypot(*(src[1] - src[0]))
    w_bot  = np.hypot(*(src[2] - src[3]))
    h_left = np.hypot(*(src[3] - src[0]))
    h_right= np.hypot(*(src[2] - src[1]))
    dst_w = int(max(w_top, w_bot))
    dst_h = int(max(h_left, h_right))
    dst_w = max(dst_w, 1); dst_h = max(dst_h, 1)
    dst = np.float32([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]])
    H = cv2.getPerspectiveTransform(src, dst)
    return H, dst_w, dst_h


def bbox_from_mask(mask):
    nz = cv2.findNonZero(mask)
    if nz is None:
        return 0, 0, mask.shape[1], mask.shape[0]
    return cv2.boundingRect(nz)

# ───────── mouse callback ──────────────────────────────────

def cb_roi(event, x, y, *_):
    global phase, roi_pts, roi_mask, H, DST_W, DST_H
    if phase != "roi" or event != cv2.EVENT_LBUTTONDOWN:
        return
    roi_pts.append([x, y])
    if len(roi_pts) == 4:
        # build mask in preview coordinates
        roi_mask = create_binary_quad(roi_pts, (PREVIEW_H, PREVIEW_W))
        roi_mask = (roi_mask > 0).astype(np.uint8) * 255
        # homography to rectangle
        H, DST_W, DST_H = build_H(roi_pts)
        phase = "run"
        cv2.setMouseCallback("raw", lambda *a: None)  # disable further clicks

# ───────── main loop ───────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    # windows
    cv2.namedWindow("raw");   cv2.setMouseCallback("raw", cb_roi)
    cv2.namedWindow("bird")
    cv2.namedWindow("roi_warp")
    cv2.namedWindow("crop")
    cv2.namedWindow("edges")
    cv2.namedWindow("lines")

    global phase, roi_pts, roi_mask, H, DST_W, DST_H

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = rotate(frame, ROTATE_CW_DEG)
        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        raw = frame.copy()

        # keyboard
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('r'), ord('R')):
            roi_pts.clear(); roi_mask = None; H = None
            phase = "roi"
            cv2.setMouseCallback("raw", cb_roi)
            continue

        # ROI selection phase
        if phase == "roi":
            vis = raw.copy()
            for p in roi_pts:
                cv2.circle(vis, tuple(p), 4, (255, 0, 255), -1)
            if len(roi_pts) == 4:
                cv2.polylines(vis, [np.int32(roi_pts)], True, (255, 255, 0), 2)
            cv2.imshow("raw", vis)
            continue

        # Run phase
        bird = cv2.warpPerspective(raw, H, (DST_W, DST_H))
        roi_warp = cv2.warpPerspective(roi_mask, H, (DST_W, DST_H), flags=cv2.INTER_NEAREST)
        cv2.imshow("roi_warp", cv2.cvtColor(roi_warp, cv2.COLOR_GRAY2BGR))

        # crop via bbox of roi_warp
        x, y, w, h = bbox_from_mask(roi_warp)
        crop_img = bird[y:y+h, x:x+w]
        roi_sub  = roi_warp[y:y+h, x:x+w]
        bird_roi = cv2.bitwise_and(crop_img, crop_img, mask=roi_sub)

        # edge & Hough
        gray = cv2.cvtColor(bird_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, CANNY_T1, CANNY_T2, apertureSize=CANNY_APER)
        raw_lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH)
        lines_c_raw = cluster_lines(raw_lines, RHO_BIAS, ANGLE_BIAS)
        lines_c = np.asarray(lines_c_raw, np.float32).reshape(-1, 1, 2)

        vis = bird_roi.copy()
        for r, t in lines_c[:, 0]:
            a, b = np.cos(t), np.sin(t)
            x0, y0 = a*r, b*r
            p1 = (int(x0 + 1000*(-b)), int(y0 + 1000*a))
            p2 = (int(x0 - 1000*(-b)), int(y0 - 1000*a))
            cv2.line(vis, p1, p2, (0, 255, 0), 2)

        # show
        cv2.imshow("raw", bird_roi)
        cv2.imshow("bird", bird)
        cv2.imshow("crop", bird_roi)
        cv2.imshow("edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        cv2.imshow("lines", vis)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
