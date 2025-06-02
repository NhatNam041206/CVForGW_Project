import cv2
import numpy as np
from collections import defaultdict
from math import atan2, cos, sin, degrees

# ──────────────────────────────────────────────
# Tunable parameters (edit for your floor/camera)
# ──────────────────────────────────────────────
ROTATE_CW_DEG    = 90                # 0, 90, 180, 270; clockwise (+) degrees
BORDER_CROP      = 100                 # pixels to ignore around full frame
TILE_PX          = 200                 # ~ pixel length of one tile edge when centred
THETA_RES        = np.pi/360          # Hough θ resolution (0.5°)
HOUGH_THRESH     = 140                # Votes threshold – raise ⇒ fewer segments
MIN_LINE_LENGTH  = int(0.8 * TILE_PX) # Shortest segment kept (px)
MAX_LINE_GAP     = int(0.15 * TILE_PX) # Max gap to stitch (px)
EPS_RHO          = 15                 # Post‑merge tolerance ρ (px)
EPS_THETA        = np.deg2rad(1.0)    # Post‑merge tolerance θ (rad)
ANGLE_BIN_RAD    = np.deg2rad(5)      # Cluster bin width for left/right grouping
ROI_KEEP         = 0.66               # Keep lower fraction of the frame
DRAW_LEN         = 120                # Length of heading & centre lines to draw (px)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def apply_rotation(frame):
    """Rotate frame clockwise by 0/90/180/270 degrees as configured."""
    if ROTATE_CW_DEG % 360 == 0:
        return frame
    if ROTATE_CW_DEG % 360 == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if ROTATE_CW_DEG % 360 == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if ROTATE_CW_DEG % 360 == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("ROTATE_CW_DEG must be 0, 90, 180, or 270")

def seg_rho_theta(x1, y1, x2, y2):
    """Return (ρ, θ) of a segment in *math* coords (y up)."""
    dy, dx = -(y2 - y1), (x2 - x1)           # invert dy because image y goes down
    theta  = atan2(dy, dx)
    rho    = x1 * cos(theta) + y1 * sin(theta)
    return rho, theta


def merge_segments(segments):
    """Merge duplicate / co‑linear segments using (ρ, θ) buckets."""
    buckets = {}
    for x1, y1, x2, y2 in segments:
        rho, th = seg_rho_theta(x1, y1, x2, y2)
        key     = (int(rho / EPS_RHO), int(th / EPS_THETA))
        l2      = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if key not in buckets or l2 > buckets[key][1]:
            buckets[key] = ((x1, y1, x2, y2), l2)
    return [seg for seg, _ in buckets.values()]


def extract_heading(segments):
    """Return heading (rad) and diagnostic angles (α1, α2) plus grouping."""
    if not segments:
        return None, {}, (None, None)

    # compute angles of each segment relative to +x axis
    angles = []
    for x1, y1, x2, y2 in segments:
        dy, dx = -(y2 - y1), (x2 - x1)
        angles.append(atan2(dy, dx))

    # group by angle bin
    bins = defaultdict(list)
    for seg, ang in zip(segments, angles):
        bins[int(ang / ANGLE_BIN_RAD)].append((seg, ang))

    # take two largest groups ⇒ left & right
    top_bins = sorted(bins.values(), key=len, reverse=True)[:2]
    if len(top_bins) != 2:
        return None, {}, (None, None)

    alpha1 = np.mean([ang for _, ang in top_bins[0]])
    alpha2 = np.mean([ang for _, ang in top_bins[1]])
    heading = 0.5 * (alpha1 + alpha2)

    grouped = {
        'left':  [s for s, _ in top_bins[0]],
        'right': [s for s, _ in top_bins[1]],
    }
    return heading, grouped, (alpha1, alpha2)

# ──────────────────────────────────────────────
# Frame processing
# ──────────────────────────────────────────────

def process_frame(frame_bgr, frame_idx, timer_prev, debug_draw=True):
    if BORDER_CROP:
        frame_bgr = frame_bgr[BORDER_CROP:-BORDER_CROP,
                        BORDER_CROP:-BORDER_CROP]
    h, w = frame_bgr.shape[:2]
    # ---- ROI ----
    roi_offset = int((1 - ROI_KEEP) * h)
    roi = frame_bgr[roi_offset:, :]

    # ---- Gray & edges ----
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # ---- HoughP ----
    lines_p = cv2.HoughLinesP(edges, 1, THETA_RES, HOUGH_THRESH,
                              minLineLength=MIN_LINE_LENGTH,
                              maxLineGap=MAX_LINE_GAP)
    if lines_p is None:
        return None, None, frame_bgr, edges, timer_prev

    raw_segments = [tuple(l[0]) for l in lines_p]
    clean_segments = merge_segments(raw_segments)
    heading, groups, (alpha1, alpha2) = extract_heading(clean_segments)

    # ---- Draw visualisation ----
    vis = frame_bgr.copy()
    off = roi_offset
    # draw all segments (grey)
    for x1, y1, x2, y2 in clean_segments:
        cv2.line(vis, (x1, y1 + off), (x2, y2 + off), (192, 192, 192), 1)
    # left green, right red
    for x1, y1, x2, y2 in groups.get('left', []):
        cv2.line(vis, (x1, y1 + off), (x2, y2 + off), (0, 255, 0), 2)
    for x1, y1, x2, y2 in groups.get('right', []):
        cv2.line(vis, (x1, y1 + off), (x2, y2 + off), (0, 0, 255), 2)

    # central imaginary line (vertical)
    start = (w // 2, h - 1)
    end_c = (w // 2, h - 1 - DRAW_LEN)
    cv2.line(vis, start, end_c, (255, 255, 0), 2)  # cyan

    # actual heading line
    if heading is not None:
        dx = int(DRAW_LEN * cos(heading))
        dy = int(DRAW_LEN * sin(heading))
        end_h = (start[0] + dx, start[1] - dy)      # minus because y down
        cv2.arrowedLine(vis, start, end_h, (0, 255, 255), 2, tipLength=0.15)  # yellow

    # ---- FPS ----
    tick_now = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (tick_now - timer_prev) if timer_prev else 0
    timer_prev = tick_now

    # ---- HUD text ----
    if heading is not None:
        cv2.putText(vis, f"heading: {degrees(heading):.1f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2, cv2.LINE_AA)
    if alpha1 is not None and alpha2 is not None:
        cv2.putText(vis, f"α1: {degrees(alpha1):.1f} | α2: {degrees(alpha2):.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2, cv2.LINE_AA)

    # ---- Log to console ----
    if heading is not None:
        print(f"frame {frame_idx:04d}	heading={degrees(heading):5.1f}°\tα1={degrees(alpha1):5.1f}°\tα2={degrees(alpha2):5.1f}°\tfps={fps:4.1f}")
    else:
        print(f"frame {frame_idx:04d}\t(no heading)\tfps={fps:4.1f}")

    return heading, groups, vis, edges, timer_prev

# ──────────────────────────────────────────────
# Main loop demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    print("Press ESC to quit…")
    frame_idx = 0
    t_prev = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame=apply_rotation(frame)
        heading, groups, vis, edges, t_prev = process_frame(frame, frame_idx, t_prev, debug_draw=True)

        cv2.imshow("Lane view", vis)
        cv2.imshow("Edges", edges)

        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
