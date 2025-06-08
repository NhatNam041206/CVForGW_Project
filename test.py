import cv2
import numpy as np

"""
Camera‑settings test utility
––––––––––––––––––––––––––––
* Live preview of three stages:
  1. Gaussian‑blurred gray
  2. Canny edges
  3. Raw + filtered Hough (standard, not probabilistic)
* Re‑implementation of your “filtered_line” grouping using a
  single‑pointer sweep + set for visited, so no nested j‑loop.
  Complexity ≈ O(N log N) after initial sort.

Usage
-----
• Press  Q  to quit.
• Adjust the tunables (CANNY_T1, …) to taste.
"""

# ───────── tunables ───────────────────────────
CANNY_T1   = 40
CANNY_T2   = 120
CANNY_APER = 3
BLUR_KSIZE = 5
ROTATE_CW_DEG    = 0                 # 0, 90, 180, 270; clockwise
HOUGH_RHO      = 1
HOUGH_THETA    = np.pi / 180   # 1°
HOUGH_THRESH   = 140           # votes

# clustering tolerances (same as your script)
ANGLE_BIAS = 0.3               # rad  (~17°)
RHO_BIAS   = 20                # px

corner_points = []
dragging_point=None

# ───────── helpers ─────────
def apply_rotation(frame):
    if ROTATE_CW_DEG % 360 == 0:
        return frame
    if ROTATE_CW_DEG % 360 == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if ROTATE_CW_DEG % 360 == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if ROTATE_CW_DEG % 360 == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("ROTATE_CW_DEG must be 0/90/180/270")

def cluster_lines(lines):
    """Group similar (rho, theta) pairs. Return list of mean lines."""
    if lines is None or len(lines) == 0:
        return []
    # flatten to list of tuples
    lines = [tuple(l[0]) for l in lines]
    # sort by rho so we can sweep once
    lines.sort(key=lambda lt: lt[0])  # rho ascending

    visited = set()
    clusters = []

    for idx, (rho_i, theta_i) in enumerate(lines):
        if idx in visited:
            continue
        # open new cluster
        cluster_rhos   = [rho_i]
        cluster_thetas = [theta_i]
        visited.add(idx)

        # sweep forward until rho diff > RHO_BIAS
        j = idx + 1
        while j < len(lines) and abs(lines[j][0] - rho_i) < RHO_BIAS:
            rho_j, theta_j = lines[j]
            if abs(theta_j - theta_i) < ANGLE_BIAS:
                visited.add(j)
                cluster_rhos.append(rho_j)
                cluster_thetas.append(theta_j)
            j += 1

        # compute mean line for cluster
        clusters.append((float(np.mean(cluster_rhos)),
                         float(np.mean(cluster_thetas))))
    # convert back to shape (N,1,2) like cv2.HoughLines
    return np.array(clusters, dtype=np.float32).reshape(-1, 1, 2)

# Callback function for mouse events to capture the points
def click_event(event, x, y, flags, param):
    global corner_points, dragging_point
    
    # If left mouse button is clicked, check if it's near a point to start dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = float('inf')
        closest_point = None
        for i, point in enumerate(corner_points):
            dist = np.linalg.norm(np.array([x, y]) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                closest_point = i
        
        # Start dragging if a point is close enough to the click
        if closest_point is not None and min_dist < 10:  # 10 pixels threshold
            dragging_point = closest_point
    
    # If the mouse is moving and a point is being dragged
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            corner_points[dragging_point] = [x, y]  # Update the dragged point's coordinates
    
    # If the left mouse button is released, stop dragging
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None  # Stop dragging

def bird_view(frame, corner_points):
    # Define the source points for perspective transform (resize coordinates accordingly)
    src_points = np.float32(corner_points)
    
    # Define the destination points (to a square, but this will be adjusted later)
    dst_points = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Transform the corners of the original image to determine output size
    h, w = 640,480
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

    # Apply the perspective warp with computed output size and padding
    # warped_img = cv2.warpPerspective(frame, adjusted_matrix, (output_width, output_height), 
    #                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # # Resize the warped image to 650x1000 (same dimensions as the output)
    # warped_img = cv2.resize(warped_img, (650, 480), interpolation=cv2.INTER_AREA)

    return adjusted_matrix, (output_width, output_height)


# ───────── main loop ─────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")

print("[cam‑tester] press Q to exit …")
ret, frm0 = cap.read()

corner_points = [[50, 200], [100, 200], [100, 300], [50, 300]] # Top left, Top right, Bottom right, Bottom left
get_bird_view=False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame=apply_rotation(frame)
    frame=cv2.resize(frame,(640,480))
    # 1. blur + gray
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur  = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    # 2. edges
    edges = cv2.Canny(gray, CANNY_T1, CANNY_T2, apertureSize=CANNY_APER)

    # 3. Hough standard (ρ,θ)
    lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH)
    flines = cluster_lines(lines)

    # create overlay copy
    hough_vis = frame.copy()

    # draw raw lines (light gray) if any
    if lines is not None:
        if not get_bird_view:
            cv2.namedWindow('Input Image')
            cv2.setMouseCallback('Input Image', click_event)

            # Draw the points and the lines between them
            print('Coner points:', corner_points)
            for i in range(len(corner_points)):
                cv2.circle(hough_vis, tuple(corner_points[i]), 5, (0, 0, 255), -1)
                cv2.circle(hough_vis, tuple(corner_points[i]), 15, (42, 255, 255), 1)

                if i > 0:
                    cv2.line(hough_vis, tuple(corner_points[i-1]), tuple(corner_points[i]), (0, 255, 0), 2)
            
            # Close the rectangle if all 4 points are selected
            cv2.line(hough_vis, tuple(corner_points[3]), tuple(corner_points[0]), (0, 255, 0), 2)
            
            adjusted_matrix, (output_width, output_height) = bird_view(hough_vis, corner_points)
            
            # Apply the perspective warp with computed output size and padding
            warped_img = cv2.warpPerspective(frame.copy(), adjusted_matrix, (output_width, output_height), 
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

            warped_img = cv2.resize(warped_img, (640, 480), interpolation=cv2.INTER_AREA)
            
            # Show the original image with corner points
            cv2.imshow('Input Image', hough_vis)
            # Show the warped image
            cv2.imshow('Warped Image', warped_img)

            # Wait for the 'q' key to exit the loop
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('c'), ord('C'), 32):
                get_bird_view=True

            if key in (ord('r'), ord('R')):
                corner_points.clear()
                get_bird_view = False

            if key in (ord('q'), ord('Q')):
                break
                print("[info] corner points reset")
            continue
        else:
            for rho, theta in lines[:,0]:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv2.line(hough_vis, pt1, pt2, (180, 180, 180), 1)
    else:
        cv2.imshow('cam-tester',frame)
        print("[Warning] No lines detected")
        continue


    # draw filtered lines (green) thicker
    for rho, theta in flines[:,0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(hough_vis, pt1, pt2, (0, 255, 0), 2)

    # stack previews blur | edges | hough
    blur_vis  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    top_row   = np.hstack((blur_vis, edges_vis))
    bottom    = np.hstack((hough_vis, np.zeros_like(hough_vis)))
    stacked   = np.vstack((top_row, bottom))

    cv2.imshow("cam-tester", stacked)

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()