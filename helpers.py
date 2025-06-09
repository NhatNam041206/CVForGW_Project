import numpy as np
import cv2
def rotate(img,ROTATE_CW_DEG):
    d = ROTATE_CW_DEG % 360
    if d == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def cluster_lines(lines,RHO_BIAS,ANGLE_BIAS):
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

# def get_homography(pts,W_TARGET=640,H_TARGET=480):
#     dst = np.float32([[0,0],[W_TARGET,0],[W_TARGET,H_TARGET],[0,H_TARGET]])
#     return cv2.getPerspectiveTransform(np.float32(pts), dst)

def bird_view(corner_pts, W_TARGET=640, H_TARGET=480):
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

def get_roi(roi_pts,frame):
    roi_pts.sorted(key=lambda p: (p[1], p[0]))  # Sort by y, then by x

    x1, y1 = roi_pts[0]
    x2, y2 = roi_pts[1]
    x3, y3 = roi_pts[2]
    x4, y4 = roi_pts[3]

    # The position of lines following this order: up(small), left, down(large), right 
    # x1,y1: top-left corner 
    # x2,y2: top-right corner
    # x3,y3: bottom-right corner
    # x4,y4: bottom-left corner
    # (with x-axis, and y-axis following OpenCV coordinates)
    #
    #
    # Calculate each lines based on the points
    delta_x=np.array([x2-x1,
             x1-x4,
             x3-x4,
             x3-x2])

    delta_y=np.array([y2-y1,
             y1-y4,
             y3-y4,
             y3-y2])

    

