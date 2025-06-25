import numpy as np
import cv2
import math
def rotate(img,ROTATE_CW_DEG):
    d = ROTATE_CW_DEG % 360
    if d == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def cluster_lines(lines,RHO_BIAS,ANGLE_BIAS):
    visited=[]
    lines_filtered=[]
    for i in range(len(lines)):
        if list(lines[i][0]) in visited:
            continue

        cluster, clustered = [], False
        for j in range(i, len(lines)):
            angleD = abs(lines[i][0][1] - lines[j][0][1])
            pD     = abs(lines[i][0][0] - lines[j][0][0])

            if angleD < ANGLE_BIAS and pD < RHO_BIAS:
                cluster.extend([lines[j]])
                visited.append(list(lines[j][0]))
                clustered = True

        # keep average   or   raw line
        out = np.mean(cluster, axis=0) if clustered else lines[i]
        lines_filtered.append(out)
    return np.array(lines_filtered)

# def get_homography(pts,W_TARGET=640,H_TARGET=480):
#     dst = np.float32([[0,0],[W_TARGET,0],[W_TARGET,H_TARGET],[0,H_TARGET]])
#     return cv2.getPerspectiveTransform(np.float32(pts), dst)

# def bird_view(corner_pts, W_TARGET=640, H_TARGET=480):
#     src_points = np.float32(corner_pts)
    
#     # Define the destination points (to a square, but this will be adjusted later)
#     dst_points = np.float32([[0, 0], [350, 0], [350, 400], [0, 400]])

#     # Compute the perspective transform matrix
#     matrix = cv2.getPerspectiveTransform(src_points, dst_points)

#     # Transform the corners of the original image to determine output size
#     h, w = H_TARGET, W_TARGET
#     original_corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
#     transformed_corners = cv2.perspectiveTransform(original_corners, matrix)

#     # Calculate the bounding box of the transformed corners
#     x_coords = transformed_corners[:,0,0]
#     y_coords = transformed_corners[:,0,1]
#     min_x, max_x = np.min(x_coords), np.max(x_coords)
#     min_y, max_y = np.min(y_coords), np.max(y_coords)

#     # Compute required output size and adjust the matrix to avoid cropping
#     output_width = max(w, int(np.ceil(max_x - min_x)))
#     output_height = max(h, int(np.ceil(max_y - min_y)))
#     adjustment_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
#     adjusted_matrix = adjustment_matrix @ matrix
#     return adjusted_matrix, (output_width, output_height)


# def create_binary_quad(points, img_size=(640, 480)):

#     mask = np.zeros(img_size, dtype=np.uint8)

#     pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

#     cv2.fillPoly(mask, [pts], 255)

#     return mask

# def apply_roi(roi,img_size=(640, 480)):
#     # resize ROI to match the original image size
#     # roi = cv2.resize(src=roi, dsize=(img_size[0], img_size[1]))    
#     # scale ROI to [0, 1] => binary mask
#     thresh, roi = cv2.threshold(roi, thresh=128, maxval=1, type=cv2.THRESH_BINARY)

#     return roi
def line_intersection(rho1, theta1, rho2, theta2, eps=1e-9):
    """
    Return (x, y) where the two Hough-space lines intersect.
    Returns None if the lines are (nearly) parallel.
    Angles in radians.
    """
    
    denom = np.sin(theta2 - theta1)         # = Δ above
    if abs(denom) < eps:
        return None  # parallel
    x = (rho1 * np.sin(theta2)-rho2 * np.sin(theta1)) / denom
    y = (rho2 * np.cos(theta1)-rho1 * np.cos(theta2)) / denom
    return x, y

import numpy as np

def horizon_split(lines, W, y_hor, eps=1e-9):
    """
    lines : ndarray (N,2)  [rho, theta]
    W     : image width
    y_hor : pixel-row của horizon
    return left_mask, right_mask
    """
    rho, th  = lines[:,0], lines[:,1]
    sin_t, cos_t = np.sin(th), np.cos(th)

    # vertical?
    vert = np.abs(sin_t) < eps
    # slope & intercept
    a = np.where(vert, np.inf, -cos_t / sin_t)
    b = np.where(vert, np.nan,  rho   / sin_t)

    # x at horizon
    x_hor = np.where(vert, rho / cos_t, (y_hor - b) / a)
    left  = x_hor <  W/2
    right = x_hor >= W/2
    return left & ~vert, right & ~vert, vert



def angle_detecting(lines,img_size=(480,640),debug=False): # Set debug to False to unlog the proccess of checking right and left lines' indexes
    H,W=img_size
    rhos=lines[:,0]
    thetas=lines[:,1]
    lines_index=np.array(range(len(rhos)))
    
    x_intercepts=rhos/np.cos(thetas)
    filter_cond=np.logical_and(x_intercepts<=W, x_intercepts>=0)


    x_intercepts_f=x_intercepts[filter_cond]
    rhos_f=rhos[filter_cond]
    lines_index_f=lines_index[filter_cond]

    # Initialize the conditions
    left_cond, right_cond = horizon_split(lines[np.array(lines_index_f)],480,640)
    # print(len(left_cond),rhos_f)
    # print(left_cond,right_cond)
    # Sides seperation
    rhos_f_l=rhos_f[left_cond]
    rhos_f_r=rhos_f[right_cond]

    left_lines=lines_index_f[left_cond]
    right_lines=lines_index_f[right_cond]


    #Left lines calculate delta_y
    delta_y_l=x_intercepts_f[left_cond]/np.tan(thetas[left_lines])

    #Right lines calculate delta_y
    delta_y_r=(W-x_intercepts_f[right_cond])/np.tan(2*np.pi-thetas[right_lines])

    # Calculate startpoints and endpoints
    #
    #
    end_points_l=[]
    start_points_l=[(x_,0) for x_ in x_intercepts_f[left_cond]]

    end_points_r=[]
    start_points_r=[(x_,0) for x_ in x_intercepts_f[right_cond]]


    for i,delta_y in enumerate(delta_y_l):
        if delta_y<=H:
            end_points_l.append((0,rhos_f_l[i]/np.sin(thetas[left_lines[i]])))
            continue
        end_points_l.append((rhos_f_l[i]-H*np.sin(thetas[left_lines[i]])/np.cos(thetas[left_lines[i]]),H))

    for i,delta_y in enumerate(delta_y_r):
        if delta_y<=H:
            end_points_r.append((W,(rhos_f_r[i]-W*np.cos(thetas[right_lines[i]]))/np.sin(thetas[right_lines[i]])))
            continue
        end_points_r.append(((rhos_f_r[i]-H*np.sin(thetas[right_lines[i]]))/np.cos(thetas[right_lines[i]]),H))
    
    if len(end_points_r)<1:
        print(f"Caution: can't find right lines. Left lines: {len(end_points_l)}")
        return False,False,False
    if len(end_points_l)<1:
        print(f"Caution: can't find left lines. Right lines: {len(end_points_r)}")
        return False,False,False
    if len(end_points_l)==0 and len(end_points_r)==0:
        print(f"Can't find left right lines!")
        return False,False,False
    # if debug:
    #     lines_index_dict=dict()
    #     print('Creating dictionary table for storing start, end points of lines based on their indexes.')
    #     for i in lines_index_f:
    #         if i in left_lines:
    #             pos=list(left_lines).index(i)
    #             lines_index_dict[str(i)]=(start_points_l[pos],end_points_l[pos])
    #             print(f'Find lines index "{i}" in position "{pos}" of left_lines. Values storing -> {i}: {(start_points_l[pos],end_points_l[pos])}')
    #         else:
    #             pos=list(right_lines).index(i)
    #             lines_index_dict[str(i)]=(start_points_r[pos],end_points_r[pos])
    #             print(f'Find lines index "{i}" in position "{pos}" of right_lines. Values storing -> {i}: {(start_points_r[pos],end_points_r[pos])}')
    #     print(f'Dictionary: {lines_index_dict}')
    
    # Sorting outliers and find intercept. 
    left_pt_outlier=min(end_points_l, key=lambda x: x[0])
    right_pt_outlier=max(end_points_r, key=lambda x: x[0])

    left_outlier_index=left_lines[end_points_l.index(left_pt_outlier)]
    right_outlier_index=right_lines[end_points_r.index(right_pt_outlier)]

    rho1, theta1, rho2, theta2=rhos[left_outlier_index],thetas[left_outlier_index],rhos[right_outlier_index],thetas[right_outlier_index]
    x_intercept_outlier,y_intercept_outlier=line_intersection(rho1, theta1, rho2, theta2)

    # Adding supplementary and accute angle conds
    distance=math.dist(left_pt_outlier,right_pt_outlier)
    if x_intercept_outlier<distance/2:
        print("LEFT")
        actual_angle_v2=abs(math.atan2((-H),(x_intercept_outlier-distance/2)))
    else:
        print("RIGHT")
        actual_angle_v2=abs(math.atan2((H),(distance/2-x_intercept_outlier))-np.pi)
    
    return actual_angle_v2,lines_index_f[left_cond],lines_index_f[right_cond]