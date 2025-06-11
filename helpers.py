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


def create_binary_quad(points, img_size=(640, 480)):

    mask = np.zeros(img_size, dtype=np.uint8)

    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(mask, [pts], 255)

    return mask

def apply_roi(roi,img_size=(640, 480)):
    # resize ROI to match the original image size
    # roi = cv2.resize(src=roi, dsize=(img_size[0], img_size[1]))    
    # scale ROI to [0, 1] => binary mask
    thresh, roi = cv2.threshold(roi, thresh=128, maxval=1, type=cv2.THRESH_BINARY)

    return roi

def angle_detecting(lines,img_size=(640, 480)):
    rhos=lines[:,0]
    thetas=lines[:,1]

    w,h=img_size=img_size

    lines_index=np.array(range(len(rhos)))

    # Running the initial calculations
    x_intercepts=rhos/np.cos(thetas)

    filter_matrix=np.logical_and(x_intercepts<=w,x_intercepts>=0)
    x_intercepts_f=x_intercepts[filter_matrix]
    rhos_f=rhos[filter_matrix]
    lines_index_f=lines_index[filter_matrix]

    left_lines_matrix=rhos_f>0
    right_lines_matrix=rhos_f<0
    
    left_lines=lines_index_f[left_lines_matrix]
    right_lines=lines_index_f[right_lines_matrix]

    rhos_f_l=rhos_f[left_lines_matrix]
    rhos_f_r=rhos_f[right_lines_matrix]

    delta_y_l=x_intercepts_f[left_lines_matrix]/np.tan(thetas[left_lines])
    delta_y_r=(w-x_intercepts_f[right_lines_matrix])/np.tan(2*np.pi-thetas[right_lines])

    # Calculate startpoints and endpoints
    #
    #
    end_points_l=[]
    start_points_l=[(x_,0) for x_ in x_intercepts_f[left_lines_matrix]]

    end_points_r=[]
    start_points_r=[(x_,0) for x_ in x_intercepts_f[right_lines_matrix]]


    for i,delta_y in enumerate(delta_y_l):
        if delta_y<=h:
            end_points_l.append((0,rhos_f_l[i]/np.sin(thetas[left_lines[i]])))
            continue
        end_points_l.append((rhos_f_l[i]-h*np.sin(thetas[left_lines[i]])/np.cos(thetas[left_lines[i]]),h))

    for i,delta_y in enumerate(delta_y_r):
        if delta_y<=h:
            end_points_r.append((w,(rhos_f_r[i]-w*np.cos(thetas[right_lines[i]]))/np.sin(thetas[right_lines[i]])))
            continue
        end_points_r.append(((rhos_f_r[i]-h*np.sin(thetas[right_lines[i]]))/np.cos(thetas[right_lines[i]]),h))
    
    delta_x_all=np.array(list(start_points_r)+list(start_points_l))[:,0]
    delta_y_all=np.array(list(delta_y_r)+list(delta_y_l))

    actual_angle_r=np.negative(np.arctan2(delta_y_r,np.array(start_points_r)[:,0])-np.pi/2) # Radian
    actual_angle_l=np.arctan2(delta_y_l,np.array(start_points_l)[:,0])-np.pi/2 # Radian

    print(f"Actual difference from central angle in degree form: {np.rad2deg(np.sum([actual_angle_l,actual_angle_r]))}")
    return np.sum([actual_angle_l,actual_angle_r])
