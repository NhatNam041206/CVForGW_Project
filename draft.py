import cv2
import numpy as np
import math
import os
from helpers import rotate, create_binary_quad, apply_roi, cluster_lines

class CamTester:
    def __init__(self):
        self.saved_path = 'points.txt'
        self.ANGLE_TRIANGLE = math.radians(70)
        self.ACCEPT = 25
        self.CROP_SIZE = 100
        self.FLIPCODE = 1
        self.CANNY_T1 = 40
        self.CANNY_T2 = 120
        self.CANNY_APER = 3
        self.BLUR_KSIZE = 5
        self.ROTATE_CW_DEG = 270
        self.HOUGH_RHO = 1
        self.HOUGH_THETA = np.pi / 180
        self.HOUGH_THRESH = 95
        self.ANGLE_BIAS = 0.3
        self.RHO_BIAS = 20
        self.W, self.H = 480, 640
        self.corner_points = []
        self.roi_created = False

    def on_click_roi(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corner_points) < 2:
            if len(self.corner_points) == 0:
                print('Left point inputted!')
                self.corner_points.append([x, y])
            elif x > self.corner_points[0][0]:
                print('Right point inputted!')
                self.corner_points.append([x, self.corner_points[0][1]])

    def save_points(self):
        np.savetxt(self.saved_path, np.array(self.corner_points), fmt='%d')
        print(f"Saved corner points to {self.saved_path}")

    def load_points(self):
        if os.path.exists(self.saved_path):
            self.corner_points = np.loadtxt(self.saved_path, dtype=int).tolist()
            if isinstance(self.corner_points[0], int):  # If only one point, wrap it in list
                self.corner_points = [self.corner_points]
            print(f"Loaded corner points from {self.saved_path}")
            if len(self.corner_points) == 3:
                self.roi_created = True

    def get_roi(self):
        choice = input("Use saved corner points (ROI)? (y/n): ").strip().lower()
        if choice == 'y':
            print('Loading corner_points from points.txt')
            self.load_points()
            return
        print('Creating corner_points.')
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open camera")

        cv2.namedWindow('inputted')
        cv2.namedWindow('raw')
        cv2.setMouseCallback('inputted', self.on_click_roi)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = rotate(frame, self.ROTATE_CW_DEG)
            frame = cv2.flip(frame, self.FLIPCODE)
            frame = cv2.resize(frame, (self.W, self.H))

            raw = frame.copy()
            for p in self.corner_points:
                cv2.circle(raw, tuple(p), 5, (0, 0, 255), -1)
                cv2.putText(raw, f"{p[0]},{p[1]}", tuple(p),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            if len(self.corner_points) == 2:
                p1, p2 = self.corner_points
                distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                x_3 = int(round((p1[0] + p2[0]) / 2))
                height = (distance / 2) * math.tan(self.ANGLE_TRIANGLE)
                y_3 = int(round(p1[1] - height))
                apex = [x_3, y_3]
                self.corner_points.append(apex)
                cv2.circle(raw, (x_3, p1[1]), 5, (0, 0, 255), -1)
                cv2.circle(raw, (x_3, y_3), 5, (0, 0, 255), -1)

            if len(self.corner_points) == 3:
                cv2.line(raw, self.corner_points[0], self.corner_points[1], (0, 255, 255), 2)
                cv2.line(raw, self.corner_points[0], self.corner_points[2], (0, 255, 255), 2)
                cv2.line(raw, self.corner_points[1], self.corner_points[2], (0, 255, 255), 2)

            cv2.imshow('raw', frame)
            cv2.imshow('inputted', raw)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')):
                break
            if k in (ord('c'), ord('C')) and len(self.corner_points) == 3:
                print(f'Left point: {self.corner_points[0]}; Right point: {self.corner_points[1]}; Top point: {self.corner_points[2]}')
                self.roi_created = True
                break
            if k in (ord('r'), ord('R')):
                self.corner_points.clear()

        cap.release()
        cv2.destroyAllWindows()

        if self.roi_created:
            save = input("Save corner points to file? (y/n): ").strip().lower()
            if save == 'y':
                self.save_points()

    def detect_lines(self):
        white_bg = np.full((self.H, self.W), 255, dtype=np.uint8)
        mask = create_binary_quad(self.corner_points, img_size=(self.H, self.W))
        roi = apply_roi(mask)

        cap = cv2.VideoCapture(0)
        while True and self.roi_created:
            ret, frame = cap.read()
            if not ret:
                break

            frame = rotate(frame, self.ROTATE_CW_DEG)
            frame = cv2.flip(frame, self.FLIPCODE)
            frame = cv2.resize(frame, (self.W, self.H))

            y_bottom = max(self.corner_points[0][1], self.corner_points[1][1])
            y_crop = max(0, y_bottom - self.CROP_SIZE)
            mask_3ch = cv2.merge([roi] * 3)
            frame = np.where(mask_3ch == 255, frame, 255).astype(np.uint8)
            frame = frame[:y_crop, :]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.CANNY_T1, self.CANNY_T2, apertureSize=self.CANNY_APER)
            lines = cv2.HoughLines(edges, self.HOUGH_RHO, self.HOUGH_THETA, self.HOUGH_THRESH)

            hough_vis = frame.copy()
            if lines is None:
                cv2.imshow('cam-tester', hough_vis)
                continue

            flines = cluster_lines(lines, self.RHO_BIAS, self.ANGLE_BIAS)
            flines = flines.reshape(len(flines), 2)

            min_ang = 180
            for rho, theta in flines:
                angle_x_axis = math.degrees(np.pi / 2 - theta)
                if -self.ACCEPT <= angle_x_axis <= self.ACCEPT:
                    a, b = np.cos(theta), np.sin(theta)
                    x0, y0 = a * rho, b * rho
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                    cv2.line(hough_vis, pt1, pt2, (255, 0, 0), 2)
                    min_ang = min(min_ang, angle_x_axis)

            if 90 - min_ang > 0:
                angle = 90 - min_ang
                print(f'Vertical Angle: {angle :.2f}')
                H_CROP = self.H - self.CROP_SIZE
                start = (self.W // 2, H_CROP - 20)
                end_c = (self.W // 2, int(H_CROP - H_CROP * 0.4))
                radius = 100
                angle_end = -int(angle)
                dx = int(H_CROP * 0.4 * np.cos(math.radians(angle)))
                dy = int(H_CROP * 0.4 * np.sin(math.radians(angle)))
                end_h = (start[0] + dx, start[1] - dy)
                cv2.arrowedLine(hough_vis, start, end_h, (0, 0, 255), 2, tipLength=0.15)
                cv2.ellipse(hough_vis, start, (radius, radius), 0, 0, angle_end, (0, 0, 255), 2)
                text_angle = f"{angle:.2f}"
                text_x = int(start[0] + radius * math.cos(math.radians(angle / 2)))
                text_y = int(start[1] - radius * math.sin(math.radians(angle / 2)))
                cv2.putText(hough_vis, text_angle, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.line(hough_vis, start, end_c, (255, 255, 0), 3)
            else:
                print("Vertical Angle: Not detected!")

            cv2.imshow("cam_tester", hough_vis)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = CamTester()
    app.get_roi()
    if app.roi_created:
        app.detect_lines()