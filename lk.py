import cv2
import numpy as np
import time
import os

class LucasKanade_OpticalFlow:
    def __init__(self, video_path, output_folder='optical_flow_images'):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def detect_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.001, minDistance=0.1)
        return corners, gray

    def calculate_optical_flow(self, prev_gray, gray, prev_pts):
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(25, 25), maxLevel=3,
                                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        h, w = gray.shape
        flow_image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros_like(flow_image)

        for new, old in zip(new_pts, prev_pts):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            flow_image = cv2.circle(flow_image, (int(a), int(b)), 3, (0, 255, 0), -2)

        flow_image = cv2.add(flow_image, mask)
        return flow_image


    def track_features(self):
        ret, frame = self.cap.read()
        prev_corners, prev_gray = self.detect_corners(frame)
        frame_count = 0

        while ret:
            ret, frame = self.cap.read()
            if not ret:
                break

            new_corners, gray = self.detect_corners(frame)
            flow_image = self.calculate_optical_flow(prev_gray, gray, prev_corners)

            #cv2.imwrite(os.path.join(self.output_folder, f'optical_flow_{frame_count:04d}.png'), flow_image)

            frame_count += 1
            prev_gray = gray
            prev_corners = new_corners.reshape(-1, 1, 2)

            cv2.imshow('Optical Flow', flow_image)
            k = cv2.waitKey(30) & 0xff
            if k == 27 or k == ord('q'):  # 'q' key to quit
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "320.mp4"
    feature_tracker = LucasKanade_OpticalFlow(video_path)
    feature_tracker.track_features()
