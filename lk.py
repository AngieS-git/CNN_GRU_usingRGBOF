import cv2
import numpy as np

def compute_optical_flow(prev_frame, next_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Detect good features to track (Shi-Tomasi corners)
    feature_params = dict(maxCorners=3000, qualityLevel=0.01, minDistance=0.1)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        good_new, good_old = [], []

    return good_old, good_new

def create_mei_mhi(flow, shape, tau=10):
    # Initialize MEI and MHI
    mei = np.zeros(shape, dtype=np.float32)  # Binary motion energy
    mhi = np.zeros(shape, dtype=np.float32)  # Motion history with decay

    # Update MEI and MHI
    for (new, old) in zip(flow[1], flow[0]):
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        
        # Draw motion on MEI (binary)
        cv2.line(mei, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
        
        # Draw motion on MHI (with intensity based on time)
        cv2.line(mhi, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)

    # Update MHI with decay
    mhi[mhi > 0] -= 255 / tau  # Decay over time
    mhi[mhi < 0] = 0  # Ensure no negative values

    return mei, mhi

def main():
    # Load two consecutive frames from a video
    prev_frame = cv2.imread(r'D:\MAPUA\CNN-GRU_exp\output_frames\frame_0000.jpg')
    next_frame = cv2.imread(r'D:\MAPUA\CNN-GRU_exp\output_frames\frame_0005.jpg')

    if prev_frame is None or next_frame is None:
        print("Error: Could not load images.")
        return

    # Compute optical flow
    good_old, good_new = compute_optical_flow(prev_frame, next_frame)

    if len(good_old) == 0 or len(good_new) == 0:
        print("Error: No optical flow vectors found.")
        return

    # Create MEI and MHI
    mei, mhi = create_mei_mhi((good_old, good_new), prev_frame.shape[:2], tau=10)

    # Normalize MEI for better visualization
    mei_norm = cv2.normalize(mei, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display only the MEI
    cv2.imshow('Motion Energy Image (MEI)', mei_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()