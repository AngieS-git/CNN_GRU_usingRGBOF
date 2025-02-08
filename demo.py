import cv2
import numpy as np
from argparse import ArgumentParser

def create_dense_grid(frame, step=10):
    """
    Create a dense grid of points to track across the frame.

    Args:
        frame: The input image frame.
        step: Spacing between grid points.

    Returns:
        A numpy array of shape (N,1,2) with coordinates of the grid points.
    """
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    return np.float32(np.vstack((x, y)).T).reshape(-1, 1, 2)

def compute_dense_lucas_kanade(video_path, frame1_index=0, frame2_index=30, step=10):
    """
    Compute dense optical flow using Lucas-Kanade method with a uniform grid of points.

    Args:
        video_path: Path to the input video file.
        frame1_index: Index of first frame.
        frame2_index: Index of second frame.
        step: Distance between tracked grid points.
    """
    # Read the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame2_index >= total_frames:
        raise ValueError(f"Frame index {frame2_index} exceeds video length of {total_frames} frames")
    
    # Read first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_index)
    ret, frame1 = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    # Read second frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_index)
    ret, frame2 = cap.read()
    if not ret:
        raise ValueError("Could not read second frame")
    
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Generate a dense grid of points
    p0 = create_dense_grid(gray1, step)

    # Define parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Compute optical flow using Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw motion vectors (arrows)
    mask = np.zeros_like(frame1)
    for (new, old) in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 1)
        frame2 = cv2.circle(frame2, (int(a), int(b)), 2, (0, 0, 255), -1)

    # Overlay optical flow visualization
    output = cv2.add(frame2, mask)

    # Display results
    cv2.imshow('Frame 1', frame1)
    cv2.imshow('Frame 2', frame2)
    cv2.imshow('Lucas-Kanade Dense Optical Flow', output)
    
    while True:
        k = cv2.waitKey(25) & 0xFF
        if k == 27:  # Press ESC to exit
            break
    
    cv2.destroyAllWindows()
    cap.release()

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["lucaskanade_dense"],
        required=True,
        help="Optical flow algorithm to use",
    )
    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to the video file",
    )
    parser.add_argument(
        "--frame1",
        type=int,
        default=0,
        help="Index of first frame (default: 0)",
    )
    parser.add_argument(
        "--frame2",
        type=int,
        default=30,
        help="Index of second frame (default: 30)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Grid spacing for dense flow visualization (default: 10)",
    )
    
    args = parser.parse_args()
    
    if args.algorithm == "lucaskanade_dense":
        compute_dense_lucas_kanade(args.video_path, args.frame1, args.frame2, args.step)
    else:
        raise ValueError("Invalid algorithm selected.")

if __name__ == "__main__":
    main()
