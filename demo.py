import cv2
import numpy as np
from argparse import ArgumentParser

def compute_dense_optical_flow(method, video_path, frame1_index=0, frame2_index=30):
    """
    Extract two frames from a video and compute dense optical flow between them.
    
    Args:
        method: Optical flow method to use
        video_path: Path to the input video file
        frame1_index: Index of first frame (default: 0)
        frame2_index: Index of second frame (default: 1)
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
    
    # Calculate dense optical flow
    flow = method(gray1, gray2, None)
    
    # Create HSV image for visualization
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    
    # Convert flow to polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Use Hue and Value to encode the optical flow
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert HSV to BGR for visualization
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Display results
    cv2.imshow('Frame 1', frame1)
    cv2.imshow('Frame 2', frame2)
    cv2.imshow('Optical Flow', flow_vis)
    
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
    
    args = parser.parse_args()
    
    if args.algorithm == "lucaskanade_dense":
        method = cv2.optflow.calcOpticalFlowSparseToDense
        compute_dense_optical_flow(method, args.video_path, args.frame1, args.frame2)

if __name__ == "__main__":
    main()