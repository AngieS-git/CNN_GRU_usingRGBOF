import cv2
import numpy as np

def dense_pyramid_lucas_kanade(video_path):
    """
    Displays Dense Pyramid Lucas-Kanade optical flow for a video in a window.

    Args:
        video_path (str): Path to the input video file.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Warning: Could not read the first frame from {video_path}.")
        cap.release()
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create a dense grid of points to track
    h, w = prev_gray.shape
    grid_step = 10  # Distance between points in the grid
    points = np.array([
        [x, y] for y in range(0, h, grid_step)
                for x in range(0, w, grid_step)
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),  # Size of the search window at each pyramid level
        maxLevel=2,        # Maximum pyramid level
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, points, None, **lk_params
        )

        # Filter out points that were not successfully tracked
        good_old = points[status.ravel() == 1]
        good_new = next_points[status.ravel() == 1]

        # Create a blank image for visualization
        flow_visualization = frame.copy()

        # Draw the optical flow vectors
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            flow_visualization = cv2.line(flow_visualization, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            flow_visualization = cv2.circle(flow_visualization, (int(a), int(b)), 3, (0, 0, 255), -1)

        # Display the optical flow visualization
        cv2.imshow("Dense Pyramid Lucas-Kanade Optical Flow", flow_visualization)

        # Update the previous frame and points
        prev_gray = gray.copy()
        points = good_new.reshape(-1, 1, 2)

        # Exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"D:\MAPUA\CNN_GRU_usingRGBOF\downgraded_raw\LumbarSideBends\VID_20240128_150039_downgraded.mp4"  # Replace with your video file path
    dense_pyramid_lucas_kanade(video_path)