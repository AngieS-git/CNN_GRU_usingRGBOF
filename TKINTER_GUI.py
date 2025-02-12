import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
import threading
import time

# ----------------------------
# Define your class names
# ----------------------------
CLASSES_LIST = ['LumbarSideBends', 'QuadrupedThoracicRotation', 'SupineNeckLift']

# Global variables to track recording state and pre-warmed camera
recording_in_progress = False
prewarm_cap = None  # This will store the pre-warmed VideoCapture instance

# ----------------------------
# Optical Flow Functions (Lucas-Kanade)
# ----------------------------
def compute_optical_flow(prev_frame, next_frame):
    """
    Compute optical flow using the Lucas-Kanade method between two frames.
    Returns:
        good_old: Array of points from the previous frame.
        good_new: Array of corresponding points from the next frame.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    feature_params = dict(maxCorners=3000, qualityLevel=0.01, minDistance=0.1)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if p0 is None:
        return [], []
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)
    if p1 is not None and st is not None:
        st = st.flatten()
        good_old = p0[st == 1]
        good_new = p1[st == 1]
    else:
        good_old, good_new = [], []
    return good_old, good_new

def create_mei_mhi(flow, shape, tau=10):
    """
    Create Motion Energy Image (MEI) and Motion History Image (MHI) from optical flow vectors.
    """
    mei = np.zeros(shape, dtype=np.float32)
    mhi = np.zeros(shape, dtype=np.float32)

    for (new, old) in zip(flow[1], flow[0]):
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        cv2.line(mei, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
        cv2.line(mhi, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)

    mhi[mhi > 0] -= 255 / tau  
    mhi[mhi < 0] = 0

    return mei, mhi

def compute_optical_flow_sequence_lk(frames, target_size=(240, 320), tau=10):
    """
    For a list of frames, resize them to target_size (width, height),
    compute Lucas-Kanade optical flow between consecutive frames, and obtain the MEI.
    Returns an array of shape (num_frames, target_height, target_width, 1).
    """
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, target_size)
        resized_frames.append(resized)
    
    flow_sequence = []
    for i in range(len(resized_frames) - 1):
        prev_frame = resized_frames[i]
        next_frame = resized_frames[i + 1]
        good_old, good_new = compute_optical_flow(prev_frame, next_frame)
        if len(good_old) == 0 or len(good_new) == 0:
            mei = np.zeros((target_size[1], target_size[0]), dtype=np.float32)
        else:
            mei, _ = create_mei_mhi((good_old, good_new), (target_size[1], target_size[0]), tau=tau)
            mei = mei / 255.0  # Normalize to [0,1]
        mei = np.expand_dims(mei, axis=-1)  # Add channel dimension
        flow_sequence.append(mei.astype(np.float32))
    
    # Pad the sequence if necessary to match the number of frames.
    desired_length = len(frames)
    while len(flow_sequence) < desired_length:
        zero_frame = np.zeros((target_size[1], target_size[0], 1), dtype=np.float32)
        flow_sequence.append(zero_frame)
    
    return np.array(flow_sequence)

# ----------------------------
# RGB Preprocessing Function
# ----------------------------
def preprocess_rgb_frames(frames, target_size=(96, 96)):
    """
    Resize each frame to target_size (width, height), convert from BGR to RGB,
    and normalize pixel values to [0,1]. Returns an array of shape (num_frames, target_height, target_width, 3).
    """
    rgb_frames = []
    for frame in frames:
        resized = cv2.resize(frame, target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb_frames.append(rgb)
    return np.array(rgb_frames)

# ----------------------------
# Frame Sampling Function
# ----------------------------
def sample_frames_from_video(video_path, num_frames):
    """
    Samples `num_frames` uniformly from the input video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        print("Warning: Video has fewer frames than requested; using available frames.")
        num_frames = total_frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    current_index = 0
    ret = True
    while ret and current_index < total_frames:
        ret, frame = cap.read()
        if current_index in indices and frame is not None:
            frames.append(frame)
        current_index += 1
    cap.release()
    return frames

# ----------------------------
# Ensure Video Resolution Function
# ----------------------------
def ensure_video_resolution(frames, target_size=(320, 240)):
    """
    Resize each frame to the specified target resolution (width, height).
    """
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, target_size)
        resized_frames.append(resized_frame)
    return resized_frames

# ----------------------------
# Inference Function
# ----------------------------
def run_inference_on_video(video_path, model, num_frames=30):
    """
    Samples frames from the video, ensures resolution, preprocesses for both streams,
    and runs inference using the provided model.
    Returns predicted class index and prediction probabilities.
    """
    frames = sample_frames_from_video(video_path, num_frames)
    if len(frames) < num_frames:
        print("Error: Not enough frames in the video for inference.")
        return None, None

    frames = ensure_video_resolution(frames, target_size=(320, 240))
    rgb_sequence = preprocess_rgb_frames(frames, target_size=(96, 96))
    flow_sequence = compute_optical_flow_sequence_lk(frames, target_size=(240, 320), tau=10)
    
    # Add batch dimension to each stream
    rgb_input = np.expand_dims(rgb_sequence, axis=0)
    flow_input = np.expand_dims(flow_sequence, axis=0)
    
    predictions = model.predict([rgb_input, flow_input])
    pred_class = np.argmax(predictions, axis=1)[0]
    return pred_class, predictions

# ----------------------------
# Video Recording Function with Centered Preview and Optional Pre-warmed Camera
# ----------------------------
def record_video(duration=10, output_path="recorded_video.avi", cap_param=None):
    """
    Records video from the provided camera (cap_param) if available; otherwise, opens a new camera.
    The preview window is centered on the screen.
    """
    # Use the pre-warmed camera if provided; else open a new one.
    if cap_param is not None:
        cap = cap_param
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to access the webcam")
            return None

    # Set the capture resolution (if not already set)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 20.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create and configure the recording window, centering it on the screen.
    cv2.namedWindow("Recording...", cv2.WINDOW_NORMAL)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width - frame_width) / 2)
    y = int((screen_height - frame_height) / 2)
    cv2.moveWindow("Recording...", x, y)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    out.release()
    cv2.destroyAllWindows()
    # Always release the camera after recording.
    cap.release()
    return output_path

# ----------------------------
# Tkinter GUI Functions
# ----------------------------
def record_and_classify():
    """
    Records the video and then processes it for inference.
    This function is run in a separate thread.
    """
    global recording_in_progress, prewarm_cap
    # Use the pre-warmed camera in recording.
    video_path = record_video(duration=10, output_path="recorded_video.avi", cap_param=prewarm_cap)
    # Reset prewarm_cap after recording.
    prewarm_cap = None

    # Signal that recording is complete so the recording countdown stops.
    recording_in_progress = False

    # Update status for inference
    root.after(0, lambda: status_label.config(text="Processing video for inference..."))
    predicted_class, probs = run_inference_on_video(video_path, late_fusion_model, num_frames=30)
    
    if predicted_class is not None:
        predicted_class_name = CLASSES_LIST[predicted_class]
        confidence_percentage = probs[0][predicted_class] * 100  # Convert to percentage
        # If confidence is less than 60%, output as "Unknown"
        if confidence_percentage < 66:
            result_text = f"Predicted Class: Unknown\n"
        else:
            result_text = f"Predicted Class: {predicted_class_name}\nConfidence: {confidence_percentage:.2f}%"
    else:
        result_text = "Error processing video."
    
    root.after(0, lambda: status_label.config(text=result_text))
    root.after(0, lambda: record_button.config(state="normal"))

def pre_record_countdown(t):
    """
    Displays a pre-record countdown (e.g., "Camera starting in X seconds").
    When t reaches 3 seconds, the camera is opened (warmed up).
    When the countdown finishes, recording starts.
    """
    global prewarm_cap, recording_in_progress
    if t >= 0:
        timer_label.config(text=f"Camera starting in {t} seconds")
        # At 3 seconds remaining, open (warm up) the camera.
        if t == 3 and prewarm_cap is None:
            # Open the camera and set resolution.
            prewarm_cap = cv2.VideoCapture(0)
            prewarm_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            prewarm_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        root.after(1000, lambda: pre_record_countdown(t-1))
    else:
        # When pre-record countdown finishes, start recording.
        recording_in_progress = True
        # Start the recording thread.
        threading.Thread(target=record_and_classify, daemon=True).start()
        # Also start the recording countdown (10 seconds).
        record_countdown(10)

def record_countdown(t):
    """
    Updates the timer label every second during recording.
    """
    global recording_in_progress
    if t >= 0 and recording_in_progress:
        timer_label.config(text=f"Recording... {t} seconds remaining")
        root.after(1000, lambda: record_countdown(t-1))
    else:
        timer_label.config(text="Recording complete!")

def on_record():
    """
    Triggered when the 'Record and Classify' button is pressed.
    It initiates a 5-second pre-record countdown.
    """
    record_button.config(state="disabled")
    pre_record_countdown(5)  # Start a 5-second pre-record countdown

# ----------------------------
# Load the Trained Model
# ----------------------------
# Update the model path as needed.
model_path = r"D:\MAPUA\CNN_GRU_usingRGBOF\late_fusion_model_(Working_1_96x96).h5"
try:
    late_fusion_model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    late_fusion_model = None

# ----------------------------
# Tkinter GUI Setup
# ----------------------------
root = tk.Tk()
root.title("Video Classification")
root.attributes("-fullscreen", True)
root.overrideredirect(True)

# Define modern colors and larger fonts.
BACKGROUND_COLOR = "#1e1e1e"       # Dark overall background
TITLE_BG = "#3a3f47"               # Title bar background
TITLE_FG = "white"
CONTENT_BG = "#282c34"             # Main content background
INFO_BG = "#3a3f47"                # Info box (for status and timer)
BUTTON_BG = "#61afef"              # Button background
BUTTON_ACTIVE_BG = "#528dc7"
TEXT_COLOR = "white"

FONT_TITLE = ("Helvetica", 32, "bold")
FONT_CONTENT = ("Helvetica", 28)
FONT_BUTTON = ("Helvetica", 32, "bold")

root.configure(background=BACKGROUND_COLOR)

# Custom Title Bar
title_bar = tk.Frame(root, bg=TITLE_BG, relief="raised", bd=0)
title_bar.pack(side="top", fill="x", pady=(10, 0))

title_label = tk.Label(title_bar, text="Video Classification", bg=TITLE_BG, fg=TITLE_FG, font=FONT_TITLE)
title_label.pack(side="left", padx=20, pady=10)

def minimize_window():
    root.overrideredirect(False)  # Temporarily allow window manager functions
    root.iconify()

def close_window():
    root.destroy()

minimize_button = tk.Button(title_bar, text="_", command=minimize_window, bg=TITLE_BG, fg=TITLE_FG,
                            relief="flat", font=FONT_TITLE, padx=10, pady=5, bd=0)
minimize_button.pack(side="right", padx=10)
close_button = tk.Button(title_bar, text="X", command=close_window, bg=TITLE_BG, fg=TITLE_FG,
                         relief="flat", font=FONT_TITLE, padx=10, pady=5, bd=0)
close_button.pack(side="right", padx=10)

# Allow dragging of the window
def start_move(event):
    root.x = event.x
    root.y = event.y
def on_move(event):
    deltax = event.x - root.x
    deltay = event.y - root.y
    x = root.winfo_x() + deltax
    y = root.winfo_y() + deltay
    root.geometry(f"+{x}+{y}")
title_bar.bind("<Button-1>", start_move)
title_bar.bind("<B1-Motion>", on_move)

# Main Content Frame
content_frame = tk.Frame(root, bg=CONTENT_BG)
content_frame.pack(expand=True, fill="both", padx=40, pady=40)

# Modern-style info box for status and timer
info_frame = tk.Frame(content_frame, bg=INFO_BG, bd=3, relief="ridge")
info_frame.pack(pady=(40, 20), padx=20, fill="x")

status_label = tk.Label(info_frame, text="Status: Idle", bg=INFO_BG, fg=TEXT_COLOR,
                        font=FONT_CONTENT, wraplength=1000, justify="center")
status_label.pack(pady=(20, 10), padx=20)

timer_label = tk.Label(info_frame, text="Timer: --", bg=INFO_BG, fg=TEXT_COLOR, font=FONT_CONTENT)
timer_label.pack(pady=(0, 20), padx=20)

# Button Frame for Centered Buttons
button_frame = tk.Frame(content_frame, bg=CONTENT_BG)
button_frame.pack(expand=True)

record_button = tk.Button(button_frame, text="Record and Classify", command=on_record,
                          font=FONT_BUTTON, bg=BUTTON_BG, fg=TEXT_COLOR, activebackground=BUTTON_ACTIVE_BG,
                          padx=70, pady=50)
record_button.pack(pady=20)

exit_button = tk.Button(button_frame, text="Exit", command=close_window,
                        font=FONT_BUTTON, bg="#d9534f", fg=TEXT_COLOR, activebackground="#c9302c",
                        padx=70, pady=50)
exit_button.pack(pady=20)

root.mainloop()