import cv2
import dlib
from utils import eye_aspect_ratio, play_alarm
from imutils import face_utils
import numpy as np

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def find_working_camera(max_index=5):
    """Find the first working camera index on Windows."""
    for i in range(max_index):
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Use MSMF backend for Windows
        if cap.isOpened():
            cap.release()
            print(f"✅ Using camera index {i}")
            return i
        cap.release()
    print("❌ No working camera found")
    return None

def ensure_bgr_uint8(frame):
    """Convert any webcam frame to 8-bit 3-channel BGR."""
    if frame is None:
        return None

    # Convert to numpy array
    frame = np.array(frame)

    # Convert to 8-bit if needed
    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)

    # Handle channel formats
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:  # BGRA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] != 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame

def start_detection():
    print("Loading face detector and predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

    camera_index = find_working_camera()
    if camera_index is None:
        return

    print("Starting video stream...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Failed to grab frame, skipping...")
            continue

        # Resize frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Convert to proper 8-bit BGR
        frame = ensure_bgr_uint8(frame)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        try:
            rects = detector(gray, 0)
        except Exception as e:
            print(f"⚠️ Error detecting faces: {e}")
            continue

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[42:48]
            rightEye = shape[36:42]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                counter += 1
                if counter >= EYE_AR_CONSEC_FRAMES:
                    play_alarm()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                counter = 0

            cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
