import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================
EAR_THRESHOLD = 0.25
DROWSY_TIME = 2.0
CONF_THRESHOLD = 0.4

# =====================================================
# LOAD MODELS
# =====================================================
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

yolo = YOLO("yolov8n.pt")

# =====================================================
# LANDMARKS
# =====================================================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

# =====================================================
# FUNCTIONS
# =====================================================
def eye_aspect_ratio(lm, eye_ids, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_ids]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def detect_seatbelt(frame):
    """
    Heuristic seat belt detection:
    looks for strong diagonal line on chest area
    """
    h, w, _ = frame.shape
    roi = frame[int(h*0.35):int(h*0.75), int(w*0.3):int(w*0.7)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=80,
        minLineLength=80,
        maxLineGap=10
    )

    if lines is None:
        return False

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 30 < angle < 60:  # diagonal like seat belt
            return True

    return False

# =====================================================
# CAMERA
# =====================================================
cap = cv2.VideoCapture(0)
eye_close_start = None

print("Driver Monitoring Started")
print("Press ENTER to exit")

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    eye_status = "UNKNOWN"
    attention = "UNKNOWN"
    drowsy = False
    phone = False

    # ---------------- FACE ----------------
    result = face_mesh.process(rgb)
    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        ear = (
            eye_aspect_ratio(lm, LEFT_EYE, w, h) +
            eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ) / 2

        if ear < EAR_THRESHOLD:
            eye_status = "EYE CLOSED"
            if eye_close_start is None:
                eye_close_start = time.time()
            elif time.time() - eye_close_start > DROWSY_TIME:
                drowsy = True
        else:
            eye_status = "EYE OPEN"
            eye_close_start = None

        nose_x = int(lm[NOSE_TIP].x * w)
        attention = "LOOKING AWAY" if (nose_x < w*0.35 or nose_x > w*0.65) else "LOOKING FORWARD"

    # ---------------- YOLO ----------------
    results = yolo(frame, verbose=False)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        label = yolo.names[cls]
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        if label == "cell phone":
            phone = True

    # ---------------- SEAT BELT ----------------
    seatbelt = detect_seatbelt(frame)

    # ---------------- UI ----------------
    y = 30
    cv2.putText(frame, f"Eye: {eye_status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2); y+=35
    cv2.putText(frame, f"Attention: {attention}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2); y+=35

    if seatbelt:
        cv2.putText(frame, "Seat Belt: WORN", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    else:
        cv2.putText(frame, "Seat Belt: NOT WORN", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    y+=35

    if phone:
        cv2.putText(frame, "Phone Usage: YES", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2); y+=35

    if drowsy:
        cv2.putText(frame, "DROWSINESS ALERT!", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
