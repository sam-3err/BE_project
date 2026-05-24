import numpy as np
import cv2
import imutils
import os
import threading

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import tflite_runtime.interpreter as tflite

# =========================
# LOAD FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================
# LOAD TFLITE MODEL
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tflite_path = os.path.join(BASE_DIR, "fer.tflite")
print("Loading tflite model from:", tflite_path)

interpreter = tflite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# GLOBAL VARIABLES
# =========================

model_lock = threading.Lock()
cascade_lock = threading.Lock()

latest_stress_info = {
    "emotion": "Analyzing...",
    "stress_value": 0.0,
    "stress_label": "System Active",
    "emotion_probs": {}
}

EMOTIONS = [
    "angry",
    "disgust",
    "scared",
    "happy",
    "sad",
    "surprised",
    "neutral"
]


# =========================
# STRESS CALCULATION FROM EMOTION
# =========================
def img_to_array(img):
    return np.array(img, dtype='float32')
def get_stress_from_emotions(preds):
    weights = np.array([1.0, 0.8, 1.0, 0.0, 0.8, 0.4, 0.1])
    stress_value = np.sum(preds * weights)

    if stress_value >= 0.65:
        stress_label = "High Stress"
    elif stress_value >= 0.35:
        stress_label = "Moderate Stress"
    else:
        stress_label = "Low Stress"

    return stress_value, stress_label


def crop_face(frame, face_bb):
    x, y, w, h = face_bb
    img_h, img_w = frame.shape[:2]

    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    pad_x = int(w * 0.12)
    pad_top = int(h * 0.18)
    pad_bottom = int(h * 0.12)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_top)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_bottom)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        roi = frame

    return cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)


def emotion_finder(face_bb, frame):
    roi = crop_face(frame, face_bb)
    roi = roi.astype("float32") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)

    with model_lock:
        interpreter.set_tensor(input_details[0]['index'], roi)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

    label = EMOTIONS[preds.argmax()]
    stress_val, stress_lbl = get_stress_from_emotions(preds)
    probs_dict = {EMOTIONS[i].title(): float(preds[i]) for i in range(len(EMOTIONS))}

    return label, stress_val, stress_lbl, probs_dict


# =========================
# STATIC IMAGE PROCESSING (FOR UPLOADS)
# =========================

def process_image_array(frame):
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    with cascade_lock:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

    info = {
        "emotion": "No Face Detected",
        "stress_value": 0.0,
        "stress_label": "Unknown",
        "emotion_probs": {e.title(): 0.0 for e in EMOTIONS}
    }

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        label, stress_val, stress_lbl, probs_dict = emotion_finder((x, y, w, h), gray)
        info = {
            "emotion": label.title(),
            "stress_value": float(stress_val),
            "stress_label": stress_lbl,
            "emotion_probs": probs_dict
        }
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        h, w = gray.shape
        label, stress_val, stress_lbl, probs_dict = emotion_finder((0, 0, w, h), gray)
        info = {
            "emotion": label.title(),
            "stress_value": float(stress_val),
            "stress_label": stress_lbl,
            "emotion_probs": probs_dict
        }
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 255), 2)

    return frame, info


# =========================
# VIDEO CAMERA CLASS
# =========================

class VideoCamera(object):

    def __init__(self, camera_index=0):
        print("Opening camera...")
        self.video = cv2.VideoCapture(camera_index)
        if not self.video.isOpened():
            print("ERROR: Camera could not be opened")

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        if not ret or frame is None:
            blank = np.zeros((500, 500, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Camera not available",
                (60, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
            _, jpeg = cv2.imencode('.jpg', blank)
            return jpeg.tobytes()

        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with cascade_lock:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            label, stress_val, stress_lbl, probs_dict = emotion_finder((x, y, w, h), gray)

            global latest_stress_info
            latest_stress_info = {
                "emotion": label.title(),
                "stress_value": float(stress_val),
                "stress_label": stress_lbl,
                "emotion_probs": probs_dict
            }

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
