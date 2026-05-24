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
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
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
MODEL_INPUT_HEIGHT = int(input_details[0]['shape'][1])
MODEL_INPUT_WIDTH = int(input_details[0]['shape'][2])

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

EMPTY_INFO = {
    "emotion": "No Face Detected",
    "stress_value": 0.0,
    "stress_label": "Unknown",
    "emotion_probs": {e.title(): 0.0 for e in EMOTIONS}
}


# =========================
# STRESS CALCULATION FROM EMOTION
# =========================
def img_to_array(img):
    return np.array(img, dtype='float32')


def get_empty_info():
    return {
        "emotion": EMPTY_INFO["emotion"],
        "stress_value": EMPTY_INFO["stress_value"],
        "stress_label": EMPTY_INFO["stress_label"],
        "emotion_probs": EMPTY_INFO["emotion_probs"].copy()
    }


def detect_faces(gray):
    min_face = max(28, int(min(gray.shape[:2]) * 0.12))
    equalized = cv2.equalizeHist(gray)

    with cascade_lock:
        faces = face_cascade.detectMultiScale(
            equalized,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_face, min_face),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(24, 24),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

    return merge_face_boxes(faces)


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def merge_face_boxes(faces):
    if len(faces) == 0:
        return []

    boxes = sorted(
        [tuple(map(int, face)) for face in faces],
        key=lambda box: box[2] * box[3],
        reverse=True
    )
    kept = []
    for box in boxes:
        if all(box_iou(box, existing) < 0.35 for existing in kept):
            kept.append(box)

    return kept


def expand_face_box(face_bb, frame_shape, padding=0.18):
    x, y, w, h = face_bb
    img_h, img_w = frame_shape[:2]

    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def preprocess_face(gray, face_bb, padding=0.16, equalize=False, flip=False):
    x, y, w, h = expand_face_box(face_bb, gray.shape, padding=padding)
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return None, (x, y, w, h), False

    smile_found = False
    if not smile_cascade.empty():
        smiles = smile_cascade.detectMultiScale(
            roi,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(25, 12)
        )
        smile_found = len(smiles) > 0

    if equalize:
        roi = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6)).apply(roi)

    if flip:
        roi = cv2.flip(roi, 1)

    target_size = (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
    interpolation = cv2.INTER_AREA if max(roi.shape[:2]) > max(target_size) else cv2.INTER_CUBIC
    roi = cv2.resize(roi, target_size, interpolation=interpolation)
    roi = roi.astype("float32") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)

    return roi, (x, y, w, h), smile_found


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


def predict_roi(roi):
    with model_lock:
        interpreter.set_tensor(input_details[0]['index'], roi)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]


def emotion_finder(face_bb, gray):
    roi, _, smile_found = preprocess_face(
        gray,
        face_bb,
        padding=0.16,
        equalize=True,
        flip=False
    )
    if roi is None:
        return None

    preds = predict_roi(roi)

    if smile_found:
        preds = preds.copy()
        preds[EMOTIONS.index("happy")] += 0.12
        preds = preds / np.sum(preds)

    label = EMOTIONS[preds.argmax()]
    stress_val, stress_lbl = get_stress_from_emotions(preds)
    probs_dict = {EMOTIONS[i].title(): float(preds[i]) for i in range(len(EMOTIONS))}

    return label, stress_val, stress_lbl, probs_dict, float(np.max(preds))


def info_from_face(face_bb, gray):
    result = emotion_finder(face_bb, gray)
    if result is None:
        return get_empty_info()

    label, stress_val, stress_lbl, probs_dict, confidence = result
    return {
        "emotion": label.title(),
        "stress_value": float(stress_val),
        "stress_label": stress_lbl,
        "emotion_probs": probs_dict,
        "confidence": confidence
    }


# =========================
# STATIC IMAGE PROCESSING (FOR UPLOADS)
# =========================

def process_image_array(frame):
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(gray)
    info = get_empty_info()

    if len(faces) > 0:
        x, y, w, h = faces[0]
        info = info_from_face((x, y, w, h), gray)
        rx, ry, rw, rh = expand_face_box((x, y, w, h), gray.shape)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    else:
        cv2.putText(
            frame,
            "No face detected",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

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
        global latest_stress_info
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
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(gray)

        if len(faces) == 0:
            latest_stress_info = get_empty_info()

        for (x, y, w, h) in faces[:1]:
            latest_stress_info = info_from_face((x, y, w, h), gray)

            rx, ry, rw, rh = expand_face_box((x, y, w, h), gray.shape)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
