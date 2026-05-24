from flask import Flask, render_template, Response, jsonify, request
from test import VideoCamera, process_image_array
import test
import cv2
import numpy as np
import base64
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):

    while True:

        try:

            frame = camera.get_frame()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame +
                b'\r\n'
            )

        except Exception as e:

            print("STREAM ERROR:", e)
            break

@app.route('/predict')
def predict():
    return jsonify({'error': 'Live webcam not supported on server'}), 410

@app.route('/status')
def status():
    return jsonify(test.latest_stress_info)


def stress_from_probs(probs):
    weights = {
        "Angry": 1.0,
        "Disgust": 0.8,
        "Scared": 1.0,
        "Happy": 0.0,
        "Sad": 0.8,
        "Surprised": 0.4,
        "Neutral": 0.1
    }
    stress_value = sum(probs.get(emotion, 0.0) * weight for emotion, weight in weights.items())
    if stress_value >= 0.65:
        stress_label = "High Stress"
    elif stress_value >= 0.35:
        stress_label = "Moderate Stress"
    else:
        stress_label = "Low Stress"
    return stress_value, stress_label


def average_infos(infos):
    valid_infos = [
        info for info in infos
        if info.get("emotion_probs")
        and info.get("emotion") != "No Face Detected"
        and sum(info.get("emotion_probs", {}).values()) > 0.01
    ]
    if not valid_infos:
        return infos[-1] if infos else {
            "emotion": "No Face Detected",
            "stress_value": 0.0,
            "stress_label": "Unknown",
            "emotion_probs": {}
        }

    emotions = valid_infos[0]["emotion_probs"].keys()
    weights = np.array([
        max(0.05, float(info.get("confidence", 0.0)))
        for info in valid_infos
    ])
    weights = weights / weights.sum()

    avg_probs = {}
    for emotion in emotions:
        values = np.array([
            info["emotion_probs"].get(emotion, 0.0)
            for info in valid_infos
        ])
        avg_probs[emotion] = float(np.sum(values * weights))

    emotion = max(avg_probs, key=avg_probs.get)
    stress_value, stress_label = stress_from_probs(avg_probs)
    return {
        "emotion": emotion,
        "stress_value": float(stress_value),
        "stress_label": stress_label,
        "emotion_probs": avg_probs,
        "confidence": float(max(info.get("confidence", 0.0) for info in valid_infos))
    }


@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        files = request.files.getlist('images') or request.files.getlist('image')
        if not files:
            return jsonify({'error': 'No image uploaded'})

        processed_img = None
        infos = []
        for file in files:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if img is None:
                continue

            processed_img, info = process_image_array(img)
            infos.append(info)

        if processed_img is None or not infos:
            return jsonify({'error': 'Invalid image file or format'})

        info = average_infos(infos)
        
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': 'data:image/jpeg;base64,' + img_b64,
            'info': info
        })
    except Exception as e:
        import traceback
        print("UPLOAD ERROR TRACEBACK:")
        traceback.print_exc()
        return jsonify({'error': f"Internal Server Error: {str(e)}"})


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
