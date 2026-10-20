
import cv2
import numpy as np
from flask import Flask, Response, render_template
from tensorflow.keras.models import load_model
from collections import Counter
from threading import Lock
from flask import jsonify

app = Flask(__name__)

#Load the trained emotion model
emotion_model = load_model("emotion_model.h5")

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]
stats_lock = Lock()

emotion_counts = Counter({
    label: 0 for label in emotion_labels
})

faces_current = 0
total_predictions = 0

#Load OpenCV's pretrained face detector
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)

#Open the webcam
camera = cv2.VideoCapture(0)


def generate_frames():
    global faces_current, total_predictions
    while True:
        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        frame_emotions = []

        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]

            model_face = cv2.resize(face_crop, (48, 48))
            model_face = model_face.astype("float32") / 255.0
            model_face = np.expand_dims(model_face, axis=-1)
            model_face = np.expand_dims(model_face, axis=0)

            pred = emotion_model.predict(model_face, verbose=0)
            label = emotion_labels[int(np.argmax(pred[0]))]
            frame_emotions.append(label)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        with stats_lock:
            faces_current = len(faces)

            for label in frame_emotions:
                emotion_counts[label] += 1
                total_predictions += 1

        # Encode the processed frame as JPEG for the browser
        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/stats")
def stats():
    with stats_lock:
        most_common = (
            emotion_counts.most_common(1)[0][0]
            if total_predictions > 0
            else "No predictions yet"
        )

        return jsonify({
            "faces_current": faces_current,
            "emotion_counts": dict(emotion_counts),
            "total_predictions": total_predictions,
            "most_common_emotion": most_common
        })



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
