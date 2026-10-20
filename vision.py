
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained emotion model
emotion_model = load_model("emotion_model.h5")

# Keep labels in the same order as the model's training folders
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


def read_video(video_source=0):
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise RuntimeError("Could not load face detector")

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                # Extract and preprocess the face
                face_crop = gray[y:y+h, x:x+w]
                model_face = cv2.resize(face_crop, (48, 48))
                model_face = model_face.astype("float32") / 255.0
                model_face = np.expand_dims(model_face, axis=-1)
                model_face = np.expand_dims(model_face, axis=0)

                # Predict the emotion
                pred = emotion_model.predict(model_face, verbose=0)
                label = emotion_labels[int(np.argmax(pred[0]))]

                # Draw the box and predicted label
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

                # Display the face crop separately
                cv2.imshow("Sentia - Face Crop", face_crop)

            cv2.imshow("Sentia - Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    read_video()
