import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("facialemotionmodel.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features
def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    pred = None
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img, verbose=0)
            pred_index = pred.argmax()
            prediction_label = labels[pred_index]
            confidence = pred[0][pred_index] * 100

            # Display label above the face box
            text = f"{prediction_label} ({confidence:.2f}%)"
            cv2.putText(im, text, (p, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 255), 2)

        # Draw horizontal table only if we have predictions
        if pred is not None:
            height, width, _ = im.shape
            bar_height = 60
            cv2.rectangle(im, (0, height - bar_height), (width, height), (255, 255, 255), -1)

            x = 10
            for idx, label in labels.items():
                acc = pred[0][idx] * 100
                text = f"{label}: {acc:.1f}%"
                cv2.putText(im, text, (x, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
                x += 130  # adjust spacing between items

        cv2.imshow("Output", im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass

# Release resources
webcam.release()
cv2.destroyAllWindows()
