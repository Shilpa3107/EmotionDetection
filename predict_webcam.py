import cv2
import joblib
import numpy as np

# Load the model and label encoder correctly
data = joblib.load('emotion_model.pkl')
model = data['model']
label_encoder = data['label_encoder']

def predict_emotion(gray_img):
    img = cv2.resize(gray_img, (48, 48)).flatten().reshape(1, -1)
    prediction = model.predict(img)
    emotion_label = label_encoder.inverse_transform(prediction)[0]
    return emotion_label

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    emotion = predict_emotion(gray)

    cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
