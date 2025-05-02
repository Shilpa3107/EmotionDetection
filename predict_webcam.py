import cv2
import joblib
import numpy as np

model = joblib.load('emotion_model.pkl')

def predict_emotion(gray_img):
    img = cv2.resize(gray_img, (48, 48)).flatten().reshape(1, -1)
    prediction = model.predict(img)
    return prediction[0]

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
