import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

def load_images_from_folder(folder_path):
    data = []
    labels = []
    for emotion in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                data.append(img.flatten())
                labels.append(emotion)
    return np.array(data), np.array(labels)

# Paths to dataset
train_path = r'C:\Users\shilp\Desktop\EmotionDetection\train'
test_path = r'C:\Users\shilp\Desktop\EmotionDetection\test'

# Load and train
X_train, y_train = load_images_from_folder(train_path)
X_test, y_test = load_images_from_folder(test_path)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, 'emotion_model.pkl')
print("Model saved as emotion_model.pkl")
