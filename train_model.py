import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import time
from sklearn.utils import shuffle

def load_images_from_folder(folder_path):
    data = []
    labels = []
    total = 0
    print(f"Loading data from: {folder_path}")

    for emotion in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        print(f"Reading images for label: {emotion}")
        for file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    img = img / 255.0  # normalize pixel values (optional but recommended)
                    data.append(img.flatten())
                    labels.append(emotion)
                    total += 1
                    if total % 100 == 0:
                        print(f"Loaded {total} images...")
                else:
                    print(f"Skipped unreadable image: {img_path}")
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")

    print(f"Finished loading {total} images from: {folder_path}")
    return np.array(data), np.array(labels)

# Paths to dataset
train_path = r'C:\Users\shilp\Desktop\EmotionDetection\train'
test_path = r'C:\Users\shilp\Desktop\EmotionDetection\test'

start_time = time.time()

try:
    X_train, y_train = load_images_from_folder(train_path)
    X_test, y_test = load_images_from_folder(test_path)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
except Exception as e:
    print("Error during data loading:", e)
    exit()

# Shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Encode string labels into numbers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train model
try:
    print("Training model...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train_encoded)
    print("Training complete.")
except Exception as e:
    print("Error during training:", e)
    exit()

# Evaluate
try:
    print("Evaluating model...")
    predictions = model.predict(X_test)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    print("Classification Report:")
    print(classification_report(y_test, decoded_predictions))
except Exception as e:
    print("Error during evaluation:", e)
    exit()

# Save model and label encoder
try:
    joblib.dump({'model': model, 'label_encoder': label_encoder}, 'emotion_model.pkl')
    print("Model and label encoder saved as emotion_model.pkl")
except Exception as e:
    print("Error saving model:", e)

print(f"Total time taken: {round(time.time() - start_time, 2)} seconds")
