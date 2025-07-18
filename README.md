# Emotion Detection in Real-Time using Machine Learning and GUI

##  Abstract

This project presents a desktop application for detecting emotions from facial images using traditional machine learning techniques. Built in Python, it leverages a Support Vector Machine (SVM) model trained on the FER-2013 dataset. Users can either upload static images or use their webcam for real-time emotion recognition. A simple and intuitive GUI is developed using Tkinter.

---

##  Objectives

- Develop a facial emotion classifier using SVM.
- Preprocess facial images (resize, grayscale, flatten).
- Build a user-friendly GUI with Tkinter.
- Integrate real-time emotion recognition via webcam.
- Demonstrate the usefulness of emotion-aware systems in real-life applications.

---

##  Technology Stack

| Area               | Tools Used                     |
|--------------------|-------------------------------|
| Language           | Python                         |
| ML Framework       | Scikit-learn, Joblib           |
| Image Processing   | OpenCV, Pillow (PIL)           |
| GUI Development    | Tkinter                        |
| Dataset            | FER-2013 from Kaggle           |

---

##  Dataset

**FER-2013**: Public dataset of 48x48 grayscale facial images.

- Total images: ~35,000
- Emotion classes: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`
- Preprocessing: Grayscale → Resize to 48x48 → Flatten

[Kaggle Dataset Link](https://www.kaggle.com/datasets/msambare/fer2013)

---

##  Model Training

- **Algorithm**: Support Vector Machine (SVM)
- **Input**: Flattened grayscale images
- **Output**: Emotion label
- **Steps**:
  - Load and preprocess dataset
  - Encode labels
  - Train SVM classifier
  - Save model using `joblib`

---

##  GUI Features

- Upload any image via file dialog
- View image preview and predicted emotion
- Supports `.jpg`, `.png`, etc.
- Display predicted result with image

---

##  Real-Time Emotion Detection (Webcam)

- Capture live frames using OpenCV
- Convert to grayscale and resize
- Predict emotion per frame
- Overlay emotion label on video
- Press `q` to quit webcam stream

---

##  Challenges Faced

- Correct loading of `.pkl` model object
- Maintaining consistent input format for prediction
- Tkinter image reference bugs (preventing image from disappearing)
- Lag in real-time prediction on low-resource devices
- Emotion overlap (e.g., Fear vs Surprise)

---

##  Results

- Successfully detects basic emotions from static and real-time input
- Works efficiently without deep learning
- GUI is lightweight, cross-platform, and user-friendly

---

##  Future Scope

- Replace SVM with CNNs for higher accuracy
- Add multimodal emotion detection (voice, text, gestures)
- Build mobile and web versions
- Detect more complex emotions (boredom, sarcasm, etc.)
- Ensure user privacy using edge AI techniques
- Integrate with smart devices for emotion-aware environments

---

## References

- [OpenCV Docs](https://docs.opencv.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Tkinter Documentation](https://docs.python.org/3/library/tkinter.html)
- [Joblib](https://joblib.readthedocs.io/)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [Pillow Docs](https://pillow.readthedocs.io/)
- Ekman, P. (1992). *An Argument for Basic Emotions*
