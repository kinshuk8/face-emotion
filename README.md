## 😄 Face Emotion Recognition API

This is a Python-based project using OpenCV and deep learning to detect facial emotions in real-time or from static images. It includes trained models, training notebooks, and a real-time detection script using your webcam.

---

## 🗂 Project Structure

```
FACEEMOTION-API/
│
├── images/                     # Dataset folder
│   ├── train/                  # Training images organized by emotion
│   └── test/                   # Testing images organized by emotion
│
├── emotion_model.h5           # Trained model file
├── emotiondetector.h5/json    # Model + metadata (variant 1)
├── facialemotionmodel.h5/json # Model + metadata (variant 2)
│
├── trainmodel.ipynb           # Jupyter notebook to train the model
├── realtimedetection.py       # Python script to detect emotion via webcam
│
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation (you are here!)
└── venv/                      # Virtual environment (ignored in version control)
```

---

## ⚙️ Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/your-username/FACEEMOTION-API.git
cd FACEEMOTION-API
```

---

### 🐍 2. Create and Activate a Virtual Environment

```bash
# Create
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

---

### 📦 3. Install Requirements

```bash
pip install -r requirements.txt
```

If you're missing a `requirements.txt`, you can manually install the main dependencies:

```bash
pip install opencv-python tensorflow keras numpy matplotlib pillow
```

---

### 📁 4. Prepare the Dataset

Ensure your dataset is structured like this inside the `images/` folder:

```
images/
├── train/
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   └── ...
└── test/
    ├── happy/
    ├── sad/
    ├── angry/
    └── ...
```

Each emotion folder contains face images labeled appropriately.

---

### 🧠 5. Train the Model (Optional)

To retrain or modify the model:

```bash
# Option 1: Use Jupyter Notebook
jupyter notebook trainmodel.ipynb

# Option 2: Convert the notebook to .py if needed and run
# jupyter nbconvert --to script trainmodel.ipynb
```

A new `.h5` model file will be saved when done.

---

### 🎥 6. Run Real-Time Detection

Make sure a model (e.g., `emotion_model.h5` or `emotiondetector.h5`) exists in the root directory.

Then run:

```bash
python realtimedetection.py
```

It will:
- Open your webcam
- Detect faces
- Classify emotions
- Display live feed with predictions

---

## 🎯 Emotion Categories (Example)

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 📌 Tips

- Ensure your webcam is functional before running real-time detection.
- You can switch between models (`emotion_model.h5`, `emotiondetector.h5`, etc.) by updating the loading code in `realtimedetection.py`.

---

## 🙋‍♂️ Need Help?

Feel free to raise an issue or open a pull request if you encounter bugs or want to contribute!

---

Would you like me to generate and upload this `README.md` file for you now?