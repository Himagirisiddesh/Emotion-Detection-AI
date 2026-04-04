# EmotionLens Pro

EmotionLens Pro is a multimodal stress detection system built with Python, TensorFlow/Keras, OpenCV, Flask, HTML, and JavaScript. It combines a facial emotion model with a binary text-based stress classifier so the app can estimate stress from uploaded images, browser webcam frames, and typed text in one clean workflow.

## Features

- Facial emotion detection with OpenCV Haarcascade face detection
- Browser webcam streaming with `getUserMedia`
- Text-based stress detection trained from `Stress.csv`
- Multimodal stress fusion from face score and text score
- Live in-browser overlays for face emotion and confidence
- Clean Flask UI for upload + webcam + text analysis
- Saved model artifacts for both face and text pipelines

## Project Structure

```text
ML/
|-- app.py
|-- emotion_utils.py
|-- stress_text_utils.py
|-- stress_history.py
|-- evaluate.py
|-- train_model.py
|-- train_text_model.py
|-- bio_model_training.py
|-- predict.py
|-- webcam.py
|-- requirements.txt
|-- README.md
|-- model/
|   |-- emotion_metadata.json
|   |-- emotion_model.h5
|   |-- text_stress_metadata.json
|   `-- text_stress_model.pkl
|-- static/
|   |-- app.js
|   |-- style.css
|   |-- processed/
|   `-- uploads/
`-- templates/
    `-- index.html
```

## Main Files

- `app.py`: Flask routes for upload prediction, browser webcam frames, and text-only prediction.
- `emotion_utils.py`: Face model loading, face detection, preprocessing, emotion prediction, annotation, and webcam smoothing.
- `stress_text_utils.py`: Text cleaning, binary text fusion, and Confidence-Weighted multi-modal stress score fusion.
- `stress_history.py`: Temporal state tracking and trend evaluation.
- `evaluate.py`: Multi-modality dataset benchmarking utility (`python evaluate.py --test-csv`).
- `train_model.py`: CNN training script for facial emotion detection.
- `bio_model_training.py`: Physiological pipeline estimator builder.
- `train_text_model.py`: `Stress.csv` training script for TF-IDF/CountVectorizer-based binary text stress detection.
- `templates/index.html`: Main multimodal UI.
- `static/app.js`: Browser webcam capture, live frame upload, and live result rendering.
- `static/style.css`: Modern UI styling.

## Install

```bash
pip install -r requirements.txt
```

## Train the Text Model

```bash
python train_text_model.py
```

Default dataset:

```text
C:\Users\chand\OneDrive\Desktop\stress\Stress.csv
```

Expected columns:

- `text`
- `label` where `0 = no stress` and `1 = stress`

Outputs:

- `model/text_stress_model.pkl`
- `model/text_stress_metadata.json`

## Train the Face Model

```bash
python train_model.py
```

## Run the App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Demo Flow

1. Start the Flask app with `python app.py`
2. Upload an image to review face emotion
3. Use the standalone text section or live webcam text box for text stress detection
4. Start the browser camera
5. Type live text while webcam detection runs
6. Watch the multimodal result update inside the page

## Evaluate Models

To test all the modalities (Face, Text, Bio, Ensemble) and generate performance analytics against true labels:
```bash
python evaluate.py --test-csv your_test_data.csv
```

## Stress Mapping

Face scores:

- Happy -> 0
- Neutral -> 0
- Sad -> 1
- Surprise -> 1
- Angry -> 2
- Fear -> 2

Text scores:

- No Stress -> 0
- Stress -> 2

Bio scores:

- Not Stressed -> 0
- Stressed -> 2

Final stress score:

- Confidence-weighted fusion replaces simple average.
- Each modality's score is weighted by its prediction confidence.
- Falls back to single-modality scoring or simple average if weights are zero.

Stress levels:

- Low: score < 1.0
- Medium: 1.0 to < 2.0
- High: 2.0 and above
