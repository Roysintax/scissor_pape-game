# PROMPT.md — Rock Paper Scissors Live Gesture Recognition

## Role

Act as a Senior Machine Learning Engineer, Computer Vision Engineer, and Frontend Engineer.

Your task is to build a complete **Rock–Paper–Scissors live hand gesture recognition system** using the Hugging Face dataset:

```python
from datasets import load_dataset

dataset = load_dataset(
    "randall-lab/rock-paper-scissors",
    split="train",
    trust_remote_code=True
)
```

The final application MUST classify these three hand gestures:

- Rock
- Paper
- Scissors

The system MUST support **live webcam inference**. The user should NOT be required to upload an image manually for normal usage.

---

# 1. Main Objective

Build an end-to-end machine learning project with the following pipeline:

```text
Hugging Face Dataset
        ↓
Dataset Inspection
        ↓
Data Cleaning / Validation
        ↓
Train / Validation Split
        ↓
Image Preprocessing
        ↓
CNN / Transfer Learning Model
        ↓
Training
        ↓
Evaluation
        ↓
Export Keras Model
        ↓
Convert Model to TensorFlow.js
        ↓
Browser Webcam
        ↓
Live Preprocessing
        ↓
Real-Time Prediction
        ↓
Rock / Paper / Scissors + Confidence
```

The web application must perform inference locally in the browser using TensorFlow.js.

---

# 2. Project Requirements

Create this project structure:

```text
rock-paper-scissors-live/
│
├── prompt.md
├── README.md
├── requirements.txt
├── train.py
├── convert_to_tfjs.py
├── model/
│   └── rock_paper_scissors.keras
│
└── web/
    ├── index.html
    ├── styles.css
    ├── app.js
    └── model/
        ├── model.json
        └── *.bin
```

Do not require a backend for live inference unless absolutely necessary.

---

# 3. Dataset Inspection

Load:

```python
dataset = load_dataset(
    "randall-lab/rock-paper-scissors",
    split="train",
    trust_remote_code=True
)
```

Inspect:

- number of samples
- feature names
- image format
- image dimensions
- label feature
- label names
- class distribution
- missing or invalid images
- duplicated records if practical

Never assume label order. Retrieve it from:

```python
dataset.features["label"].names
```

Store the class order used during training because the website must use exactly the same order.

---

# 4. Train / Validation Split

If the source dataset has no suitable validation split, create one reproducibly:

```python
dataset = dataset.train_test_split(
    test_size=0.20,
    seed=42,
    stratify_by_column="label"
)
```

Use:

- 80% training
- 20% validation

Preserve class balance.

---

# 5. Image Preprocessing

Use a practical image size such as:

```text
224 × 224 × 3
```

Required preprocessing:

1. Convert image to RGB.
2. Resize to 224 × 224.
3. Convert to float32.
4. Normalize consistently with the selected model.
5. Use the exact same preprocessing at browser inference time.

Recommended augmentation for training only:

- small rotation
- small zoom
- horizontal flip when appropriate
- slight contrast variation
- slight translation

Do not use augmentation during validation or inference.

---

# 6. Model

Prefer a lightweight model suitable for real-time browser inference.

Recommended first choice:

```text
MobileNetV2
```

Use transfer learning:

```text
Input
 ↓
Data Augmentation
 ↓
MobileNetV2 backbone
 ↓
GlobalAveragePooling2D
 ↓
Dropout
 ↓
Dense(3, softmax)
```

Start with ImageNet weights.

Initially freeze the backbone.

Optionally perform a short fine-tuning stage after the classifier head stabilizes.

Do not build an unnecessarily large network.

---

# 7. Training

Use:

```python
optimizer = Adam(...)
loss = SparseCategoricalCrossentropy()
metrics = ["accuracy"]
```

Include:

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

Example target:

```text
epochs: 10–20
batch_size: 16 or 32
```

The script must automatically use GPU if TensorFlow detects one, while remaining compatible with CPU.

---

# 8. Evaluation

Report at minimum:

- training accuracy
- validation accuracy
- training loss
- validation loss
- confusion matrix
- classification report
- per-class precision
- per-class recall
- per-class F1-score

Also test several individual samples and print:

```text
Actual: paper
Predicted: paper
Confidence: 98.42%
```

---

# 9. Export Model

Save the trained model:

```python
model.save("model/rock_paper_scissors.keras")
```

Also save class metadata:

```json
{
  "input_size": 224,
  "classes": [
    "rock",
    "paper",
    "scissors"
  ]
}
```

The real class list must come from the dataset, not from a guessed order.

---

# 10. TensorFlow.js Conversion

Create:

```text
convert_to_tfjs.py
```

Convert:

```text
model/rock_paper_scissors.keras
```

into:

```text
web/model/model.json
web/model/*.bin
```

Use:

```bash
tensorflowjs_converter
```

or the supported Python TensorFlow.js converter.

Ensure the generated model can be loaded with:

```javascript
tf.loadLayersModel("./model/model.json")
```

---

# 11. Website Requirements

Create a modern responsive website.

The page must contain:

## Header

```text
Rock Paper Scissors AI
Live Hand Gesture Recognition
```

## Camera Card

Display:

- live webcam feed
- prediction overlay
- camera status
- Start Camera button
- Stop Camera button

## Prediction Card

Show:

```text
Detected Gesture

PAPER

Confidence
98.42%
```

## Probability Bars

Show all three probabilities:

```text
Rock       1.20%
Paper     98.42%
Scissors   0.38%
```

## Status Information

Show:

```text
Model: Ready
Camera: Active
Inference: Running
```

---

# 12. Webcam

Use:

```javascript
navigator.mediaDevices.getUserMedia({
    video: {
        facingMode: "user"
    },
    audio: false
})
```

The browser must ask for camera permission.

Do not capture audio.

Implement:

- start camera
- stop camera
- camera permission denied handling
- no camera detected handling
- model loading errors
- unsupported browser handling

---

# 13. Live Inference

Use TensorFlow.js.

Recommended prediction loop:

```text
requestAnimationFrame()
```

Do NOT run prediction unnecessarily at extremely high frequency.

Target approximately:

```text
8–15 inference predictions / second
```

while keeping the video visually smooth.

Pipeline:

```text
video frame
   ↓
tf.browser.fromPixels(video)
   ↓
resizeBilinear([224, 224])
   ↓
normalization
   ↓
expandDims(0)
   ↓
model.predict()
   ↓
softmax probabilities
   ↓
argMax
   ↓
display prediction
```

Use:

```javascript
tf.tidy(...)
```

where appropriate to avoid browser memory leaks.

Dispose tensors correctly.

---

# 14. IMPORTANT: Preprocessing Consistency

Browser preprocessing MUST match training preprocessing.

For a model trained with pixel values in:

```text
[-1, 1]
```

for MobileNetV2, use equivalent preprocessing in JavaScript.

For example:

```javascript
tensor.div(127.5).sub(1)
```

Do not blindly use `/255` if training used MobileNetV2 preprocessing.

---

# 15. Gesture Stabilization

Live predictions often flicker.

Implement a small prediction history.

Example:

```text
Last 5 predictions:
paper
paper
rock
paper
paper
```

Result:

```text
PAPER
```

Use:

- moving average probabilities
or
- majority vote

Recommended:

```text
5–10 recent predictions
```

Add minimum confidence threshold:

```text
0.60
```

If confidence is lower:

```text
Not Sure
```

---

# 16. Camera Region / Guidance

Show a visible guide area in the camera:

```text
┌────────────────────────────┐
│                            │
│       PLACE HAND HERE      │
│                            │
│            ✋              │
│                            │
└────────────────────────────┘
```

The user should place one hand inside the center area.

For the initial version, classifying the centered camera crop is acceptable.

An optional later enhancement may use MediaPipe hand detection before classification.

---

# 17. UI / UX

Design requirements:

- modern
- clean
- dark/light-neutral appearance
- responsive
- desktop
- tablet
- smartphone

Use CSS only.

Do not require React unless there is a strong reason.

Recommended visual layout:

```text
------------------------------------------------
 Rock Paper Scissors AI
 Live Hand Gesture Recognition
------------------------------------------------

┌──────────────────────────┐ ┌─────────────────┐
│                          │ │ Detected Gesture │
│       LIVE CAMERA        │ │                 │
│                          │ │      PAPER      │
│          ✋              │ │      98.4%      │
│                          │ │                 │
└──────────────────────────┘ └─────────────────┘

Rock       █░░░░░░░░░  1.2%
Paper      ██████████  98.4%
Scissors   ░░░░░░░░░░  0.4%

[ Start Camera ] [ Stop Camera ]
```

---

# 18. Performance

The implementation must:

- avoid excessive tensor allocations
- dispose tensors
- avoid creating multiple camera streams
- stop all MediaStream tracks when camera stops
- avoid duplicated prediction loops
- use requestAnimationFrame
- throttle inference

The application should be usable on ordinary laptops.

---

# 19. Security / Privacy

Display a short privacy notice:

```text
Camera frames are processed locally in your browser.
No camera image is uploaded to a server.
```

Do not record video.

Do not save camera frames.

Do not send frames to an external API.

---

# 20. README

Create a complete `README.md` containing:

## Project Description

Explain what the application does.

## Features

- Hugging Face dataset
- CNN / MobileNetV2
- TensorFlow
- TensorFlow.js
- live webcam inference
- confidence score
- real-time probability display

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py
```

Convert:

```bash
python convert_to_tfjs.py
```

Run web server:

```bash
cd web
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Warn the user NOT to open `index.html` directly via `file://` because browser webcam/model loading restrictions may apply.

---

# 21. Code Quality

All source code must be:

- readable
- modular
- documented
- logically structured
- free from dead code
- free from duplicated logic

Use descriptive function names.

Add comments for important machine learning operations.

---

# 22. Acceptance Criteria

The project is considered complete only when:

1. Dataset loads successfully.
2. Model trains successfully.
3. Validation metrics are printed.
4. Keras model is exported.
5. TensorFlow.js model is generated.
6. Website loads the model.
7. Browser requests webcam permission.
8. Webcam video is visible.
9. User can show Rock, Paper, or Scissors.
10. Prediction updates automatically.
11. Confidence is displayed.
12. All three probability values are displayed.
13. Camera can be stopped.
14. Camera resources are released.
15. Low-confidence predictions show `Not Sure`.
16. Website works without image upload.
17. No webcam image is sent to a server.

---

# 23. Optional Phase 2

After the base system works, optionally add MediaPipe Hands:

```text
Webcam
 ↓
MediaPipe Hand Detection
 ↓
Hand Bounding Box
 ↓
Crop Hand
 ↓
CNN
 ↓
Rock / Paper / Scissors
```

Do not implement this optional phase until the base webcam classifier is stable.

---

# Final Instruction

Build the solution incrementally and verify each stage before continuing.

Do not alter the primary requirement:

> The final application must perform Rock–Paper–Scissors classification from a LIVE WEBCAM feed without requiring the user to upload an image.
