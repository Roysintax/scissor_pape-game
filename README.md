# Rock Paper Scissors AI — Live Hand Gesture Recognition

A complete, end-to-end deep learning system and web application for real-time Rock–Paper–Scissors hand gesture recognition. Inference runs entirely in the browser using **TensorFlow.js** and a transfer-learned **MobileNetV2** model trained on the canonical Rock-Paper-Scissors dataset.

---

## Features

- **Dataset**: Laurence Moroney's canonical Rock-Paper-Scissors dataset (`randall-lab/rock-paper-scissors`).
- **Transfer Learning Backbone**: Lightweight `MobileNetV2` with ImageNet pre-trained weights, frozen backbone, global average pooling, dropout, and a 3-class softmax classifier.
- **In-Browser Inference**: Real-time client-side inference using TensorFlow.js with zero backend latency and no server dependency during inference.
- **Privacy First**: All camera frames are processed 100% locally in your browser. No frames or video streams are ever uploaded to any server.
- **Stabilized Predictions**: Sliding-window temporal probability averaging (8 frames) to eliminate visual flicker.
- **Confidence Threshold**: Predictions with confidence below 60% are flagged as `Not Sure` to prevent false positives.
- **Interactive UI / UX**: Modern dark-neutral aesthetic, responsive CSS grid, live probability breakdown bars for Rock, Paper, and Scissors, hand placement HUD overlay, and accessible SVG icons.

---

## Project Structure

```text
rock-paper-scissors-live/
├── prompt.md                # Project specifications and acceptance criteria
├── README.md                # Documentation and setup instructions
├── requirements.txt         # Python dependencies
├── train.py                 # Dataset loading, training, evaluation, and export
├── convert_to_tfjs.py       # Converts Keras model to TensorFlow.js format
├── model/
│   ├── rock_paper_scissors.keras  # Saved Keras model
│   └── classes.json               # Class labels and preprocessing metadata
└── web/
    ├── index.html           # Web application layout
    ├── styles.css           # Modern dark-neutral styling & design system
    ├── app.js               # TensorFlow.js camera stream & inference logic
    └── model/
        ├── model.json       # Converted TF.js model topology
        ├── *.bin            # TF.js model weights
        └── classes.json     # Class metadata
```

---

## Prerequisites & Installation

> **Note**: TensorFlow 2.x and `tensorflowjs` are verified compatible on **Python 3.13**. If you have multiple Python versions installed on Windows, use `py -3.13`.

### 1. Create and Activate Virtual Environment

```bash
# Windows
py -3.13 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training and Conversion Pipeline

### 1. Train the Model

The training script automatically inspects the dataset, applies stratified 80/20 train/validation splitting, data augmentation, trains MobileNetV2 with early stopping, prints classification metrics, and exports `model/rock_paper_scissors.keras`.

```bash
python train.py
```

### 2. Convert to TensorFlow.js

Convert the exported Keras model into browser-compatible JSON and binary shard format:

```bash
python convert_to_tfjs.py
```

This creates `web/model/model.json` and the corresponding weight binary files (`*.bin`).

---

## Running the Web Application

To run the web app, start a local HTTP server inside the `web/` directory:

```bash
cd web
python -m http.server 8000
```

Now open your browser and navigate to:

```text
http://localhost:8000
```

> [!WARNING]
> **Do not open `index.html` directly via `file://`**. Modern browsers restrict webcam access (`getUserMedia`) and fetching local model files (`model.json`) when opened via the `file://` protocol. Always use an HTTP server (e.g. `http://localhost:8000`).

---

## Usage Guide

1. Allow camera permissions when prompted by your browser.
2. Click **Start Camera** to activate the webcam stream.
3. Place your hand inside the centered **"PLACE HAND HERE"** guide box.
4. Form a **Rock** (fist), **Paper** (open palm), or **Scissors** (two fingers extended) gesture.
5. Watch the real-time prediction and probability breakdown meters update live.
6. Click **Stop Camera** to stop the webcam stream and release hardware resources.

---

## Preprocessing Consistency

The training pipeline uses MobileNetV2 normalization:
$$\text{normalized} = \frac{\text{pixel}}{127.5} - 1.0 \in [-1, 1]$$

The browser application in `web/app.js` mirrors this exactly using TensorFlow.js:
```javascript
const tensor = tf.browser.fromPixels(video)
  .resizeBilinear([224, 224])
  .toFloat()
  .div(127.5)
  .sub(1.0)
  .expandDims(0);
```

---

## License

MIT License.

# scissor_pape-game
