from pathlib import Path

content = r"""# PROMPT.md — Fix TensorFlow.js Weight Mismatch Error

## Role

Act as a Senior TensorFlow Engineer, Keras Engineer, TensorFlow.js Engineer, and Machine Learning Deployment Engineer.

Your task is to diagnose and completely fix this exact browser runtime error:

```text
Model loading error: Provided weight data has no target variable: block_10_depthwise/kernel.
Ensure "web/model/model.json" exists and local web server is running.
```

This project is a Rock–Paper–Scissors image classifier using TensorFlow/Keras and TensorFlow.js.

The expected project structure is similar to:

```text
rock-paper-scissors-live/
│
├── train.py
├── convert_to_tfjs.py
├── requirements.txt
│
├── model/
│   ├── rock_paper_scissors.keras
│   └── metadata.json
│
└── web/
    ├── index.html
    ├── styles.css
    ├── app.js
    │
    └── model/
        ├── model.json
        ├── group1-shard*.bin
        └── metadata.json
```

The frontend currently loads:

```javascript
model = await tf.loadLayersModel("./model/model.json");
```

The model is based on a Keras / TensorFlow architecture using MobileNetV2 or an equivalent CNN backbone.

---

# Main Objective

Fix the model loading failure so this succeeds:

```javascript
const model = await tf.loadLayersModel("./model/model.json");
```

and this also succeeds:

```javascript
const output = model.predict(inputTensor);
```

Do not suppress the error.

Do not only change the frontend error message.

Do not replace the model with a random placeholder.

Fix the underlying weight/topology incompatibility.

---

# Important Interpretation of This Error

The error:

```text
Provided weight data has no target variable: block_10_depthwise/kernel
```

strongly suggests that the TensorFlow.js model architecture and the weight data do not correspond to exactly the same model.

Possible causes include:

- stale `.bin` files from a previous conversion
- `model.json` generated from one model while `.bin` files belong to another model
- a changed MobileNetV2 architecture after conversion
- Keras/TensorFlow/TensorFlow.js converter incompatibility
- mixed output from multiple conversion attempts
- layer naming differences
- Keras 3 serialization differences
- a model topology generated without matching target variables
- an incorrectly converted `.keras` model
- a manual edit to `model.json`
- cached browser model assets

Do not assume which cause is correct.

Inspect first.

---

# Rule 1 — Preserve Existing Functionality

Do not redesign the application.

Preserve:

- webcam
- Start Camera button
- Stop Camera button
- prediction display
- probability bars
- Rock / Paper / Scissors classes
- confidence threshold
- inference loop
- responsive UI

Only modify deployment/conversion/loading code as needed.

---

# Step 1 — Inspect the Exact Current Model Files

Inspect:

```text
model/rock_paper_scissors.keras
model/metadata.json

web/model/model.json
web/model/*.bin
web/model/metadata.json
```

Print a directory listing including:

- filename
- file size
- modification date if available

Example:

```text
web/model/
├── model.json                  85 KB
├── group1-shard1of4.bin       ...
├── group1-shard2of4.bin       ...
├── group1-shard3of4.bin       ...
├── group1-shard4of4.bin       ...
└── metadata.json
```

Look for duplicate or old files such as:

```text
group1-shard1of2.bin
group1-shard1of4.bin
weights.bin
model-old.json
```

Stale model files must not remain mixed with a new conversion.

---

# Step 2 — Validate the Original Keras Model First

Before debugging TensorFlow.js, verify the original Python model.

Load:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "model/rock_paper_scissors.keras"
)

model.summary()
```

Then test:

```python
dummy = tf.zeros((1, 224, 224, 3))

output = model(
    dummy,
    training=False
)

print(output.shape)
```

Expected:

```text
(1, 3)
```

If the Keras model itself cannot load or predict correctly, stop and fix that first.

Do not continue conversion from a broken source model.

---

# Step 3 — Inspect Keras Weight Names

Print all source-model weight names.

Use code similar to:

```python
for weight in model.weights:
    print(
        weight.name,
        weight.shape
    )
```

Search specifically for:

```text
block_10_depthwise/kernel
```

Determine whether it exists in the Keras model.

Also inspect MobileNetV2-related variables such as:

```text
block_10_depthwise/kernel
block_10_depthwise_BN/gamma
block_10_depthwise_BN/beta
block_10_depthwise_BN/moving_mean
block_10_depthwise_BN/moving_variance
```

Do not assume weight names.

Record the actual names.

---

# Step 4 — Inspect TensorFlow.js weightsManifest

Open:

```text
web/model/model.json
```

Locate:

```json
"weightsManifest"
```

Inspect the weight entries.

Search for:

```text
block_10_depthwise/kernel
```

Determine whether the manifest contains this weight.

Compare:

```text
Keras model.weights
```

against:

```text
TensorFlow.js weightsManifest weight names
```

The architecture and weight manifest must correspond.

---

# Step 5 — Detect Model Topology / Weight Mismatch

Build a validation script that compares:

```text
source Keras model
vs
TFJS model.json
```

Check:

- number of source model weights
- number of TFJS manifest entries
- missing TFJS variables
- unexpected TFJS variables

Report:

```text
Missing in TFJS:
...

Unexpected in TFJS:
...
```

Specifically verify:

```text
block_10_depthwise/kernel
```

If this variable exists in source Keras but not in TFJS manifest, the conversion is invalid.

If it exists in the manifest but TensorFlow.js says there is no target variable, then the model topology and manifest disagree.

---

# Step 6 — Completely Remove Stale Converted Files

Before each conversion, delete ALL generated TensorFlow.js files from:

```text
web/model/
```

Do not preserve:

```text
model.json
*.bin
```

from previous conversions.

Preserve only source-independent files if necessary.

Recommended logic:

```python
from pathlib import Path

target_dir = Path("web/model")

for file in target_dir.glob("*"):
    if file.is_file():
        file.unlink()
```

Then regenerate the model from scratch.

Do not mix model shards across conversions.

---

# Step 7 — Rebuild Conversion Script Safely

Rewrite:

```text
convert_to_tfjs.py
```

with a reliable sequence:

```text
1. Load the source Keras model.
2. Run a dummy inference.
3. Delete old TFJS output.
4. Convert exactly once.
5. Validate model.json.
6. Validate weightsManifest.
7. Validate every shard exists.
8. Copy metadata.json.
9. Print a success report.
```

Do not allow stale artifacts.

---

# Step 8 — Prefer One Deterministic Conversion Path

Try a single supported TensorFlow.js LayersModel conversion approach first.

Preferred:

```python
import tensorflowjs as tfjs

model = tf.keras.models.load_model(
    "model/rock_paper_scissors.keras"
)

tfjs.converters.save_keras_model(
    model,
    "web/model"
)
```

After conversion, validate the output before using it.

If this path is incompatible with the installed Keras/TensorFlow/TensorFlow.js versions, use a legacy HDF5 bridge.

---

# Step 9 — Safe HDF5 Fallback

If `.keras` to TFJS conversion is producing invalid topology/weights, create:

```text
model/rock_paper_scissors.h5
```

with:

```python
model.save(
    "model/rock_paper_scissors.h5"
)
```

Then convert:

```bash
tensorflowjs_converter \
  --input_format=keras \
  model/rock_paper_scissors.h5 \
  web/model
```

After conversion, validate again.

Do not assume HDF5 is automatically correct.

Test it.

---

# Step 10 — Keras 3 Compatibility Check

Inspect installed versions:

```bash
python --version
pip show tensorflow
pip show keras
pip show tensorflowjs
pip show numpy
```

Identify whether this is a Keras 3 environment.

If the TensorFlow.js converter has compatibility problems with the generated Keras format, use a version combination known to work together.

Do not use incompatible arbitrary latest versions.

If version pinning is needed, update:

```text
requirements.txt
```

with exact tested versions.

Report those versions after validation.

---

# Step 11 — Inspect MobileNetV2 Architecture

Because the error refers to:

```text
block_10_depthwise/kernel
```

inspect the MobileNetV2 backbone.

Check whether it was created like:

```python
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
```

Check whether:

```python
base_model.trainable = False
```

or fine-tuning later altered trainable state.

Trainable state should not change the actual required weight names.

However, confirm the model being converted is the same model that was saved after training.

Do not rebuild a fresh MobileNetV2 architecture separately and then attach weights from a differently serialized model unless exact compatibility is verified.

---

# Step 12 — Do Not Reconstruct Model Manually Unless Necessary

Avoid code such as:

```python
new_model = build_model()
new_model.load_weights(...)
```

unless the architecture is guaranteed identical.

Prefer loading the full trained model:

```python
tf.keras.models.load_model(...)
```

because manual reconstruction may change:

- layer names
- nesting
- weight paths
- variable names

and can lead to errors such as:

```text
Provided weight data has no target variable
```

---

# Step 13 — Check Custom / Nested Layers

Inspect whether the saved model contains:

- Sequential augmentation layers
- nested MobileNetV2 model
- custom preprocessing
- Lambda layers
- custom layers
- custom names

Print:

```python
for layer in model.layers:
    print(
        layer.name,
        type(layer).__name__
    )
```

If the conversion tool cannot serialize a particular layer correctly, simplify only the deployment model.

---

# Step 14 — Create a Clean Deployment Model if Needed

If the training model contains augmentation or layers that should not exist at inference time, create a separate deployment model.

Example concept:

```text
Training Model:
Input
 ↓
Augmentation
 ↓
MobileNetV2
 ↓
Classifier

Deployment Model:
Input
 ↓
MobileNetV2
 ↓
Classifier
```

However, preserve the actual learned weights correctly.

Do not create a structurally different model and blindly reuse incompatible weights.

If augmentation layers are Keras preprocessing layers, determine whether they are active during inference.

Remember:

```python
RandomFlip
RandomRotation
RandomZoom
RandomTranslation
RandomContrast
```

should not perform augmentation when:

```python
training=False
```

They do not necessarily need removal if TFJS supports them, but inspect compatibility.

---

# Step 15 — Consider Moving Preprocessing Outside the Model

If MobileNetV2 preprocessing or augmentation serialization creates TFJS compatibility issues, create a deployment model where preprocessing is performed in JavaScript.

Then the model itself receives already normalized input.

Example deployment pipeline:

```text
browser frame
 ↓
resize 224×224
 ↓
normalize [0,255] -> [-1,1]
 ↓
deployment model
```

The frontend code would use:

```javascript
tensor
    .toFloat()
    .div(127.5)
    .sub(1)
```

Do this only if the saved deployment architecture is intentionally built this way.

Avoid double preprocessing.

---

# Step 16 — Browser Cache Must Be Considered

The browser may cache an older:

```text
model.json
```

or older `.bin` files.

During debugging:

```javascript
fetch("./model/model.json", {
    cache: "no-store"
});
```

If necessary, load with cache-busting during development:

```javascript
const modelUrl =
    `./model/model.json?v=${Date.now()}`;
```

But do not use random cache-busting permanently without reason.

Also test with browser DevTools:

```text
Network
→ Disable cache
→ Reload
```

This error can occur if the browser loads a new `model.json` with old cached weight shards or vice versa.

---

# Step 17 — Verify Weight Shards Over HTTP

Start:

```bash
cd web
python -m http.server 8000
```

Verify:

```text
http://localhost:8000/model/model.json
```

Then verify every `.bin` referenced by `weightsManifest`.

All must return:

```text
HTTP 200
```

No shard may return:

```text
404
```

No `.bin` request may return HTML.

---

# Step 18 — Add Frontend Diagnostics

Improve `web/app.js`.

Before loading the model, log:

```javascript
console.log("TensorFlow.js version:", tf.version.tfjs);
```

Fetch model metadata:

```javascript
const response = await fetch(
    "./model/model.json",
    {
        cache: "no-store"
    }
);
```

Then:

```javascript
const json = await response.json();

console.log({
    format: json.format,
    generatedBy: json.generatedBy,
    convertedBy: json.convertedBy,
    topologyAvailable: Boolean(json.modelTopology),
    weightGroups: json.weightsManifest?.length
});
```

Preserve the original runtime exception:

```javascript
try {
    model = await tf.loadLayersModel(
        "./model/model.json"
    );
} catch (error) {
    console.error(
        "TensorFlow.js model loading failed:",
        error
    );

    throw error;
}
```

Do not replace the original stack trace with a generic string.

---

# Step 19 — Pin Browser TensorFlow.js Version

Do not use:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
```

during a compatibility-sensitive deployment.

Use an explicit tested version:

```html
<script
    src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@X.Y.Z/dist/tf.min.js">
</script>
```

Select `X.Y.Z` based on the tested converter/runtime pair.

Do not invent a version without testing.

---

# Step 20 — Validate Converted Model Programmatically

After conversion, create a validation script or add validation to `convert_to_tfjs.py`.

Check:

```python
import json

with open(
    "web/model/model.json",
    "r",
    encoding="utf-8"
) as f:
    tfjs_json = json.load(f)
```

Validate:

```text
modelTopology
weightsManifest
```

Count all manifest weights.

Print:

```text
TFJS weight count:
TFJS shard count:
```

Also print whether this exists:

```text
block_10_depthwise/kernel
```

If it is missing, conversion must be treated as failed.

---

# Step 21 — Compare Weight Names

Generate:

```text
source_weights.txt
tfjs_weights.txt
```

For debugging only.

Example:

```python
source_weight_names = {
    weight.name
    for weight in model.weights
}
```

From `weightsManifest`, collect:

```python
tfjs_weight_names = {
    weight["name"]
    ...
}
```

Because serialization may format names slightly differently, normalize common suffixes carefully before comparing.

Do not silently ignore mismatches.

Print:

```text
Source-only weights:
...

TFJS-only weights:
...
```

This will directly expose topology/manifest mismatches.

---

# Step 22 — Verify Browser Model Load Before Webcam

Do not start webcam inference until the model passes a browser dummy test.

After model load:

```javascript
console.log(
    model.inputs,
    model.outputs
);
```

Then:

```javascript
const dummy = tf.zeros([
    1,
    224,
    224,
    3
]);

const result = model.predict(dummy);

console.log(
    await result.data()
);

dummy.dispose();
result.dispose();
```

Expected output:

```text
3 values
```

Only after this passes should webcam inference start.

---

# Step 23 — Expected Model Contract

The final model must satisfy:

```text
input:
[null, 224, 224, 3]

output:
[null, 3]
```

Classes:

```text
rock
paper
scissors
```

Class order must come from:

```text
metadata.json
```

and must match training.

---

# Step 24 — Do Not Mix GraphModel and LayersModel

Inspect the converted format.

If it is a TensorFlow.js LayersModel:

```javascript
tf.loadLayersModel(...)
```

If it is a TensorFlow.js GraphModel:

```javascript
tf.loadGraphModel(...)
```

Do not convert one format and load it with the other API.

For this project, prefer:

```text
LayersModel
```

because the source is a standard Keras classifier.

---

# Step 25 — Recommended Final Conversion Flow

The final preferred pipeline should look like:

```text
rock_paper_scissors.keras
        ↓
tf.keras.models.load_model()
        ↓
Python dummy prediction
        ↓
delete web/model/*
        ↓
TensorFlow.js conversion
        ↓
validate model.json
        ↓
validate weightsManifest
        ↓
validate every .bin
        ↓
compare key weight names
        ↓
browser fetch
        ↓
tf.loadLayersModel()
        ↓
browser dummy inference
        ↓
webcam inference
```

---

# Step 26 — Improve Error Messages

The current message:

```text
Ensure "web/model/model.json" exists and local web server is running.
```

is misleading for a weight mismatch.

Replace UI output with something that reflects the real issue.

Example:

```text
The TensorFlow.js model files are inconsistent.
model.json and its weight shards may come from different conversion runs.
Check the browser console for exact details.
```

But preserve the original exception in console:

```text
Provided weight data has no target variable:
block_10_depthwise/kernel
```

---

# Step 27 — Required Verification Checklist

Do not mark this task complete until all of these are checked.

## Python Source Model

- [ ] Keras model exists
- [ ] Keras model loads
- [ ] model.summary() works
- [ ] input shape is correct
- [ ] output shape is `(1, 3)`
- [ ] dummy inference succeeds

## Source Weights

- [ ] source weight names are inspected
- [ ] `block_10_depthwise/kernel` is checked
- [ ] MobileNetV2 weights are present

## Conversion

- [ ] old web model files are deleted
- [ ] one clean conversion is performed
- [ ] `model.json` is newly generated
- [ ] weight shards are newly generated
- [ ] all shards exist
- [ ] weight manifest is valid
- [ ] no stale shards remain

## Weight Compatibility

- [ ] topology and manifest match
- [ ] source and TFJS weight names are compared
- [ ] missing weights are reported
- [ ] unexpected weights are reported
- [ ] `block_10_depthwise/kernel` mismatch is resolved

## Browser

- [ ] TensorFlow.js version is pinned
- [ ] model.json returns HTTP 200
- [ ] all .bin files return HTTP 200
- [ ] browser cache is ruled out
- [ ] tf.loadLayersModel succeeds
- [ ] no weight mismatch error appears
- [ ] dummy browser inference succeeds
- [ ] webcam inference succeeds

---

# Step 28 — Required Final Report

After fixing the project, return:

## Root Cause

State the exact root cause of:

```text
Provided weight data has no target variable:
block_10_depthwise/kernel
```

Do not guess.

Examples of acceptable root causes only if actually verified:

```text
model.json came from conversion A while .bin files came from conversion B
```

or:

```text
browser cache served an old model.json with new weight shards
```

or:

```text
Keras 3 serialization produced an incompatible LayersModel conversion using the installed tensorflowjs version
```

or:

```text
deployment architecture was rebuilt with different layer names
```

---

## Files Changed

List all modified files.

Example:

```text
convert_to_tfjs.py
web/app.js
web/index.html
requirements.txt
README.md
```

---

## Tested Versions

Report:

```text
Python:
TensorFlow:
Keras:
tensorflowjs:
Browser TensorFlow.js:
```

---

## Validation Result

Return:

```text
Keras load: PASS
Python dummy inference: PASS
Clean TFJS conversion: PASS
model.json validation: PASS
weight manifest validation: PASS
weight name validation: PASS
block_10_depthwise/kernel compatibility: PASS
browser fetch: PASS
tf.loadLayersModel: PASS
browser dummy inference: PASS
webcam inference: PASS
```

---

# Critical Final Instruction

Do not treat this as a missing-file problem unless the file is actually missing.

The key error is:

```text
Provided weight data has no target variable:
block_10_depthwise/kernel
```

This is primarily a MODEL TOPOLOGY / WEIGHT COMPATIBILITY problem.

Trace:

```text
Keras architecture
        ↓
Keras weights
        ↓
TFJS modelTopology
        ↓
TFJS weightsManifest
        ↓
.bin weight shards
        ↓
browser cache
        ↓
tf.loadLayersModel()
```

Find the first stage where the model topology and weights diverge.

Fix that stage.

Do not patch around the exception.
"""

path = Path("/mnt/data/prompt_fix_tfjs_weight_mismatch.md")
path.write_text(content, encoding="utf-8")

print(path)
