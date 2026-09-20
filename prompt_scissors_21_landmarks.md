# PROMPT.md — SCISSORS Gesture Recognition Using 21 Hand Landmarks

## Role

Act as a Senior Computer Vision Engineer specializing in:

- hand pose estimation,
- MediaPipe Hand Landmarker,
- geometric gesture recognition,
- real-time webcam inference,
- robust landmark-based classification.

Your task is to implement a stable **SCISSORS gesture detector** using the standard **21 hand landmarks**.

The target posture is:

```text
INDEX   = extended
MIDDLE  = extended
RING    = folded
PINKY   = folded
THUMB   = tolerant / supporting
```

The system must recognize SCISSORS from live webcam input without requiring image upload.

Do not classify SCISSORS from raw RGB image appearance alone.

The primary signal must come from landmark geometry.

---

# 1. Target Pipeline

Implement:

```text
Webcam
   ↓
Hand Detection
   ↓
21 MediaPipe Landmarks
   ↓
Landmark Normalization
   ↓
Joint Angle Analysis
   ↓
Finger Open / Folded States
   ↓
Index-Middle V Shape Analysis
   ↓
SCISSORS Geometric Score
   ↓
Temporal Smoothing
   ↓
SCISSORS / UNKNOWN
```

---

# 2. Standard 21 MediaPipe Landmarks

Use exactly:

```text
0  = WRIST

THUMB
1  = THUMB_CMC
2  = THUMB_MCP
3  = THUMB_IP
4  = THUMB_TIP

INDEX
5  = INDEX_MCP
6  = INDEX_PIP
7  = INDEX_DIP
8  = INDEX_TIP

MIDDLE
9  = MIDDLE_MCP
10 = MIDDLE_PIP
11 = MIDDLE_DIP
12 = MIDDLE_TIP

RING
13 = RING_MCP
14 = RING_PIP
15 = RING_DIP
16 = RING_TIP

PINKY
17 = PINKY_MCP
18 = PINKY_PIP
19 = PINKY_DIP
20 = PINKY_TIP
```

---

# 3. Skeleton Connections

Use:

```python
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (0, 9), (9, 10), (10, 11), (11, 12),

    (0, 13), (13, 14), (14, 15), (15, 16),

    (0, 17), (17, 18), (18, 19), (19, 20),

    (5, 9),
    (9, 13),
    (13, 17)
]
```

Draw all 21 points and connections on the webcam.

Also show landmark numbers `0–20` in debug mode.

---

# 4. Expected SCISSORS Landmark Pattern

Conceptually:

```text
       8                12
       ●                ●
       │               /
       7●             ●11
       │             /
       6●           ●10
       │           /
       5●─────────●9
         \         \
          \         ●13
           \       / \
            \    ●14  ●17
             \   │    │
              \ ●15  ●18
               \ │    │
                ●16  ●19
                  \    │
                   \  ●20

                     ●0
```

The exact drawing can vary based on hand rotation.

The geometric condition must remain:

```text
Index  = open
Middle = open
Ring   = closed
Pinky  = closed
```

---

# 5. Critical Classification Rule

The strongest SCISSORS condition is:

```python
scissors_core = (
    index_extended
    and middle_extended
    and ring_folded
    and pinky_folded
)
```

Do not let thumb posture determine the whole class.

Thumb position varies naturally.

---

# 6. Landmark Normalization

Do not use raw pixel coordinates as the primary feature.

Translate all landmarks relative to:

```text
WRIST = landmark 0
```

Example:

```python
relative = landmark - wrist
```

Normalize scale using:

```python
palm_size = distance_3d(
    landmarks[0],
    landmarks[9]
)
```

Alternative reference:

```python
palm_width = distance_3d(
    landmarks[5],
    landmarks[17]
)
```

Then:

```python
normalized = relative / palm_size
```

This prevents camera distance and hand size from changing the gesture logic too much.

---

# 7. Palm Center

Calculate:

```python
palm_center = mean([
    landmarks[0],
    landmarks[5],
    landmarks[9],
    landmarks[13],
    landmarks[17]
])
```

Use palm center for fingertip distances.

---

# 8. Required Geometry Functions

Implement:

```python
distance_2d(a, b)
distance_3d(a, b)
angle_3points(a, b, c)
normalize_landmarks(landmarks)
calculate_palm_center(landmarks)
```

For:

```text
A → B → C
```

calculate the angle at:

```text
B
```

Formula:

```python
BA = A - B
BC = C - B

cosine = dot(BA, BC) / (
    norm(BA) * norm(BC)
)

cosine = clip(
    cosine,
    -1.0,
    1.0
)

angle = degrees(
    arccos(cosine)
)
```

---

# 9. Detect INDEX as Extended

Index landmarks:

```text
5 → 6 → 7 → 8
```

Calculate:

```python
index_pip_angle = angle_3points(
    landmarks[5],
    landmarks[6],
    landmarks[7]
)

index_dip_angle = angle_3points(
    landmarks[6],
    landmarks[7],
    landmarks[8]
)
```

Initial rule:

```python
index_extended = (
    index_pip_angle >= PIP_EXTENDED_MIN
    and
    index_dip_angle >= DIP_EXTENDED_MIN
)
```

Recommended starting thresholds:

```python
PIP_EXTENDED_MIN = 155.0
DIP_EXTENDED_MIN = 150.0
```

---

# 10. Detect MIDDLE as Extended

Middle landmarks:

```text
9 → 10 → 11 → 12
```

Calculate:

```python
middle_pip_angle = angle_3points(
    landmarks[9],
    landmarks[10],
    landmarks[11]
)

middle_dip_angle = angle_3points(
    landmarks[10],
    landmarks[11],
    landmarks[12]
)
```

Initial rule:

```python
middle_extended = (
    middle_pip_angle >= PIP_EXTENDED_MIN
    and
    middle_dip_angle >= DIP_EXTENDED_MIN
)
```

---

# 11. Detect RING as Folded

Ring landmarks:

```text
13 → 14 → 15 → 16
```

Calculate:

```python
ring_pip_angle = angle_3points(
    landmarks[13],
    landmarks[14],
    landmarks[15]
)

ring_dip_angle = angle_3points(
    landmarks[14],
    landmarks[15],
    landmarks[16]
)
```

Initial rule:

```python
ring_folded = (
    ring_pip_angle <= PIP_FOLDED_MAX
    or
    ring_dip_angle <= DIP_FOLDED_MAX
)
```

Suggested initial values:

```python
PIP_FOLDED_MAX = 145.0
DIP_FOLDED_MAX = 150.0
```

Do not use angle alone.

Also use fingertip-to-palm distance.

---

# 12. Detect PINKY as Folded

Pinky landmarks:

```text
17 → 18 → 19 → 20
```

Calculate:

```python
pinky_pip_angle = angle_3points(
    landmarks[17],
    landmarks[18],
    landmarks[19]
)

pinky_dip_angle = angle_3points(
    landmarks[18],
    landmarks[19],
    landmarks[20]
)
```

Initial:

```python
pinky_folded = (
    pinky_pip_angle <= PIP_FOLDED_MAX
    or
    pinky_dip_angle <= DIP_FOLDED_MAX
)
```

Also validate using palm-relative distances.

---

# 13. Fingertip-to-Palm Distance

Calculate:

```python
index_tip_palm = distance_3d(
    landmarks[8],
    palm_center
) / palm_size

middle_tip_palm = distance_3d(
    landmarks[12],
    palm_center
) / palm_size

ring_tip_palm = distance_3d(
    landmarks[16],
    palm_center
) / palm_size

pinky_tip_palm = distance_3d(
    landmarks[20],
    palm_center
) / palm_size
```

Expected SCISSORS pattern:

```text
Index tip   = FAR from palm
Middle tip  = FAR from palm

Ring tip    = NEAR palm
Pinky tip   = NEAR palm
```

Recommended initial thresholds:

```python
EXTENDED_TIP_DISTANCE_MIN = 0.95
FOLDED_TIP_DISTANCE_MAX = 0.85
```

Tune these empirically.

---

# 14. Combine Angle + Distance

A stronger extended-finger rule:

```python
index_extended = (
    index_pip_angle >= PIP_EXTENDED_MIN
    and
    index_dip_angle >= DIP_EXTENDED_MIN
    and
    index_tip_palm >= EXTENDED_TIP_DISTANCE_MIN
)
```

Middle:

```python
middle_extended = (
    middle_pip_angle >= PIP_EXTENDED_MIN
    and
    middle_dip_angle >= DIP_EXTENDED_MIN
    and
    middle_tip_palm >= EXTENDED_TIP_DISTANCE_MIN
)
```

For folded fingers:

```python
ring_folded = (
    (
        ring_pip_angle <= PIP_FOLDED_MAX
        or
        ring_dip_angle <= DIP_FOLDED_MAX
    )
    and
    ring_tip_palm <= FOLDED_TIP_DISTANCE_MAX
)
```

Do the same for pinky.

---

# 15. Detect the V Shape

SCISSORS should have separation between:

```text
INDEX_TIP = 8
MIDDLE_TIP = 12
```

Calculate:

```python
v_gap = distance_3d(
    landmarks[8],
    landmarks[12]
) / palm_size
```

Recommended initial:

```python
V_GAP_MIN = 0.30
```

A wide V should pass.

A moderately narrow V should also pass.

Do not make this threshold excessively strict.

---

# 16. Better V Angle

Also calculate the angle between the two open fingers.

One option:

```text
INDEX_TIP → midpoint/base → MIDDLE_TIP
```

Use an anchor around:

```text
midpoint between INDEX_MCP(5) and MIDDLE_MCP(9)
```

Example:

```python
v_anchor = (
    landmarks[5] + landmarks[9]
) / 2
```

Then:

```python
v_angle = angle_3points(
    landmarks[8],
    v_anchor,
    landmarks[12]
)
```

Use this as secondary evidence.

Example starting range:

```python
V_ANGLE_MIN = 10.0
V_ANGLE_MAX = 80.0
```

Do not require one exact V angle.

---

# 17. Thumb Handling

The thumb may:

```text
rest across ring/pinky
rest beside the palm
be slightly open
```

Therefore:

```text
THUMB = supporting feature only
```

Calculate if useful:

```python
thumb_tip_palm = distance_3d(
    landmarks[4],
    palm_center
) / palm_size
```

But do NOT reject valid SCISSORS just because thumb position differs.

Thumb should contribute only a small weight to the final score.

---

# 18. SCISSORS Score

Build a weighted geometric score.

Example:

```text
Index extended            +2
Middle extended           +2

Ring folded               +2
Pinky folded              +2

Index tip far from palm   +1
Middle tip far from palm  +1

Ring tip near palm        +1
Pinky tip near palm       +1

Valid V gap               +1
Valid V angle             +0.5

Thumb acceptable          +0.25
```

Calculate:

```python
scissors_score = (
    matched_weight /
    total_weight
)
```

Initial classification:

```python
SCISSORS_SCORE_THRESHOLD = 0.80
```

---

# 19. Mandatory Core Conditions

Even with a high score, do NOT classify as SCISSORS unless:

```python
mandatory_conditions = (
    index_extended
    and
    middle_extended
    and
    ring_folded
    and
    pinky_folded
)
```

Then:

```python
if (
    mandatory_conditions
    and
    scissors_score >= SCISSORS_SCORE_THRESHOLD
):
    gesture = "SCISSORS"
else:
    gesture = "UNKNOWN"
```

This avoids accidental false positives.

---

# 20. Explicit False-Positive Rejection

## Reject PAPER

If:

```text
Index = extended
Middle = extended
Ring = extended
Pinky = extended
```

then:

```text
NOT SCISSORS
```

## Reject ROCK

If:

```text
Index = folded
Middle = folded
Ring = folded
Pinky = folded
```

then:

```text
NOT SCISSORS
```

## Reject One-Finger Pose

If only index or only middle is extended:

```text
NOT SCISSORS
```

## Reject Three-Finger Pose

If ring or pinky becomes clearly extended together with index and middle:

```text
NOT SCISSORS
```

---

# 21. Palm-Facing and Back-Hand Facing

The detector must support:

```text
front of hand
back of hand
```

Do not depend on fixed:

```text
x coordinate direction
y coordinate direction
```

for classification.

Use:

```text
joint angles
3D distances
normalized distances
relative geometry
```

---

# 22. Left and Right Hands

Support:

```text
LEFT
RIGHT
```

Do not write logic that only works when:

```text
thumb.x < index.x
```

or vice versa.

Handedness can be displayed but should not be necessary for the four primary finger states.

---

# 23. Rotated Hands

Test at approximately:

```text
0°
15°
30°
45°
```

Natural rotation should not destroy classification.

Use world/3D landmarks when MediaPipe provides them.

---

# 24. Temporal Smoothing

Use:

```python
from collections import deque

history = deque(maxlen=7)
```

Store recent:

```text
SCISSORS
UNKNOWN
```

Example:

```text
SCISSORS
SCISSORS
UNKNOWN
SCISSORS
SCISSORS
SCISSORS
SCISSORS
```

Final:

```text
SCISSORS
```

Do not let a single bad frame immediately change the visible prediction.

---

# 25. Confidence

Use:

```python
confidence = scissors_score
```

Display:

```text
SCISSORS
Confidence: 94.2%
```

Clarify internally that this is:

```text
geometric confidence
```

not neural-network probability.

---

# 26. Live Landmark Overlay

Display:

```text
Hand: RIGHT

SCISSORS
Confidence: 94.2%

Landmarks: 21/21
```

Draw:

```text
21 landmark dots
skeleton connections
landmark indices
```

Recommended debug colors:

```text
Point      = red
Skeleton   = white
Index text = yellow
```

Any readable palette is acceptable.

---

# 27. Debug Panel

Provide an optional detailed panel:

```text
Index:
  PIP = 173.2°
  DIP = 169.8°
  State = EXTENDED

Middle:
  PIP = 171.9°
  DIP = 168.1°
  State = EXTENDED

Ring:
  PIP = 103.4°
  DIP = 118.2°
  State = FOLDED

Pinky:
  PIP = 97.8°
  DIP = 109.3°
  State = FOLDED

Index Tip / Palm  = 1.18
Middle Tip / Palm = 1.12

Ring Tip / Palm   = 0.61
Pinky Tip / Palm  = 0.56

V Gap   = 0.48
V Angle = 31.7°

SCISSORS Score = 0.94
```

This panel is important for threshold calibration.

---

# 28. Browser Implementation

For an existing website, use:

```text
MediaPipe Tasks Vision
HandLandmarker
JavaScript
Canvas
navigator.mediaDevices.getUserMedia()
```

Pipeline:

```text
Webcam video
   ↓
HandLandmarker.detectForVideo()
   ↓
21 landmarks
   ↓
normalize
   ↓
geometry functions
   ↓
finger states
   ↓
SCISSORS score
   ↓
Canvas overlay
```

No image upload.

No server-side inference required for this landmark rule.

---

# 29. Suggested JavaScript Modules

Recommended:

```text
web/
├── handLandmarker.js
├── handGeometry.js
├── scissorsGesture.js
├── handRenderer.js
└── app.js
```

Functions:

```javascript
calculateDistance()
calculateAngle()
calculatePalmCenter()
normalizeLandmarks()

isFingerExtended()
isFingerFolded()

calculateVGap()
calculateVAngle()

calculateScissorsScore()

drawLandmarks()
drawConnections()
```

Keep gesture logic separate from rendering code.

---

# 30. Python Alternative

For desktop testing:

```text
opencv-python
mediapipe
numpy
```

Suggested:

```text
src/
├── camera.py
├── geometry.py
├── hand_landmarks.py
├── scissors_gesture.py
├── visualization.py
└── main.py
```

---

# 31. Central Threshold Configuration

Use one configuration object/file.

Example:

```python
MIN_HAND_CONFIDENCE = 0.60

PIP_EXTENDED_MIN = 155.0
DIP_EXTENDED_MIN = 150.0

PIP_FOLDED_MAX = 145.0
DIP_FOLDED_MAX = 150.0

EXTENDED_TIP_DISTANCE_MIN = 0.95
FOLDED_TIP_DISTANCE_MAX = 0.85

V_GAP_MIN = 0.30

V_ANGLE_MIN = 10.0
V_ANGLE_MAX = 80.0

SCISSORS_SCORE_THRESHOLD = 0.80

HISTORY_SIZE = 7
```

Do not scatter magic numbers throughout the source code.

---

# 32. Positive Test Cases

Test SCISSORS with:

```text
palm-facing SCISSORS
back-facing SCISSORS

left hand
right hand

narrow V
medium V
wide V

slight left rotation
slight right rotation

hand close to camera
hand far from camera

thumb across folded fingers
thumb beside folded fingers
```

---

# 33. Negative Test Cases

Must NOT become SCISSORS:

```text
PAPER
ROCK

one finger
three fingers
four fingers
five fingers

index folded
middle folded

ring extended
pinky extended

half-open hand
```

Return:

```text
UNKNOWN
```

until other gesture classes are integrated.

---

# 34. Required Acceptance Criteria

The task is complete only if:

- [ ] live webcam works;
- [ ] one hand is detected;
- [ ] 21 landmarks are returned;
- [ ] all landmark points are drawn;
- [ ] skeleton is drawn;
- [ ] landmark IDs can be displayed;
- [ ] coordinates are normalized;
- [ ] palm center is calculated;
- [ ] index extension is correctly detected;
- [ ] middle extension is correctly detected;
- [ ] ring folding is correctly detected;
- [ ] pinky folding is correctly detected;
- [ ] fingertip-to-palm distances are calculated;
- [ ] V gap is calculated;
- [ ] V angle is calculated;
- [ ] thumb variation is tolerated;
- [ ] SCISSORS score is calculated;
- [ ] palm-facing SCISSORS works;
- [ ] back-facing SCISSORS works;
- [ ] left hand works;
- [ ] right hand works;
- [ ] narrow V works;
- [ ] PAPER is rejected;
- [ ] ROCK is rejected;
- [ ] temporal smoothing works;
- [ ] confidence is displayed;
- [ ] no manual image upload is required.

---

# 35. Required Final Report

After implementation, report:

## Final Pipeline

```text
Webcam
→ Hand Landmarker
→ 21 Landmarks
→ Normalize
→ Joint Angles
→ Finger States
→ V Geometry
→ SCISSORS Score
→ Temporal Smoothing
→ SCISSORS / UNKNOWN
```

## Finger Result

Example:

```text
Thumb:  TOLERANT
Index:  EXTENDED
Middle: EXTENDED
Ring:   FOLDED
Pinky:  FOLDED
```

## Thresholds

List actual final tuned values.

## Tests

Return:

```text
Palm-facing:      PASS/FAIL
Back-facing:      PASS/FAIL
Left hand:        PASS/FAIL
Right hand:       PASS/FAIL
Narrow V:         PASS/FAIL
Wide V:           PASS/FAIL
Rotated hand:     PASS/FAIL

PAPER rejection:  PASS/FAIL
ROCK rejection:   PASS/FAIL
3-finger reject:  PASS/FAIL
```

## Files Changed

List every created or modified source file.

---

# Final Instruction

Recognize SCISSORS primarily from this landmark state:

```text
INDEX  = EXTENDED
MIDDLE = EXTENDED

RING   = FOLDED
PINKY  = FOLDED

THUMB  = TOLERANT
```

Then confirm it with:

```text
index/middle tips far from palm
+
ring/pinky tips near palm
+
valid V separation
+
valid V angle
```

Do not rely primarily on RGB appearance, background, skin color, or camera orientation.

Use the geometric structure of the 21 hand landmarks.

Stabilize SCISSORS independently before combining it with PAPER and ROCK.
