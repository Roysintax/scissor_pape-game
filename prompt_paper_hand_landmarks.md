# PROMPT.md — PAPER Gesture Recognition with 21 Hand Landmarks

## Role
Act as a Senior Computer Vision Engineer specializing in real-time hand pose estimation and gesture recognition.

## Goal
Improve the existing Rock–Paper–Scissors system so that the **PAPER** gesture is recognized from the geometric structure of the hand, not only from raw image classification.

The system must use **21 hand landmarks** (MediaPipe convention), draw the points and skeleton on the live webcam, then classify an open palm as `PAPER`.

Do not implement ROCK or SCISSORS yet. Focus only on making PAPER stable.

---

## 1. Pipeline

```text
Webcam
  ↓
Hand Detection
  ↓
21 Hand Landmarks
  ↓
Landmark Normalization
  ↓
Finger Joint Geometry
  ↓
Open/Closed Finger Detection
  ↓
PAPER Score
  ↓
Temporal Smoothing
  ↓
PAPER / UNKNOWN
```

Prefer MediaPipe Hand Landmarker / MediaPipe Hands.

---

## 2. Standard 21 Landmark Indexes

Use exactly this convention:

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

The open-palm PAPER pose must have all five fingers extended.

---

## 3. Hand Skeleton

Use these connections:

```python
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (0, 9), (9, 10), (10, 11), (11, 12),

    (0, 13), (13, 14), (14, 15), (15, 16),

    (0, 17), (17, 18), (18, 19), (19, 20),

    (5, 9), (9, 13), (13, 17)
]
```

For every frame:

- draw all 21 points;
- draw landmark numbers `0–20`;
- draw all skeleton connections;
- show handedness if available;
- show current gesture and confidence.

Recommended visual style:

```text
Landmark point : green
Skeleton line  : blue
Point number   : red
```

---

## 4. Expected PAPER Shape

Conceptually:

```text
        8       12      16      20
        ●       ●       ●       ●
        │       │       │       │
        7       11      15      19
        ●       ●       ●       ●
        │       │       │       │
        6       10      14      18
        ●       ●       ●       ●
        │       │       │       │
        5       9       13      17
         \      │       │      /
          \     │       │     /
           \    │       │    /
            \   │       │   /
             \  │       │  /
              \ │       │ /
               \│       │/
                ● 0

Thumb:
0 → 1 → 2 → 3 → 4
```

PAPER means:

```text
thumb   = extended
index   = extended
middle  = extended
ring    = extended
pinky   = extended
```

---

## 5. Do Not Use Only Y Coordinates

Do not classify a finger using only logic like:

```python
tip.y < pip.y
```

That breaks when the hand rotates.

Instead, use joint angles and normalized distances.

Use 3D `(x, y, z)` landmarks when available.

---

## 6. Landmark Normalization

Raw coordinates change with camera distance and hand location.

Normalize all landmarks relative to the wrist.

Use:

```text
origin = landmark[0]
```

Then:

```text
relative_landmark = landmark - wrist
```

Normalize the scale using:

```text
palm_size = distance(WRIST, MIDDLE_MCP)
```

or:

```text
palm_width = distance(INDEX_MCP, PINKY_MCP)
```

Then:

```text
normalized_coordinate =
relative_coordinate / palm_size
```

The gesture rules must operate on normalized geometry whenever possible.

---

## 7. Geometry Utilities

Implement reusable functions:

```python
distance_2d(a, b)
distance_3d(a, b)
angle_3points(a, b, c)
normalize_landmarks(landmarks)
calculate_palm_center(landmarks)
```

For `angle_3points(A, B, C)`, calculate the angle at `B`.

Use vector math:

```text
BA = A - B
BC = C - B

angle =
acos(
    dot(BA, BC) /
    (norm(BA) * norm(BC))
)
```

Convert to degrees.

Clamp the cosine to:

```text
[-1, 1]
```

before `acos`.

---

## 8. Non-Thumb Finger Extension

For each non-thumb finger, evaluate the two main straightness angles.

### Index

```text
5 → 6 → 7 → 8
```

Calculate:

```python
index_pip_angle = angle_3points(5, 6, 7)
index_dip_angle = angle_3points(6, 7, 8)
```

Initial rule:

```python
index_extended = (
    index_pip_angle > PIP_STRAIGHT_THRESHOLD
    and
    index_dip_angle > DIP_STRAIGHT_THRESHOLD
)
```

### Middle

```text
9 → 10 → 11 → 12
```

### Ring

```text
13 → 14 → 15 → 16
```

### Pinky

```text
17 → 18 → 19 → 20
```

Recommended initial thresholds:

```python
PIP_STRAIGHT_THRESHOLD = 155.0
DIP_STRAIGHT_THRESHOLD = 150.0
```

Keep them configurable. Do not scatter magic numbers throughout the code.

---

## 9. Thumb Extension

The thumb needs separate logic.

Use:

```text
1 → 2 → 3 → 4
```

Calculate:

```python
thumb_mcp_angle = angle_3points(1, 2, 3)
thumb_ip_angle  = angle_3points(2, 3, 4)
```

Also calculate thumb separation from the palm.

Create palm center from:

```text
0, 5, 9, 13, 17
```

Example:

```python
palm_center = mean([
    landmarks[0],
    landmarks[5],
    landmarks[9],
    landmarks[13],
    landmarks[17]
])
```

Then:

```python
thumb_distance = distance_3d(
    landmarks[4],
    palm_center
)
```

Normalize it by palm size.

The thumb is open when:

```text
thumb joint geometry is sufficiently straight
AND
thumb tip is sufficiently separated from palm
```

Do not use only:

```text
thumb_tip.x > thumb_ip.x
```

because this fails between left and right hands.

---

## 10. Fingertip-to-Palm Distances

For PAPER, these fingertips should be clearly away from the palm:

```text
8   INDEX_TIP
12  MIDDLE_TIP
16  RING_TIP
20  PINKY_TIP
```

Calculate:

```python
distance(8, palm_center)
distance(12, palm_center)
distance(16, palm_center)
distance(20, palm_center)
```

Normalize every distance using palm size.

Use these values as supporting evidence that the hand is open.

---

## 11. Finger Spread

Calculate:

```text
distance(8, 12)
distance(12, 16)
distance(16, 20)
distance(4, 8)
```

Finger spread is only supporting evidence.

Do not reject PAPER merely because the user keeps the fingers close together.

---

## 12. PAPER Score

Do not use only a single boolean condition.

Build a geometric score.

Example evidence:

```text
Thumb extended              +1
Index extended              +1
Middle extended             +1
Ring extended               +1
Pinky extended              +1
Fingertips far from palm    +1
Palm openness valid         +1
```

Then:

```python
paper_score = matched_conditions / total_conditions
```

Suggested initial threshold:

```python
PAPER_SCORE_THRESHOLD = 0.80
```

Classification:

```python
if paper_score >= PAPER_SCORE_THRESHOLD:
    gesture = "PAPER"
else:
    gesture = "UNKNOWN"
```

Keep the score threshold configurable.

---

## 13. Strong PAPER Condition

Also expose a strict boolean:

```python
all_fingers_extended = (
    thumb_extended
    and index_extended
    and middle_extended
    and ring_extended
    and pinky_extended
)
```

Use it together with `paper_score`.

Do not force PAPER from only four extended fingers.

---

## 14. Temporal Smoothing

Predictions must not flicker frame by frame.

Use:

```python
from collections import deque

history = deque(maxlen=7)
```

Store recent:

```text
PAPER / UNKNOWN
```

Display PAPER only if the majority of recent frames agree.

Example:

```text
PAPER
PAPER
UNKNOWN
PAPER
PAPER
PAPER
PAPER
```

Final display:

```text
PAPER
```

---

## 15. Confidence

For this geometric version, confidence can initially be:

```python
confidence = paper_score
```

Display:

```text
PAPER
Confidence: 91.4%
```

This is a geometric confidence score, not a neural-network probability.

---

## 16. Debug Overlay

Provide a debug mode that displays:

```text
Hand: RIGHT
Landmarks: 21/21

Thumb:   OPEN
Index:   OPEN
Middle:  OPEN
Ring:    OPEN
Pinky:   OPEN

Index PIP:   171.4°
Index DIP:   167.8°
Middle PIP:  174.3°
Middle DIP:  170.9°

Palm Size:   ...
Paper Score: 0.91

Gesture:
PAPER
```

This is required because thresholds must be tuned empirically.

---

## 17. Handedness

Support:

```text
LEFT
RIGHT
```

If the MediaPipe result provides handedness, use it.

Do not create thumb rules that only work for one hand.

If the webcam preview is mirrored visually, make sure coordinate logic is not accidentally mirrored twice.

---

## 18. Invalid Detection

Return:

```text
UNKNOWN
```

when:

- 21 valid landmarks are not available;
- detection confidence is too low;
- most fingers are outside the frame;
- landmark geometry becomes unstable;
- the hand is heavily occluded.

Do not force PAPER when evidence is weak.

---

## 19. Required Live Webcam UI

The live display should show:

```text
┌────────────────────────────────────┐
│ RIGHT HAND                         │
│ PAPER                              │
│ Confidence: 91.4%                  │
│                                    │
│      landmark skeleton             │
│      with points 0–20              │
│                                    │
│ Landmarks: 21/21                   │
└────────────────────────────────────┘
```

The webcam must remain live.

No manual image upload is required.

---

## 20. Recommended Browser Implementation

If this is part of the existing website, prefer:

```text
MediaPipe Tasks Vision
HandLandmarker
JavaScript
Canvas
getUserMedia()
```

Browser pipeline:

```text
HTML Video
   ↓
MediaPipe HandLandmarker
   ↓
21 landmarks
   ↓
JavaScript geometry functions
   ↓
PAPER score
   ↓
Canvas overlay
```

Use canvas to draw:

```text
points
skeleton
indices
gesture text
```

---

## 21. Recommended Python Implementation

For desktop testing:

```text
opencv-python
mediapipe
numpy
```

Suggested structure:

```text
src/
├── camera.py
├── hand_landmarks.py
├── geometry.py
├── paper_gesture.py
├── visualization.py
└── main.py
```

Functions should be modular:

```python
detect_hand(frame)
normalize_landmarks(landmarks)
calculate_palm_center(landmarks)
calculate_joint_angle(a, b, c)
is_finger_extended(...)
is_thumb_extended(...)
calculate_paper_score(...)
draw_landmarks(...)
draw_skeleton(...)
```

---

## 22. Configuration

Put all thresholds in one configuration section.

Example:

```python
MIN_HAND_DETECTION_CONFIDENCE = 0.60

PIP_STRAIGHT_THRESHOLD = 155.0
DIP_STRAIGHT_THRESHOLD = 150.0

PAPER_SCORE_THRESHOLD = 0.80

HISTORY_SIZE = 7
```

Do not hard-code these values in many functions.

---

## 23. Tests

PAPER must be tested with:

```text
front-facing open palm
slightly rotated open palm
left hand
right hand
hand close to camera
hand far from camera
fingers close together
fingers spread apart
```

It must NOT classify these as PAPER:

```text
closed fist
rock pose
scissors pose
one finger
two fingers
three fingers
four fingers
bent fingers
partially closed palm
```

For this stage, all non-PAPER poses should return:

```text
UNKNOWN
```

Do not implement ROCK and SCISSORS yet.

---

## 24. Acceptance Criteria

The task is complete only when:

- [ ] webcam detects a hand;
- [ ] 21 landmarks are returned;
- [ ] points `0–20` are displayed;
- [ ] skeleton connections are displayed;
- [ ] wrist is landmark `0`;
- [ ] standard MediaPipe indices are used;
- [ ] normalized coordinates are available;
- [ ] joint angles are calculated;
- [ ] thumb open/closed state is detected;
- [ ] index open/closed state is detected;
- [ ] middle open/closed state is detected;
- [ ] ring open/closed state is detected;
- [ ] pinky open/closed state is detected;
- [ ] PAPER score is calculated;
- [ ] open palm becomes PAPER;
- [ ] closed or bent hands do not become PAPER;
- [ ] left and right hands work;
- [ ] prediction smoothing works;
- [ ] confidence is displayed;
- [ ] camera remains responsive;
- [ ] no image upload is required.

---

## 25. Required Final Report

After implementation, report:

### Architecture

```text
Webcam
→ Hand Landmarker
→ 21 Landmarks
→ Normalization
→ Joint Angles
→ Finger States
→ PAPER Score
→ Temporal Smoothing
→ PAPER / UNKNOWN
```

### Thresholds

List the actual values used.

### Test Result

Use:

```text
Open palm front:          PASS/FAIL
Open palm rotated:        PASS/FAIL
Left hand:                PASS/FAIL
Right hand:               PASS/FAIL
Rock pose:                NOT PAPER / FAIL
Scissors pose:            NOT PAPER / FAIL
Bent fingers:             NOT PAPER / FAIL
Partially closed hand:    NOT PAPER / FAIL
```

### Files Changed

List all created or modified files.

---

# Final Instruction

Do not recognize PAPER from the entire RGB image alone.

The primary recognition signal must come from the geometric relationship between the 21 hand landmarks:

```text
WRIST
MCP
PIP
DIP
FINGERTIPS
PALM CENTER
```

Implement and stabilize **PAPER only** first.

The next phases, ROCK and SCISSORS, will reuse the same 21-landmark pipeline after PAPER works reliably.
