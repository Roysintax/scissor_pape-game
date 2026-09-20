# PROMPT.md — ROCK Gesture Recognition with 21 Hand Landmarks

## Role

Act as a Senior Computer Vision Engineer specializing in real-time hand pose estimation and gesture recognition.

Your task is to implement a robust **ROCK / closed-fist gesture detector** using the standard **21 MediaPipe hand landmarks**.

The detector must work for a clenched fist seen from:

- palm-facing view,
- back-of-hand view,
- slight left/right rotation,
- slight wrist tilt,
- left hand,
- right hand.

Do NOT classify ROCK from the entire RGB image alone.

The primary signal must come from hand landmark geometry.

---

# 1. Main Objective

Build this pipeline:

```text
Webcam
  ↓
Hand Detection
  ↓
21 Hand Landmarks
  ↓
Landmark Normalization
  ↓
Finger Flexion Analysis
  ↓
Fingertip-to-Palm Distances
  ↓
Closed-Fist / ROCK Score
  ↓
Temporal Smoothing
  ↓
ROCK / UNKNOWN
```

Do not implement PAPER or SCISSORS in this task.

Focus on stabilizing ROCK first.

---

# 2. Standard 21 Landmark Indexes

Use exactly this MediaPipe convention:

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

# 3. Hand Skeleton

Use:

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

Draw:

- 21 landmark points,
- landmark index numbers,
- skeleton connections,
- handedness,
- current gesture,
- confidence.

---

# 4. ROCK Pose Definition

ROCK is a closed fist.

Expected finger state:

```text
Thumb   = folded / resting against fist
Index   = folded
Middle  = folded
Ring    = folded
Pinky   = folded
```

Do not require an identical thumb pose.

The thumb may:

```text
rest across the index/middle fingers
or
rest beside the fist
```

Both should still qualify as ROCK.

---

# 5. Important: Do Not Use Only Tip Y Position

Do NOT use simple logic such as:

```python
index_tip.y > index_pip.y
```

This breaks when the fist rotates.

Use:

- joint angles,
- normalized fingertip distances,
- palm-relative geometry,
- 3D coordinates when available.

---

# 6. Landmark Normalization

Normalize landmarks relative to the wrist:

```text
origin = landmark[0]
```

Then:

```text
relative_point = point - wrist
```

Normalize scale using one stable palm measurement.

Recommended:

```text
palm_size =
distance(WRIST, MIDDLE_MCP)
```

or:

```text
palm_width =
distance(INDEX_MCP, PINKY_MCP)
```

Use:

```text
normalized_point =
relative_point / palm_size
```

Do not compare raw pixel distances across different camera distances.

---

# 7. Palm Center

Create approximate palm center using:

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

This point is important for measuring how close fingertips are to the palm.

---

# 8. Geometry Utilities

Implement:

```python
distance_2d(a, b)
distance_3d(a, b)
angle_3points(a, b, c)
normalize_landmarks(landmarks)
calculate_palm_center(landmarks)
```

For angle at B:

```text
A → B → C
```

use:

```text
BA = A - B
BC = C - B

angle =
acos(
    dot(BA, BC) /
    (norm(BA) * norm(BC))
)
```

Convert radians to degrees.

Clamp cosine to:

```text
[-1, 1]
```

before `acos`.

---

# 9. Finger Flexion Logic

For ROCK, the four non-thumb fingers should be bent.

## Index

Landmarks:

```text
5 → 6 → 7 → 8
```

Calculate:

```python
index_pip_angle = angle_3points(5, 6, 7)
index_dip_angle = angle_3points(6, 7, 8)
```

An extended finger tends toward approximately:

```text
180°
```

A curled finger has smaller joint angles.

Initial configurable rule:

```python
index_curled = (
    index_pip_angle < PIP_CURLED_THRESHOLD
    or
    index_dip_angle < DIP_CURLED_THRESHOLD
)
```

Do not rely on angle alone.

Also require fingertip proximity to the palm.

---

# 10. Middle Finger

Use:

```text
9 → 10 → 11 → 12
```

Calculate:

```python
middle_pip_angle
middle_dip_angle
```

Determine:

```python
middle_curled
```

---

# 11. Ring Finger

Use:

```text
13 → 14 → 15 → 16
```

Calculate:

```python
ring_pip_angle
ring_dip_angle
```

Determine:

```python
ring_curled
```

---

# 12. Pinky Finger

Use:

```text
17 → 18 → 19 → 20
```

Calculate:

```python
pinky_pip_angle
pinky_dip_angle
```

Determine:

```python
pinky_curled
```

---

# 13. Recommended Initial Curled Thresholds

Start with configurable thresholds such as:

```python
PIP_CURLED_THRESHOLD = 145.0
DIP_CURLED_THRESHOLD = 150.0
```

These are initial tuning values only.

Do not assume they are universally optimal.

Use the debug panel to tune them empirically.

---

# 14. Fingertip-to-Palm Distance

For ROCK, fingertips should be relatively close to the palm center.

Measure:

```text
INDEX_TIP   8
MIDDLE_TIP 12
RING_TIP   16
PINKY_TIP  20
```

Calculate:

```python
index_tip_distance = distance_3d(
    landmarks[8],
    palm_center
)

middle_tip_distance = distance_3d(
    landmarks[12],
    palm_center
)

ring_tip_distance = distance_3d(
    landmarks[16],
    palm_center
)

pinky_tip_distance = distance_3d(
    landmarks[20],
    palm_center
)
```

Normalize using palm size:

```python
index_ratio =
index_tip_distance / palm_size
```

Do the same for all fingers.

In a closed fist, these ratios should generally be smaller than in PAPER.

---

# 15. Fingertip-to-MCP Evidence

Also measure each fingertip relative to its own MCP.

Example:

```text
INDEX:
distance(8, 5)

MIDDLE:
distance(12, 9)

RING:
distance(16, 13)

PINKY:
distance(20, 17)
```

Normalize these distances.

A folded finger usually has a shorter fingertip-to-MCP spatial span than an extended finger.

Use this only as supporting evidence.

---

# 16. Tip-to-Base Comparison

For each non-thumb finger, compare:

```text
TIP → palm center
vs
PIP → palm center
```

or:

```text
TIP → MCP
```

This helps distinguish:

```text
curled finger
```

from:

```text
extended finger viewed at an unusual rotation
```

Do not use one single distance feature as the whole classifier.

---

# 17. Thumb Logic for ROCK

The thumb is the most variable part of a fist.

Do NOT require one exact thumb shape.

Use a tolerant ROCK thumb rule.

Evaluate:

```text
1 → 2 → 3 → 4
```

Calculate:

```python
thumb_mcp_angle = angle_3points(1, 2, 3)
thumb_ip_angle  = angle_3points(2, 3, 4)
```

Also calculate:

```python
thumb_to_palm =
distance_3d(
    landmarks[4],
    palm_center
)
```

and optionally:

```text
distance(4, 5)
distance(4, 9)
distance(4, 13)
```

A ROCK thumb should generally remain reasonably close to the fist.

However:

```text
thumb_open slightly to side
```

must not immediately reject a valid ROCK if the four main fingers are clearly curled.

Therefore, thumb evidence should have lower weight than the four primary fingers.

---

# 18. Strong ROCK Core Rule

The main ROCK condition should prioritize:

```python
four_fingers_curled = (
    index_curled
    and middle_curled
    and ring_curled
    and pinky_curled
)
```

Then combine with compactness:

```python
fingertips_near_palm = (
    index_tip_near_palm
    and middle_tip_near_palm
    and ring_tip_near_palm
    and pinky_tip_near_palm
)
```

Core ROCK evidence:

```python
rock_core = (
    four_fingers_curled
    and fingertips_near_palm
)
```

---

# 19. ROCK Score

Use a score rather than one fragile boolean.

Example evidence:

```text
Index curled              +1
Middle curled             +1
Ring curled               +1
Pinky curled              +1

Index tip near palm       +1
Middle tip near palm      +1
Ring tip near palm        +1
Pinky tip near palm       +1

Thumb compact             +0.5
Overall fist compactness  +1
```

Calculate:

```python
rock_score =
matched_weight / total_weight
```

Suggested initial threshold:

```python
ROCK_SCORE_THRESHOLD = 0.80
```

Keep this configurable.

---

# 20. Fist Compactness

Create an additional compactness feature.

For example, calculate average normalized distance of:

```text
8, 12, 16, 20
```

to palm center:

```python
mean_tip_to_palm = mean([
    normalized_distance(8, palm_center),
    normalized_distance(12, palm_center),
    normalized_distance(16, palm_center),
    normalized_distance(20, palm_center)
])
```

A compact fist should have a much smaller value than an open palm.

Use this as a strong secondary feature.

---

# 21. Palm-Facing and Back-of-Hand Views

The detector must support both:

```text
palm side toward camera
```

and:

```text
back of hand toward camera
```

Do not use:

```text
finger tip must be left/right of another point
```

as a hard ROCK requirement.

Prefer rotation-invariant geometry:

```text
angles
3D distances
normalized distances
relative compactness
```

---

# 22. Hand Rotation

Test ROCK with:

```text
0°
15°
30°
45°
```

of natural wrist/hand rotation.

The classification should remain stable.

If orientation changes significantly, use world landmarks or 3D coordinates when available.

---

# 23. Occlusion Handling

A closed fist can cause fingertips and joints to overlap visually.

MediaPipe may still estimate landmarks, but confidence can become unstable.

Return:

```text
UNKNOWN
```

if:

- fewer than 21 valid landmarks are available,
- landmark confidence is too low,
- geometry changes unrealistically between frames,
- hand is partially outside camera,
- several finger landmarks collapse into invalid positions.

Do not force ROCK when landmark quality is weak.

---

# 24. Temporal Smoothing

Use:

```python
from collections import deque

history = deque(maxlen=7)
```

Store:

```text
ROCK / UNKNOWN
```

Only show ROCK when recent frames agree.

Example:

```text
ROCK
ROCK
UNKNOWN
ROCK
ROCK
ROCK
ROCK
```

Final:

```text
ROCK
```

---

# 25. Confidence

For this landmark-based version:

```python
confidence = rock_score
```

Display:

```text
ROCK
Confidence: 93.2%
```

This is a geometric confidence score, not a neural-network probability.

---

# 26. Debug Panel

Add optional debug information:

```text
Hand: RIGHT
Landmarks: 21/21

Index:   CURLED
Middle:  CURLED
Ring:    CURLED
Pinky:   CURLED
Thumb:   COMPACT

Index PIP:   91.4°
Index DIP:   103.2°
Middle PIP:  86.9°
Middle DIP:  98.1°

Index Tip/Palm:   0.42
Middle Tip/Palm:  0.38
Ring Tip/Palm:    0.41
Pinky Tip/Palm:   0.46

Fist Compactness: 0.42
Rock Score:       0.93

Gesture:
ROCK
```

This is required for empirical threshold tuning.

---

# 27. Visualization

Live webcam overlay must display:

```text
┌────────────────────────────────────┐
│ RIGHT HAND                         │
│ ROCK                               │
│ Confidence: 93.2%                  │
│                                    │
│        21-point landmark fist      │
│        with skeleton overlay       │
│                                    │
│ Landmarks: 21/21                   │
└────────────────────────────────────┘
```

Draw points over the detected fist.

Even when landmarks overlap visually, keep the 21-point topology intact.

---

# 28. Styling

Recommended:

```text
Landmark points = red or green
Skeleton lines  = white or blue
Index numbers   = high-contrast
```

Make landmark points visible on both light and dark backgrounds.

---

# 29. Browser Implementation

For website implementation, prefer:

```text
MediaPipe Tasks Vision
HandLandmarker
JavaScript
Canvas
getUserMedia
```

Pipeline:

```text
HTML video
   ↓
MediaPipe HandLandmarker
   ↓
21 landmarks
   ↓
normalize
   ↓
joint-angle calculations
   ↓
fingertip compactness
   ↓
ROCK score
   ↓
Canvas overlay
```

No image upload should be required.

---

# 30. Python Implementation

For desktop testing:

```text
opencv-python
mediapipe
numpy
```

Suggested project structure:

```text
src/
├── camera.py
├── hand_landmarks.py
├── geometry.py
├── rock_gesture.py
├── visualization.py
└── main.py
```

Recommended functions:

```python
detect_hand(frame)
normalize_landmarks(landmarks)
calculate_palm_center(landmarks)
calculate_joint_angle(a, b, c)
is_finger_curled(...)
is_thumb_compact(...)
calculate_fist_compactness(...)
calculate_rock_score(...)
draw_landmarks(...)
draw_skeleton(...)
```

---

# 31. Central Configuration

Keep thresholds in one location.

Example:

```python
MIN_HAND_DETECTION_CONFIDENCE = 0.60

PIP_CURLED_THRESHOLD = 145.0
DIP_CURLED_THRESHOLD = 150.0

TIP_TO_PALM_MAX_RATIO = 0.75

ROCK_SCORE_THRESHOLD = 0.80

HISTORY_SIZE = 7
```

The values above are starting points only.

Tune them using actual webcam results.

---

# 32. Positive Test Cases

ROCK must be tested with:

```text
closed fist — palm facing camera
closed fist — back of hand facing camera
closed fist — slight left rotation
closed fist — slight right rotation
left hand
right hand
thumb across fingers
thumb resting beside fist
hand near camera
hand farther from camera
```

---

# 33. Negative Test Cases

These should NOT become ROCK:

```text
open palm / PAPER
scissors
one finger extended
two fingers extended
three fingers extended
four fingers extended
half-open hand
loosely curled fingers
```

Return:

```text
UNKNOWN
```

for now.

---

# 34. Special Rule Against False ROCK

Reject ROCK if any two or more of these are clearly extended:

```text
Index
Middle
Ring
Pinky
```

Example:

```python
extended_count = sum([
    index_extended,
    middle_extended,
    ring_extended,
    pinky_extended
])

if extended_count >= 2:
    rock_score = min(
        rock_score,
        ROCK_REJECTION_CAP
    )
```

This helps prevent SCISSORS or partially open hands from being misclassified as ROCK.

---

# 35. Landmark Stability

Track normalized landmark movement across recent frames.

If hand geometry suddenly jumps drastically while camera remains active:

```text
do not immediately change class
```

Use temporal smoothing.

Optionally reject unstable frames.

---

# 36. Acceptance Criteria

The task is complete only when:

- [ ] webcam detects a hand;
- [ ] all 21 landmarks are available;
- [ ] indices 0–20 are drawn;
- [ ] skeleton is drawn;
- [ ] landmark coordinates are normalized;
- [ ] palm center is calculated;
- [ ] index curl state is detected;
- [ ] middle curl state is detected;
- [ ] ring curl state is detected;
- [ ] pinky curl state is detected;
- [ ] thumb compactness is evaluated;
- [ ] fingertip-to-palm distances are calculated;
- [ ] fist compactness is calculated;
- [ ] ROCK score is calculated;
- [ ] palm-facing fist becomes ROCK;
- [ ] back-facing fist becomes ROCK;
- [ ] left hand works;
- [ ] right hand works;
- [ ] PAPER does not become ROCK;
- [ ] SCISSORS does not become ROCK;
- [ ] prediction smoothing works;
- [ ] confidence is displayed;
- [ ] no image upload is required.

---

# 37. Required Final Report

After implementation, report:

## Architecture

```text
Webcam
→ Hand Landmarker
→ 21 Landmarks
→ Normalization
→ Finger Flexion
→ Tip-to-Palm Distances
→ Fist Compactness
→ ROCK Score
→ Temporal Smoothing
→ ROCK / UNKNOWN
```

## Thresholds

List all final values used.

## Test Results

Return:

```text
Palm-facing fist:       PASS/FAIL
Back-facing fist:       PASS/FAIL
Rotated fist:           PASS/FAIL
Left hand:              PASS/FAIL
Right hand:             PASS/FAIL
Thumb across fingers:   PASS/FAIL
Thumb beside fist:      PASS/FAIL

PAPER:                  NOT ROCK / FAIL
SCISSORS:               NOT ROCK / FAIL
Half-open hand:         NOT ROCK / FAIL
```

## Files Changed

List all created or modified files.

---

# Final Instruction

ROCK must be recognized as a **compact closed-fist geometry**.

Do not rely on the whole image appearance.

Use the geometric relationship between:

```text
WRIST
MCP
PIP
DIP
FINGERTIPS
PALM CENTER
```

The strongest ROCK evidence should be:

```text
four main fingers curled
+
fingertips near palm
+
compact hand geometry
```

Thumb position must be tolerant because valid fists can place the thumb differently.

Stabilize ROCK first before integrating it with PAPER and SCISSORS.
