# PROMPT.md — Rock Paper Scissors vs Robot (Random Opponent + 3-Second Battle Countdown)

## Role

Act as a Senior Computer Vision Engineer, JavaScript Game Developer, Frontend Engineer, and Real-Time Interaction Designer.

Your task is to build a complete **Rock Paper Scissors vs Robot** web application.

The user will fight against a **robot opponent**.

The application must use:

1. **live webcam hand gesture recognition** for the user,
2. **three provided robot hand images** for the opponent,
3. a **Start** button,
4. a **3-second countdown** before the round is resolved,
5. a **fair random algorithm** that cannot easily be predicted by users,
6. correct Rock–Paper–Scissors rules.

---

# 1. Core Goal

Build a playable browser game:

```text
User clicks Start
        ↓
3-second countdown begins
        ↓
User shows hand gesture to webcam
        ↓
System detects user's gesture
        ↓
At countdown end, user gesture is locked
        ↓
Robot chooses Rock / Paper / Scissors randomly
        ↓
Robot image is revealed
        ↓
Winner is calculated
        ↓
Result is displayed
```

The robot opponent must use the provided image assets:

- `assets/robot_rock.png`
- `assets/robot_paper.png`
- `assets/robot_scissors.png`

---

# 2. Required Assets

Use these exact provided image assets for the robot opponent:

```text
assets/robot_rock.png
assets/robot_paper.png
assets/robot_scissors.png
```

Map them exactly:

```text
robot_rock.png     → ROCK
robot_paper.png    → PAPER
robot_scissors.png → SCISSORS
```

The robot display must update based on the robot’s random choice.

---

# 3. Main Rules

Use the normal Rock–Paper–Scissors rules:

```text
SCISSORS beats PAPER
ROCK beats SCISSORS
PAPER beats ROCK
```

Also:

```text
SCISSORS loses to ROCK
ROCK loses to PAPER
PAPER loses to SCISSORS
```

And:

```text
same move vs same move = DRAW
```

Required logic table:

| Player | Robot | Result |
|--------|-------|--------|
| rock | rock | draw |
| rock | paper | lose |
| rock | scissors | win |
| paper | rock | win |
| paper | paper | draw |
| paper | scissors | lose |
| scissors | rock | lose |
| scissors | paper | win |
| scissors | scissors | draw |

---

# 4. Randomness Requirement

The robot choice must be random and not easily predictable.

Do NOT use weak predictable logic such as:

```javascript
Math.random()
```

unless there is no alternative.

Preferred browser method:

```javascript
window.crypto.getRandomValues()
```

Example concept:

```javascript
function secureRandomChoice() {
  const array = new Uint32Array(1);
  window.crypto.getRandomValues(array);
  const value = array[0] % 3;

  const choices = ["rock", "paper", "scissors"];
  return choices[value];
}
```

If a backend version is used, prefer:

- Python: `secrets.choice(...)`
- Node.js: `crypto.randomInt(...)`

The goal is a fair unpredictable robot choice.

---

# 5. User Gesture Input

The user gesture must come from **live webcam recognition**, not button clicks for rock/paper/scissors.

Preferred input pipeline:

```text
Webcam
   ↓
Hand landmark detection
   ↓
Gesture classification
   ↓
rock / paper / scissors
```

Recommended approach:

- MediaPipe Hand Landmarker
- 21 hand landmarks
- gesture rules for ROCK, PAPER, SCISSORS

If the landmark-based detector already exists, integrate it directly.

If gesture certainty is low, show:

```text
UNKNOWN
```

The round must only resolve if a valid gesture is available at the end of countdown.

---

# 6. Round Flow

Required round flow:

## Idle State

Display:

- webcam ready,
- robot waiting state,
- Start button enabled,
- current score,
- instruction to place hand in camera.

## On Start Button Click

1. Disable the Start button.
2. Reset previous result.
3. Start a 3-second countdown:
   - `3`
   - `2`
   - `1`
4. During countdown:
   - webcam remains active,
   - gesture recognition continues,
   - latest stable user gesture is tracked.

## At Countdown End

1. Lock the user's final stable gesture.
2. Generate the robot's random choice.
3. Reveal the robot image.
4. Compare user choice vs robot choice.
5. Show result:
   - WIN
   - LOSE
   - DRAW
6. Update scoreboard.
7. Re-enable Start button for the next round.

---

# 7. Countdown Rules

The countdown must last exactly approximately 3 seconds.

Recommended visual:

```text
3
2
1
FIGHT!
```

or:

```text
3
2
1
GO!
```

At the moment of `GO!`, freeze the player's gesture and resolve the round.

Do not allow the player to change the result after the round is locked.

---

# 8. Stable Gesture Locking

Because real-time webcam predictions may flicker, use temporal smoothing.

Recommended:

```javascript
const gestureHistory = [];
```

or a fixed-size queue.

Use the latest 5–10 predictions.

At countdown end, choose the player's move from the **most stable recent prediction**.

Example methods:

- majority vote,
- average confidence over recent frames,
- latest stable label above threshold.

Recommended rule:

```text
If a gesture is consistently detected during the last ~0.5–1.0 seconds,
lock that gesture as the final player move.
```

If no stable gesture is available:

```text
Result = INVALID / NO MOVE
```

and the round should not count as a normal win/loss/draw.

---

# 9. Required Gesture Labels

The only valid player gesture labels are:

```text
rock
paper
scissors
unknown
```

Map exactly to lowercase internal values.

Display to user as:

```text
ROCK
PAPER
SCISSORS
UNKNOWN
```

---

# 10. Winner Evaluation Logic

Implement a dedicated function:

```javascript
function getRoundResult(playerMove, robotMove) {
  if (playerMove === robotMove) return "draw";

  if (
    (playerMove === "rock" && robotMove === "scissors") ||
    (playerMove === "paper" && robotMove === "rock") ||
    (playerMove === "scissors" && robotMove === "paper")
  ) {
    return "win";
  }

  return "lose";
}
```

If player move is invalid:

```javascript
if (!["rock", "paper", "scissors"].includes(playerMove)) {
  return "invalid";
}
```

---

# 11. Required UI

Design a clean modern UI.

The screen should include:

## Header

```text
Rock Paper Scissors vs Robot
```

## Left Panel — Player

- live webcam
- detected gesture
- gesture confidence
- label: `You`

## Center Panel

- large countdown display
- Start button
- result text
- optional round status

## Right Panel — Robot

- label: `Robot`
- displayed robot gesture image
- robot move label
- robot waiting placeholder before reveal

## Scoreboard

Display:

```text
Wins
Losses
Draws
Invalid
```

Optionally also show total rounds.

---

# 12. Required UI States

## Before Start

Robot panel shows:

```text
Waiting...
```

or a neutral placeholder.

## During Countdown

Robot gesture must remain hidden or show a neutral standby state.

## After Reveal

Show one of:

- robot_rock.png
- robot_paper.png
- robot_scissors.png

and display corresponding text.

---

# 13. Visual Example

Conceptual layout:

```text
---------------------------------------------------------
      Rock Paper Scissors vs Robot
---------------------------------------------------------

┌──────────────────┐   ┌──────────────┐   ┌──────────────────┐
│      YOU         │   │   COUNTDOWN  │   │      ROBOT       │
│                  │   │      3       │   │                  │
│   [ LIVE CAM ]   │   │ [ START BTN ]│   │ [ robot image ]  │
│                  │   │              │   │                  │
│ Gesture: PAPER   │   │ Result: WIN  │   │ Move: ROCK       │
└──────────────────┘   └──────────────┘   └──────────────────┘

Wins: 3    Losses: 1    Draws: 2    Invalid: 0
```

---

# 14. Required Robot Asset Mapping

Implement helper mapping:

```javascript
const robotImages = {
  rock: "assets/robot_rock.png",
  paper: "assets/robot_paper.png",
  scissors: "assets/robot_scissors.png"
};
```

When robot move is chosen, update:

- `img.src`
- alt text
- visible label

---

# 15. Sound / Optional Effects

Optional, but good if implemented:

- countdown beep,
- reveal sound,
- win sound,
- lose sound,
- draw sound.

Do not make audio mandatory.

The main requirement is the gameplay logic.

---

# 16. Fairness / Anti-Predictability

Important: the robot choice must not depend on the user's current move.

Do NOT make the robot “cheat”.

Correct flow:

```text
user gesture tracking
        +
secure random robot choice
        +
standard winner comparison
```

Robot choice must be independent.

Do not implement adaptive AI that counters the user.

The game should be fair and random.

---

# 17. Recommended Tech Stack

Preferred stack:

- HTML
- CSS
- JavaScript
- MediaPipe Hand Landmarker
- Canvas overlay (optional)
- `navigator.mediaDevices.getUserMedia()`

No backend is required unless needed.

If no backend is used, randomness should still use:

```javascript
window.crypto.getRandomValues()
```

---

# 18. Suggested File Structure

Create something similar to:

```text
rps-robot/
├── prompt.md
├── README.md
├── index.html
├── styles.css
├── app.js
├── assets/
│   ├── robot_rock.png
│   ├── robot_paper.png
│   └── robot_scissors.png
└── js/
    ├── gameLogic.js
    ├── randomChoice.js
    ├── gestureRecognition.js
    └── uiController.js
```

You may simplify if needed, but keep logic modular.

---

# 19. Required Functions

Implement at minimum:

```javascript
startRound()
startCountdown()
lockPlayerMove()
generateRobotMove()
getRoundResult(playerMove, robotMove)
updateRobotDisplay(robotMove)
updateScoreboard(result)
resetRoundUI()
```

Recommended random function:

```javascript
getSecureRandomMove()
```

Recommended gesture function:

```javascript
getStablePlayerGesture()
```

---

# 20. Required Scoreboard Logic

Maintain counters for:

```javascript
wins
losses
draws
invalid
```

Rules:

- `win` → wins + 1
- `lose` → losses + 1
- `draw` → draws + 1
- `invalid` → invalid + 1

Invalid rounds should not be silently ignored.

---

# 21. Invalid Round Handling

If the webcam cannot confidently detect:

- rock,
- paper,
- scissors,

then result should become:

```text
INVALID MOVE
```

Possible causes:

- hand not visible,
- gesture unstable,
- unknown pose,
- no hand detected.

In that case:

- show the robot choice only if you want consistent round flow, OR skip robot reveal,
- increase invalid counter,
- re-enable Start button.

Recommended simple behavior:

```text
Countdown ends
→ no valid user gesture
→ result = INVALID MOVE
→ robot does not attack or reveal
```

---

# 22. Round Timing Details

Recommended timing:

- countdown: 3 seconds total
- optional “GO!” flash: 300–500 ms
- result shown until next round starts

Use either:

- `setInterval` for countdown,
- or a controlled `requestAnimationFrame` timer.

Avoid timing drift if possible.

---

# 23. Robot Reveal Timing

At the exact end of countdown:

1. player move is locked,
2. robot move is generated,
3. robot image is displayed,
4. result is calculated.

Do not reveal the robot choice before the countdown ends.

---

# 24. State Management

Use explicit game states:

```javascript
"idle"
"countdown"
"locked"
"result"
```

This prevents repeated clicks and race conditions.

Example behavior:

- Start button works only in `idle` or `result`.
- During `countdown`, ignore additional Start clicks.
- At `locked/result`, wait until UI is ready before next round.

---

# 25. Acceptance Criteria

The task is complete only if:

- [ ] Start button exists.
- [ ] Countdown lasts about 3 seconds.
- [ ] Webcam gesture detection works.
- [ ] User gesture can be locked at countdown end.
- [ ] Robot move is randomly chosen.
- [ ] Randomness uses a strong unpredictable source.
- [ ] Robot uses the provided images.
- [ ] Rock/Paper/Scissors rules are correct.
- [ ] Result displays WIN / LOSE / DRAW / INVALID.
- [ ] Scoreboard updates correctly.
- [ ] Start button is disabled during countdown.
- [ ] Start button re-enables after round ends.
- [ ] Robot move is hidden until reveal.
- [ ] Robot does not cheat based on player move.
- [ ] UI works for repeated rounds.

---

# 26. Required Final Output from Coding Agent

After implementation, report:

## Architecture

```text
Webcam
→ Gesture Recognition
→ Stable Player Move
→ 3-second Countdown
→ Secure Random Robot Move
→ Rule Evaluation
→ Result UI
```

## Randomness Method

State exactly how randomness is implemented.

Example:

```text
window.crypto.getRandomValues() with modulo selection over 3 choices
```

## Round Logic

Explain:

- when gesture is tracked,
- when gesture is locked,
- when robot move is created,
- when winner is determined.

## Files Changed

List all created / modified files.

## Test Results

Return:

```text
Start button: PASS
3-second countdown: PASS
Gesture lock: PASS
Secure random robot move: PASS
Robot image reveal: PASS
Rock rule: PASS
Paper rule: PASS
Scissors rule: PASS
Scoreboard: PASS
Invalid move handling: PASS
```

---

# 27. Final Instruction

This is a **Rock Paper Scissors vs Robot game**.

The user plays with real hand gestures through the webcam.

The robot uses the provided images as its move visuals.

The round must feel fair, responsive, and unpredictable.

The most important requirements are:

```text
Start button
3-second countdown
stable user gesture lock
secure random robot move
correct RPS logic
robot reveal image
clear win/lose/draw result
scoreboard
```

Implement the full experience cleanly and modularly.
