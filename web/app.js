/**
 * Rock Paper Scissors AI vs Robot
 * 21 Hand Landmarks Geometric Recognition + Battle Arena
 */

import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

(function () {
  'use strict';

  // --- Configuration ---
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index
    [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
    [0, 13], [13, 14], [14, 15], [15, 16],// Ring
    [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
    [5, 9], [9, 13], [13, 17]             // Palm knuckles
  ];

  const ROBOT_IMAGES = {
    rock: 'assets/robot_rock.png',
    paper: 'assets/robot_paper.png',
    scissors: 'assets/robot_scissors.png'
  };

  // Preload robot images into browser cache for instant switching
  Object.values(ROBOT_IMAGES).forEach((src) => {
    const img = new Image();
    img.src = src;
  });

  const GESTURE_THRESHOLD = 0.80; // 80% geometric match score
  const HISTORY_SIZE = 7;         // Temporal smoothing frames
  const INFERENCE_INTERVAL_MS = 33; // ~30 FPS

  // Angle & Distance Thresholds
  const PIP_EXTENDED_MIN = 150.0;
  const DIP_EXTENDED_MIN = 140.0;
  const PIP_FOLDED_MAX = 145.0;
  const DIP_FOLDED_MAX = 150.0;

  const EXTENDED_TIP_DISTANCE_MIN = 0.88;
  const FOLDED_TIP_DISTANCE_MAX = 0.85;

  const V_GAP_MIN = 0.25;
  const V_ANGLE_MIN = 10.0;
  const V_ANGLE_MAX = 80.0;

  // --- DOM Elements ---
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('output-canvas');
  const ctx = canvas.getContext('2d');

  const btnStart = document.getElementById('btn-start');
  const btnStop = document.getElementById('btn-stop');
  const guideBox = document.querySelector('.guide-box');
  const placeholderOverlay = document.getElementById('placeholder-overlay');
  const cameraAlert = document.getElementById('camera-alert');
  const cameraAlertMsg = document.getElementById('camera-alert-msg');
  const fpsBadge = document.getElementById('fps-badge');
  const handBadge = document.getElementById('hand-badge');

  // Status Elements
  const modelDot = document.getElementById('model-dot');
  const modelStatusText = document.getElementById('model-status-text');
  const cameraDot = document.getElementById('camera-dot');
  const cameraStatusText = document.getElementById('camera-status-text');
  const inferenceDot = document.getElementById('inference-dot');
  const inferenceStatusText = document.getElementById('inference-status-text');

  // Prediction Elements
  const gestureBox = document.getElementById('gesture-box');
  const detectedGesture = document.getElementById('detected-gesture');
  const confidenceValue = document.getElementById('confidence-value');
  const probRock = document.getElementById('prob-rock');
  const probPaper = document.getElementById('prob-paper');
  const probScissors = document.getElementById('prob-scissors');
  const barRock = document.querySelector('.rock-bar');
  const barPaper = document.querySelector('.paper-bar');
  const barScissors = document.querySelector('.scissors-bar');

  // Battle Arena Elements
  const btnStartRound = document.getElementById('btn-start-round');
  const countdownDisplay = document.getElementById('countdown-display');
  const countdownHint = document.getElementById('countdown-hint');
  const roundStatusPill = document.getElementById('round-status-pill');
  const roundResultBanner = document.getElementById('round-result-banner');
  const resultTitle = document.getElementById('result-title');
  const resultSubtitle = document.getElementById('result-subtitle');

  const robotImg = document.getElementById('robot-img');
  const robotStandby = document.getElementById('robot-standby');
  const robotMoveText = document.getElementById('robot-move-text');

  const scoreWins = document.getElementById('score-wins');
  const scoreLosses = document.getElementById('score-losses');
  const scoreDraws = document.getElementById('score-draws');
  const scoreInvalid = document.getElementById('score-invalid');

  // Debug Elements
  const dbgIndex = document.getElementById('dbg-index');
  const dbgMiddle = document.getElementById('dbg-middle');
  const dbgRing = document.getElementById('dbg-ring');
  const dbgPinky = document.getElementById('dbg-pinky');
  const dbgThumb = document.getElementById('dbg-thumb');
  const dbgV = document.getElementById('dbg-v');

  // --- State ---
  let handLandmarker = null;
  let stream = null;
  let isRunning = false;
  let animationFrameId = null;
  let lastInferenceTime = 0;
  let history = []; // smoothed gestures
  let scoreHistory = [];
  let frameCount = 0;
  let lastFpsCalcTime = performance.now();

  // Game State
  let gameState = 'idle'; // 'idle' | 'countdown' | 'locked' | 'result'
  let countdownTimer = null;
  const scores = { wins: 0, losses: 0, draws: 0, invalid: 0 };

  // --- Status Helpers ---
  function updateModelStatus(status, text) {
    modelDot.className = `status-dot ${status}`;
    modelStatusText.textContent = text;
  }

  function updateCameraStatus(status, text) {
    cameraDot.className = `status-dot ${status}`;
    cameraStatusText.textContent = text;
  }

  function updateInferenceStatus(status, text) {
    inferenceDot.className = `status-dot ${status}`;
    inferenceStatusText.textContent = text;
  }

  function showAlert(message) {
    cameraAlertMsg.textContent = message;
    cameraAlert.classList.remove('hidden');
  }

  function hideAlert() {
    cameraAlert.classList.add('hidden');
    cameraAlertMsg.textContent = '';
  }

  // --- MediaPipe HandLandmarker Initialization ---
  async function initHandLandmarker() {
    updateModelStatus('loading', 'Loading...');
    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );

      try {
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 1
        });
      } catch (gpuErr) {
        console.warn('GPU delegate failed, falling back to CPU:', gpuErr);
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
          },
          runningMode: 'VIDEO',
          numHands: 1
        });
      }

      console.log('MediaPipe HandLandmarker initialized successfully.');
      updateModelStatus('active', 'Ready');
      btnStart.disabled = false;
    } catch (err) {
      console.error('Failed to initialize HandLandmarker:', err);
      updateModelStatus('error', 'Error');
      showAlert(`Tracker error: ${err.message || err}.`);
    }
  }

  // --- Geometry Utilities ---
  function dist3d(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = (a.z || 0) - (b.z || 0);
    return Math.hypot(dx, dy, dz);
  }

  function angle3points(a, b, c) {
    const v1 = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
    const v2 = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };

    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const mag1 = Math.hypot(v1.x, v1.y, v1.z);
    const mag2 = Math.hypot(v2.x, v2.y, v2.z);

    if (mag1 * mag2 === 0) return 0;
    const cos = Math.max(-1.0, Math.min(1.0, dot / (mag1 * mag2)));
    return Math.acos(cos) * (180.0 / Math.PI);
  }

  function calculatePalmCenter(lm) {
    return {
      x: (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5,
      y: (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5,
      z: ((lm[0].z || 0) + (lm[5].z || 0) + (lm[9].z || 0) + (lm[13].z || 0) + (lm[17].z || 0)) / 5
    };
  }

  // --- Gesture Recognition Analysis ---
  function analyzeHand(lm) {
    const palmCenter = calculatePalmCenter(lm);
    const palmSize = dist3d(lm[0], lm[9]) || 0.001;

    // Joint Angles
    const indexPip = angle3points(lm[5], lm[6], lm[7]);
    const indexDip = angle3points(lm[6], lm[7], lm[8]);

    const middlePip = angle3points(lm[9], lm[10], lm[11]);
    const middleDip = angle3points(lm[10], lm[11], lm[12]);

    const ringPip = angle3points(lm[13], lm[14], lm[15]);
    const ringDip = angle3points(lm[14], lm[15], lm[16]);

    const pinkyPip = angle3points(lm[17], lm[18], lm[19]);
    const pinkyDip = angle3points(lm[18], lm[19], lm[20]);

    const thumbIp = angle3points(lm[2], lm[3], lm[4]);

    // Normalized Tip-to-Palm Distances
    const thumbTipDist = dist3d(lm[4], palmCenter) / palmSize;
    const indexTipDist = dist3d(lm[8], palmCenter) / palmSize;
    const middleTipDist = dist3d(lm[12], palmCenter) / palmSize;
    const ringTipDist = dist3d(lm[16], palmCenter) / palmSize;
    const pinkyTipDist = dist3d(lm[20], palmCenter) / palmSize;

    // Finger States
    const indexExtended = indexPip >= PIP_EXTENDED_MIN && indexDip >= DIP_EXTENDED_MIN && indexTipDist >= EXTENDED_TIP_DISTANCE_MIN;
    const middleExtended = middlePip >= PIP_EXTENDED_MIN && middleDip >= DIP_EXTENDED_MIN && middleTipDist >= EXTENDED_TIP_DISTANCE_MIN;
    const ringExtended = ringPip >= PIP_EXTENDED_MIN && ringDip >= DIP_EXTENDED_MIN && ringTipDist >= EXTENDED_TIP_DISTANCE_MIN;
    const pinkyExtended = pinkyPip >= PIP_EXTENDED_MIN && pinkyDip >= DIP_EXTENDED_MIN && pinkyTipDist >= EXTENDED_TIP_DISTANCE_MIN;
    const thumbExtended = thumbIp >= 135.0 && thumbTipDist >= 0.80;

    const indexFolded = (indexPip <= PIP_FOLDED_MAX || indexDip <= DIP_FOLDED_MAX) && indexTipDist <= FOLDED_TIP_DISTANCE_MAX;
    const middleFolded = (middlePip <= PIP_FOLDED_MAX || middleDip <= DIP_FOLDED_MAX) && middleTipDist <= FOLDED_TIP_DISTANCE_MAX;
    const ringFolded = (ringPip <= PIP_FOLDED_MAX || ringDip <= DIP_FOLDED_MAX) && ringTipDist <= FOLDED_TIP_DISTANCE_MAX;
    const pinkyFolded = (pinkyPip <= PIP_FOLDED_MAX || pinkyDip <= DIP_FOLDED_MAX) && pinkyTipDist <= FOLDED_TIP_DISTANCE_MAX;
    const thumbCompact = thumbTipDist <= 0.85;

    // V-Shape for Scissors
    const vGap = dist3d(lm[8], lm[12]) / palmSize;
    const vAnchor = {
      x: (lm[5].x + lm[9].x) / 2,
      y: (lm[5].y + lm[9].y) / 2,
      z: ((lm[5].z || 0) + (lm[9].z || 0)) / 2
    };
    const vAngle = angle3points(lm[8], vAnchor, lm[12]);
    const validV = vGap >= V_GAP_MIN && vAngle >= V_ANGLE_MIN && vAngle <= V_ANGLE_MAX;

    // 1. PAPER Score
    let paperScore = (
      (indexExtended ? 2 : 0) +
      (middleExtended ? 2 : 0) +
      (ringExtended ? 2 : 0) +
      (pinkyExtended ? 2 : 0) +
      (thumbExtended ? 1.5 : 0) +
      ((indexTipDist >= 0.9 && middleTipDist >= 0.9 && ringTipDist >= 0.9 && pinkyTipDist >= 0.9) ? 1.5 : 0)
    ) / 11.0;

    // 2. ROCK Score
    const meanTipDist = (indexTipDist + middleTipDist + ringTipDist + pinkyTipDist) / 4.0;
    let rockScore = (
      (indexFolded ? 2 : 0) +
      (middleFolded ? 2 : 0) +
      (ringFolded ? 2 : 0) +
      (pinkyFolded ? 2 : 0) +
      (meanTipDist <= 0.65 ? 1.5 : (meanTipDist <= 0.75 ? 0.8 : 0)) +
      (thumbCompact ? 1.0 : 0)
    ) / 10.5;

    // Hard reject rock if any finger is clearly extended
    const extendedCount = (indexExtended ? 1 : 0) + (middleExtended ? 1 : 0) + (ringExtended ? 1 : 0) + (pinkyExtended ? 1 : 0);
    if (extendedCount >= 1) {
      rockScore = Math.min(rockScore, 0.35);
    }

    // 3. SCISSORS Score
    let scissorsScore = (
      (indexExtended ? 2.5 : 0) +
      (middleExtended ? 2.5 : 0) +
      (ringFolded ? 2.5 : 0) +
      (pinkyFolded ? 2.5 : 0) +
      (validV ? 2.0 : (vGap >= 0.20 ? 1.0 : 0)) +
      (thumbCompact || thumbExtended ? 0.5 : 0)
    ) / 12.5;

    // Hard reject scissors if ring or pinky is extended, or index or middle is folded
    if (ringExtended || pinkyExtended || indexFolded || middleFolded || extendedCount !== 2) {
      scissorsScore = Math.min(scissorsScore, 0.30);
    }

    // Determine winning candidate
    let gesture = 'UNKNOWN';
    let confidence = 0;

    const paperMandatory = indexExtended && middleExtended && ringExtended && pinkyExtended;
    const rockMandatory = indexFolded && middleFolded && ringFolded && pinkyFolded && extendedCount === 0;
    const scissorsMandatory = indexExtended && middleExtended && ringFolded && pinkyFolded && validV;

    if (paperMandatory && paperScore >= GESTURE_THRESHOLD && paperScore > rockScore && paperScore > scissorsScore) {
      gesture = 'PAPER';
      confidence = paperScore;
    } else if (rockMandatory && rockScore >= GESTURE_THRESHOLD && rockScore > paperScore && rockScore > scissorsScore) {
      gesture = 'ROCK';
      confidence = rockScore;
    } else if (scissorsMandatory && scissorsScore >= GESTURE_THRESHOLD && scissorsScore > paperScore && scissorsScore > rockScore) {
      gesture = 'SCISSORS';
      confidence = scissorsScore;
    }

    return {
      gesture,
      confidence,
      scores: { rock: rockScore, paper: paperScore, scissors: scissorsScore },
      telemetry: {
        index: `${indexExtended ? 'EXT' : (indexFolded ? 'FLD' : 'MID')} (P:${indexPip.toFixed(0)}° D:${indexDip.toFixed(0)}° R:${indexTipDist.toFixed(2)})`,
        middle: `${middleExtended ? 'EXT' : (middleFolded ? 'FLD' : 'MID')} (P:${middlePip.toFixed(0)}° D:${middleDip.toFixed(0)}° R:${middleTipDist.toFixed(2)})`,
        ring: `${ringExtended ? 'EXT' : (ringFolded ? 'FLD' : 'MID')} (P:${ringPip.toFixed(0)}° D:${ringDip.toFixed(0)}° R:${ringTipDist.toFixed(2)})`,
        pinky: `${pinkyExtended ? 'EXT' : (pinkyFolded ? 'FLD' : 'MID')} (P:${pinkyPip.toFixed(0)}° D:${pinkyDip.toFixed(0)}° R:${pinkyTipDist.toFixed(2)})`,
        thumb: `${thumbExtended ? 'EXT' : (thumbCompact ? 'CMP' : 'MID')} (R:${thumbTipDist.toFixed(2)})`,
        v: `Gap:${vGap.toFixed(2)} Ang:${vAngle.toFixed(0)}°`
      }
    };
  }

  // --- Secure Random & Game Logic ---
  function getSecureRandomMove() {
    const array = new Uint32Array(1);
    window.crypto.getRandomValues(array);
    const choices = ['rock', 'paper', 'scissors'];
    return choices[array[0] % 3];
  }

  function getRoundResult(playerMove, robotMove) {
    if (!['rock', 'paper', 'scissors'].includes(playerMove)) return 'invalid';
    if (playerMove === robotMove) return 'draw';
    if (
      (playerMove === 'rock' && robotMove === 'scissors') ||
      (playerMove === 'paper' && robotMove === 'rock') ||
      (playerMove === 'scissors' && robotMove === 'paper')
    ) {
      return 'win';
    }
    return 'lose';
  }

  function getStablePlayerGesture() {
    if (history.length === 0) return 'unknown';
    const counts = {};
    for (const g of history) {
      counts[g] = (counts[g] || 0) + 1;
    }
    let dominant = 'UNKNOWN';
    let maxCount = 0;
    for (const [g, c] of Object.entries(counts)) {
      if (c > maxCount) {
        maxCount = c;
        dominant = g;
      }
    }
    // Require at least 4 out of last 7 frames
    if (maxCount >= 4 && dominant !== 'UNKNOWN') {
      return dominant.toLowerCase();
    }
    return 'unknown';
  }

  function startRound() {
    if (gameState === 'countdown') return;
    if (!isRunning) {
      showAlert('Please start the camera first before playing!');
      return;
    }

    gameState = 'countdown';
    btnStartRound.disabled = true;
    roundResultBanner.classList.add('hidden');

    robotImg.classList.add('hidden');
    robotStandby.classList.remove('hidden');
    robotMoveText.textContent = 'Thinking...';

    roundStatusPill.textContent = 'Battle in progress';
    countdownHint.textContent = 'Hold your gesture in front of the camera!';

    startCountdown();
  }

  function startCountdown() {
    let count = 3;
    updateCountdownDisplay(count);

    if (countdownTimer) clearInterval(countdownTimer);

    countdownTimer = setInterval(() => {
      count--;
      if (count > 0) {
        updateCountdownDisplay(count);
      } else if (count === 0) {
        updateCountdownDisplay('FIGHT!');
        countdownDisplay.classList.add('fight');
        clearInterval(countdownTimer);
        countdownTimer = null;
        resolveRound(); // Instant resolution and image reveal
      }
    }, 1000);
  }

  function updateCountdownDisplay(val) {
    countdownDisplay.textContent = val;
    countdownDisplay.classList.remove('animating', 'fight');
    void countdownDisplay.offsetWidth; // trigger reflow
    countdownDisplay.classList.add('animating');
  }

  function resolveRound() {
    gameState = 'locked';
    countdownDisplay.classList.remove('animating');

    const playerMove = getStablePlayerGesture();

    if (playerMove === 'unknown') {
      // Invalid round
      scores.invalid++;
      roundResultBanner.className = 'round-result-banner invalid';
      resultTitle.textContent = 'INVALID MOVE';
      resultSubtitle.textContent = 'Could not detect a stable gesture. Try holding hand clearly in view.';
      roundResultBanner.classList.remove('hidden');

      robotMoveText.textContent = 'Standby';
    } else {
      // Valid round - generate robot move
      const robotMove = getSecureRandomMove();
      updateRobotDisplay(robotMove);

      const result = getRoundResult(playerMove, robotMove);

      if (result === 'win') {
        scores.wins++;
        roundResultBanner.className = 'round-result-banner win';
        resultTitle.textContent = 'YOU WIN!';
        resultSubtitle.textContent = `${playerMove.toUpperCase()} beats ${robotMove.toUpperCase()}`;
      } else if (result === 'lose') {
        scores.losses++;
        roundResultBanner.className = 'round-result-banner lose';
        resultTitle.textContent = 'YOU LOSE!';
        resultSubtitle.textContent = `${robotMove.toUpperCase()} beats ${playerMove.toUpperCase()}`;
      } else {
        scores.draws++;
        roundResultBanner.className = 'round-result-banner draw';
        resultTitle.textContent = 'DRAW!';
        resultSubtitle.textContent = `Both played ${playerMove.toUpperCase()}`;
      }
      roundResultBanner.classList.remove('hidden');
    }

    updateScoreboard();

    // Re-enable for next round
    gameState = 'result';
    btnStartRound.disabled = false;
    btnStartRound.querySelector('span').textContent = 'Play Again (3s)';
    roundStatusPill.textContent = 'Round Finished';
    countdownHint.textContent = 'Click Play Again to start another round!';
  }

  function updateRobotDisplay(robotMove) {
    robotImg.src = ROBOT_IMAGES[robotMove];
    robotImg.alt = `Robot ${robotMove}`;
    robotImg.classList.remove('hidden');
    robotStandby.classList.add('hidden');
    robotMoveText.textContent = robotMove.toUpperCase();
  }

  function updateScoreboard() {
    scoreWins.textContent = scores.wins;
    scoreLosses.textContent = scores.losses;
    scoreDraws.textContent = scores.draws;
    scoreInvalid.textContent = scores.invalid;
  }

  function resetRoundUI() {
    gameState = 'idle';
    if (countdownTimer) {
      clearInterval(countdownTimer);
      countdownTimer = null;
    }
    countdownDisplay.textContent = 'VS';
    countdownDisplay.classList.remove('animating', 'fight');
    countdownHint.textContent = 'Show your hand gesture to the camera!';
    roundResultBanner.classList.add('hidden');
    robotImg.classList.add('hidden');
    robotStandby.classList.remove('hidden');
    robotMoveText.textContent = 'Waiting...';
    roundStatusPill.textContent = 'Ready';
    btnStartRound.disabled = !isRunning;
    btnStartRound.querySelector('span').textContent = 'Start Battle (3s)';
  }

  // --- Camera Management ---
  async function startCamera() {
    hideAlert();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showAlert('Webcam is not supported on this browser or context.');
      return;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });

      video.srcObject = stream;
      await new Promise((resolve) => {
        video.onloadedmetadata = () => {
          video.play();
          resolve();
        };
      });

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      isRunning = true;
      placeholderOverlay.classList.add('hidden');
      guideBox.classList.add('active');
      btnStart.disabled = true;
      btnStop.disabled = false;
      btnStartRound.disabled = false;

      updateCameraStatus('active', 'Active');
      updateInferenceStatus('active', 'Running');

      history = [];
      scoreHistory = [];
      lastInferenceTime = performance.now();
      lastFpsCalcTime = performance.now();
      frameCount = 0;

      predictLoop();
    } catch (err) {
      console.error('Camera access error:', err);
      updateCameraStatus('error', 'Denied');
      showAlert(`Unable to access camera: ${err.message || err.name}`);
    }
  }

  function stopCamera() {
    isRunning = false;
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }

    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    placeholderOverlay.classList.remove('hidden');
    guideBox.classList.remove('active');
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnStartRound.disabled = true;

    updateCameraStatus('idle', 'Inactive');
    updateInferenceStatus('idle', 'Idle');
    fpsBadge.textContent = '0 FPS';
    handBadge.textContent = 'No Hand';

    resetPredictions();
    resetRoundUI();
  }

  function resetPredictions() {
    detectedGesture.textContent = 'Waiting...';
    confidenceValue.textContent = '0.0%';
    gestureBox.className = 'gesture-result-box';

    probRock.textContent = '0.0%';
    probPaper.textContent = '0.0%';
    probScissors.textContent = '0.0%';

    barRock.style.width = '0%';
    barPaper.style.width = '0%';
    barScissors.style.width = '0%';

    dbgIndex.textContent = '--';
    dbgMiddle.textContent = '--';
    dbgRing.textContent = '--';
    dbgPinky.textContent = '--';
    dbgThumb.textContent = '--';
    dbgV.textContent = '--';
  }

  // --- Real-Time Loop ---
  function predictLoop() {
    if (!isRunning) return;

    const now = performance.now();
    const elapsed = now - lastInferenceTime;

    if (elapsed >= INFERENCE_INTERVAL_MS && video.readyState >= 2) {
      lastInferenceTime = now;
      processFrame(now);

      frameCount++;
      const fpsElapsed = now - lastFpsCalcTime;
      if (fpsElapsed >= 1000) {
        const currentFps = Math.round((frameCount * 1000) / fpsElapsed);
        fpsBadge.textContent = `${currentFps} FPS`;
        frameCount = 0;
        lastFpsCalcTime = now;
      }
    }

    animationFrameId = requestAnimationFrame(predictLoop);
  }

  function processFrame(timestamp) {
    if (!handLandmarker) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const results = handLandmarker.detectForVideo(video, timestamp);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results && results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      const handedness = (results.handedness && results.handedness[0] && results.handedness[0][0])
        ? results.handedness[0][0].displayName
        : 'Hand';

      handBadge.textContent = `${handedness} (21/21)`;

      renderSkeleton(landmarks);

      const analysis = analyzeHand(landmarks);

      history.push(analysis.gesture);
      scoreHistory.push(analysis.scores);
      if (history.length > HISTORY_SIZE) history.shift();
      if (scoreHistory.length > HISTORY_SIZE) scoreHistory.shift();

      const counts = {};
      for (const g of history) counts[g] = (counts[g] || 0) + 1;
      let dominantGesture = 'UNKNOWN';
      let maxCount = 0;
      for (const [g, c] of Object.entries(counts)) {
        if (c > maxCount) {
          maxCount = c;
          dominantGesture = g;
        }
      }

      const avgScores = { rock: 0, paper: 0, scissors: 0 };
      for (const s of scoreHistory) {
        avgScores.rock += s.rock;
        avgScores.paper += s.paper;
        avgScores.scissors += s.scissors;
      }
      avgScores.rock /= scoreHistory.length;
      avgScores.paper /= scoreHistory.length;
      avgScores.scissors /= scoreHistory.length;

      const displayConfidence = dominantGesture !== 'UNKNOWN' ? (avgScores[dominantGesture.toLowerCase()] || 0) : 0;

      updateUI(dominantGesture, displayConfidence, avgScores, analysis.telemetry);
    } else {
      handBadge.textContent = 'No Hand';
      history = [];
      scoreHistory = [];
      updateUI('UNKNOWN', 0, { rock: 0, paper: 0, scissors: 0 }, null);
    }
  }

  // --- Rendering ---
  function renderSkeleton(landmarks) {
    const w = canvas.width;
    const h = canvas.height;

    ctx.strokeStyle = 'rgba(59, 130, 246, 0.85)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (const [i, j] of HAND_CONNECTIONS) {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      ctx.moveTo(p1.x * w, p1.y * h);
      ctx.lineTo(p2.x * w, p2.y * h);
    }
    ctx.stroke();

    ctx.font = '10px monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';

    const tipIndices = new Set([4, 8, 12, 16, 20]);

    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      const x = lm.x * w;
      const y = lm.y * h;

      ctx.beginPath();
      ctx.arc(x, y, tipIndices.has(i) ? 5 : 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = tipIndices.has(i) ? '#ef4444' : '#10b981';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = '#fef08a';
      ctx.fillText(i.toString(), x + 7, y - 5);
    }
  }

  // --- UI Updates ---
  function updateUI(gesture, confidence, scores, telemetry) {
    const rockPct = (scores.rock * 100).toFixed(1);
    const paperPct = (scores.paper * 100).toFixed(1);
    const scissorsPct = (scores.scissors * 100).toFixed(1);

    probRock.textContent = `${rockPct}%`;
    probPaper.textContent = `${paperPct}%`;
    probScissors.textContent = `${scissorsPct}%`;

    barRock.style.width = `${Math.min(100, scores.rock * 100)}%`;
    barPaper.style.width = `${Math.min(100, scores.paper * 100)}%`;
    barScissors.style.width = `${Math.min(100, scores.scissors * 100)}%`;

    if (gesture !== 'UNKNOWN') {
      const className = gesture.toLowerCase();
      detectedGesture.textContent = gesture;
      confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
      gestureBox.className = `gesture-result-box ${className}`;
    } else {
      detectedGesture.textContent = 'Not Sure';
      confidenceValue.textContent = '0.0%';
      gestureBox.className = 'gesture-result-box';
    }

    if (telemetry) {
      dbgIndex.textContent = telemetry.index;
      dbgMiddle.textContent = telemetry.middle;
      dbgRing.textContent = telemetry.ring;
      dbgPinky.textContent = telemetry.pinky;
      dbgThumb.textContent = telemetry.thumb;
      dbgV.textContent = telemetry.v;
    }
  }

  // --- Event Listeners ---
  btnStart.addEventListener('click', startCamera);
  btnStop.addEventListener('click', stopCamera);
  btnStartRound.addEventListener('click', startRound);

  window.addEventListener('beforeunload', () => {
    if (isRunning) stopCamera();
  });

  initHandLandmarker();
})();
