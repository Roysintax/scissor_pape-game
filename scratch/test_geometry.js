// scratch/test_geometry.js
// Verification of geometric rules for Rock, Paper, Scissors

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

const PIP_EXTENDED_MIN = 150.0;
const DIP_EXTENDED_MIN = 140.0;
const PIP_FOLDED_MAX = 145.0;
const DIP_FOLDED_MAX = 150.0;

const EXTENDED_TIP_DISTANCE_MIN = 0.88;
const FOLDED_TIP_DISTANCE_MAX = 0.85;

const V_GAP_MIN = 0.25;
const V_ANGLE_MIN = 10.0;
const V_ANGLE_MAX = 80.0;
const GESTURE_THRESHOLD = 0.80;

function analyzeHand(lm) {
  const palmCenter = calculatePalmCenter(lm);
  const palmSize = dist3d(lm[0], lm[9]) || 0.001;

  const indexPip = angle3points(lm[5], lm[6], lm[7]);
  const indexDip = angle3points(lm[6], lm[7], lm[8]);

  const middlePip = angle3points(lm[9], lm[10], lm[11]);
  const middleDip = angle3points(lm[10], lm[11], lm[12]);

  const ringPip = angle3points(lm[13], lm[14], lm[15]);
  const ringDip = angle3points(lm[14], lm[15], lm[16]);

  const pinkyPip = angle3points(lm[17], lm[18], lm[19]);
  const pinkyDip = angle3points(lm[18], lm[19], lm[20]);

  const thumbIp = angle3points(lm[2], lm[3], lm[4]);

  const thumbTipDist = dist3d(lm[4], palmCenter) / palmSize;
  const indexTipDist = dist3d(lm[8], palmCenter) / palmSize;
  const middleTipDist = dist3d(lm[12], palmCenter) / palmSize;
  const ringTipDist = dist3d(lm[16], palmCenter) / palmSize;
  const pinkyTipDist = dist3d(lm[20], palmCenter) / palmSize;

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

  if (ringExtended || pinkyExtended || indexFolded || middleFolded || extendedCount !== 2) {
    scissorsScore = Math.min(scissorsScore, 0.30);
  }

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

  return { gesture, confidence, scores: { rock: rockScore, paper: paperScore, scissors: scissorsScore } };
}

// Generate test hands
function createHand(fingerStates) {
  // fingerStates: { thumb: 'ext'|'cmp', index: 'ext'|'fld', middle: 'ext'|'fld', ring: 'ext'|'fld', pinky: 'ext'|'fld', vSeparated: bool }
  const lm = [];
  // 0: Wrist
  lm[0] = { x: 0.5, y: 0.8, z: 0 };

  // Knuckles (MCP)
  lm[1] = { x: 0.42, y: 0.72, z: 0 }; // Thumb CMC
  lm[2] = { x: 0.38, y: 0.65, z: 0 }; // Thumb MCP
  lm[5] = { x: 0.42, y: 0.55, z: 0 }; // Index MCP
  lm[9] = { x: 0.50, y: 0.52, z: 0 }; // Middle MCP
  lm[13] = { x: 0.58, y: 0.55, z: 0 }; // Ring MCP
  lm[17] = { x: 0.65, y: 0.60, z: 0 }; // Pinky MCP

  // Thumb
  if (fingerStates.thumb === 'ext') {
    lm[3] = { x: 0.32, y: 0.58, z: 0 };
    lm[4] = { x: 0.25, y: 0.50, z: 0 }; // Far from palm
  } else {
    lm[3] = { x: 0.42, y: 0.60, z: 0 };
    lm[4] = { x: 0.45, y: 0.58, z: 0 }; // Compact
  }

  // Fingers helper
  function setupFinger(mcpIdx, pipIdx, dipIdx, tipIdx, state, xOffset, xTipSpread = 0) {
    const base = lm[mcpIdx];
    if (state === 'ext') {
      lm[pipIdx] = { x: base.x + xOffset * 0.3, y: base.y - 0.12, z: 0 };
      lm[dipIdx] = { x: base.x + xOffset * 0.6, y: base.y - 0.22, z: 0 };
      lm[tipIdx] = { x: base.x + xOffset + xTipSpread, y: base.y - 0.32, z: 0 };
    } else {
      // Folded back towards palm
      lm[pipIdx] = { x: base.x + xOffset * 0.2, y: base.y - 0.08, z: 0 };
      lm[dipIdx] = { x: base.x, y: base.y - 0.02, z: -0.05 };
      lm[tipIdx] = { x: base.x, y: base.y + 0.04, z: -0.02 };
    }
  }

  const vSpread = fingerStates.vSeparated ? 0.06 : 0;
  setupFinger(5, 6, 7, 8, fingerStates.index, -0.02, -vSpread);
  setupFinger(9, 10, 11, 12, fingerStates.middle, 0, vSpread);
  setupFinger(13, 14, 15, 16, fingerStates.ring, 0.02);
  setupFinger(17, 18, 19, 20, fingerStates.pinky, 0.04);

  return lm;
}

// Test cases
const paperHand = createHand({ thumb: 'ext', index: 'ext', middle: 'ext', ring: 'ext', pinky: 'ext' });
const rockHand = createHand({ thumb: 'cmp', index: 'fld', middle: 'fld', ring: 'fld', pinky: 'fld' });
const scissorsHand = createHand({ thumb: 'cmp', index: 'ext', middle: 'ext', ring: 'fld', pinky: 'fld', vSeparated: true });
const oneFingerHand = createHand({ thumb: 'cmp', index: 'ext', middle: 'fld', ring: 'fld', pinky: 'fld' });
const threeFingerHand = createHand({ thumb: 'cmp', index: 'ext', middle: 'ext', ring: 'ext', pinky: 'fld' });

console.log('--- TEST RESULTS ---');
console.log('PAPER Hand:', analyzeHand(paperHand));
console.log('ROCK Hand:', analyzeHand(rockHand));
console.log('SCISSORS Hand:', analyzeHand(scissorsHand));
console.log('1-Finger Hand:', analyzeHand(oneFingerHand));
console.log('3-Finger Hand:', analyzeHand(threeFingerHand));

// Assertions
const rPaper = analyzeHand(paperHand);
const rRock = analyzeHand(rockHand);
const rScissors = analyzeHand(scissorsHand);
const rOne = analyzeHand(oneFingerHand);
const rThree = analyzeHand(threeFingerHand);

let allPass = true;
function assert(desc, condition) {
  if (condition) {
    console.log(`[PASS] ${desc}`);
  } else {
    console.error(`[FAIL] ${desc}`);
    allPass = false;
  }
}

assert('Paper hand detected as PAPER', rPaper.gesture === 'PAPER' && rPaper.confidence >= 0.80);
assert('Rock hand detected as ROCK', rRock.gesture === 'ROCK' && rRock.confidence >= 0.80);
assert('Scissors hand detected as SCISSORS', rScissors.gesture === 'SCISSORS' && rScissors.confidence >= 0.80);
assert('1-Finger rejected (not Scissors, not Paper, not Rock)', rOne.gesture === 'UNKNOWN');
assert('3-Finger rejected (not Scissors, not Paper, not Rock)', rThree.gesture === 'UNKNOWN');

if (!allPass) process.exit(1);
console.log('ALL GEOMETRY TESTS PASSED!');
