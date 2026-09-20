// scratch/test_game_logic.js
const fs = require('fs');
const path = require('path');

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

console.log('--- Testing Game Rules ---');
const tests = [
  ['rock', 'scissors', 'win'],
  ['paper', 'rock', 'win'],
  ['scissors', 'paper', 'win'],
  ['rock', 'paper', 'lose'],
  ['paper', 'scissors', 'lose'],
  ['scissors', 'rock', 'lose'],
  ['rock', 'rock', 'draw'],
  ['paper', 'paper', 'draw'],
  ['scissors', 'scissors', 'draw'],
  ['unknown', 'rock', 'invalid'],
  ['invalid', 'scissors', 'invalid']
];

let allPass = true;
tests.forEach(([p, r, expected]) => {
  const actual = getRoundResult(p, r);
  if (actual === expected) {
    console.log(`[PASS] ${p} vs ${r} => ${actual}`);
  } else {
    console.error(`[FAIL] ${p} vs ${r} => ${actual} (expected: ${expected})`);
    allPass = false;
  }
});

console.log('\n--- Checking Assets ---');
const assets = [
  'web/assets/robot_rock.png',
  'web/assets/robot_paper.png',
  'web/assets/robot_scissors.png'
];

assets.forEach(assetPath => {
  const fullPath = path.resolve(__dirname, '..', assetPath);
  if (fs.existsSync(fullPath)) {
    console.log(`[PASS] Asset exists: ${assetPath} (${fs.statSync(fullPath).size} bytes)`);
  } else {
    console.error(`[FAIL] Asset missing: ${assetPath}`);
    allPass = false;
  }
});

if (!allPass) process.exit(1);
console.log('\nALL GAME LOGIC & ASSET TESTS PASSED!');
