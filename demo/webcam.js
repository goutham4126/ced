import * as faceapi from '../dist/face-api.esm.js';

const modelPath = '../model/';
const minScore = 0.2;
const maxResults = 5;
let optionsSSDMobileNet;

// anger tracking and alarm
const angryThreshold = 0.3;
const angryDurationLimit = 3000;
const angryState = {};
let alarmPlaying = false;

function playAlarm() {
  const alarm = document.getElementById('alarm');
  if (!alarmPlaying && alarm) {
    alarmPlaying = true;
    alarm.play().catch(() => {});
    setTimeout(() => { alarmPlaying = false; }, 3100);
  }
}

function str(json) {
  let text = '<font color="lightblue">';
  text += json ? JSON.stringify(json).replace(/{|}|"|\[|\]/g, '').replace(/,/g, ', ') : '';
  text += '</font>';
  return text;
}

function log(...txt) {
  console.log(...txt);
  const div = document.getElementById('log');
  if (div) div.innerHTML += `<br>${txt}`;
}

function drawFaces(canvas, data, fps) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.font = 'small-caps 20px "Segoe UI"';
  ctx.fillStyle = 'white';
  ctx.fillText(`FPS: ${fps}`, 10, 25);

  for (const person of data) {
    const box = person.detection.box;
    const angryScore = person.expressions.angry || 0;
    const id = `${Math.round(box.x)}-${Math.round(box.y)}`;
    const now = Date.now();

    // Update angryState
    if (angryScore > angryThreshold) {
      if (!angryState[id]) {
        angryState[id] = { start: now, flagged: false, persistUntil: 0 };
      }
      if (!angryState[id].flagged && now - angryState[id].start > angryDurationLimit) {
        angryState[id].flagged = true;
        angryState[id].persistUntil = now + 3100; // Red box persists during alarm
        playAlarm();
      }
    } else {
      if (angryState[id]?.flagged && now < angryState[id].persistUntil) {
        // Do nothing — still in red persistence window
      } else {
        angryState[id] = null;
      }
    }

    const isAngry = angryState[id]?.flagged && now < angryState[id]?.persistUntil;

    // Draw bounding box
    ctx.lineWidth = 3;
    ctx.strokeStyle = isAngry ? 'red' : 'deepskyblue';
    ctx.fillStyle = isAngry ? 'red' : 'deepskyblue';
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.rect(box.x, box.y, box.width, box.height);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Expression data
    const expression = Object.entries(person.expressions).sort((a, b) => b[1] - a[1]);

    // Top label
    ctx.fillStyle = isAngry ? 'red' : 'black';
    if (isAngry) {
      ctx.fillText('⚠️ Angry > 3s', box.x, box.y - 78);
    }

    ctx.fillText(`gender: ${person.gender}`, box.x, box.y - 59);
    ctx.fillText(`expression: ${Math.round(100 * expression[0][1])}% ${expression[0][0]}`, box.x, box.y - 41);

    // Shadow color label
    ctx.fillStyle = isAngry ? 'red' : 'blue';
    ctx.fillText(`gender: ${person.gender}`, box.x, box.y - 60);
    ctx.fillText(`expression: ${Math.round(100 * expression[0][1])}% ${expression[0][0]}`, box.x, box.y - 42);

    // Face landmarks
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = 'lightblue';
    const pointSize = 2;
    for (let i = 0; i < person.landmarks.positions.length; i++) {
      ctx.beginPath();
      ctx.arc(person.landmarks.positions[i].x, person.landmarks.positions[i].y, pointSize, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}


async function detectVideo(video, canvas) {
  if (!video || video.paused) return false;
  const t0 = performance.now();
  faceapi
    .detectAllFaces(video, optionsSSDMobileNet)
    .withFaceLandmarks()
    .withFaceExpressions()
    .withAgeAndGender()
    .then((result) => {
      const fps = 1000 / (performance.now() - t0);
      drawFaces(canvas, result, fps.toLocaleString());
      requestAnimationFrame(() => detectVideo(video, canvas));
      return true;
    })
    .catch((err) => {
      log(`Detect Error: ${str(err)}`);
      return false;
    });
  return false;
}

async function setupCamera() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  if (!video || !canvas) return null;

  log('Setting up camera');

  if (!navigator.mediaDevices) {
    log('Camera Error: access not supported');
    return null;
  }

  let stream;
  const constraints = { audio: false, video: { facingMode: 'user', resizeMode: 'crop-and-scale' } };
  if (window.innerWidth > window.innerHeight) constraints.video.width = { ideal: window.innerWidth };
  else constraints.video.height = { ideal: window.innerHeight };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    log(`Camera Error: ${err.message || err}`);
    return null;
  }

  if (stream) {
    video.srcObject = stream;
  } else {
    log('Camera Error: stream empty');
    return null;
  }

  const track = stream.getVideoTracks()[0];
  const settings = track.getSettings();
  if (settings.deviceId) delete settings.deviceId;
  if (settings.groupId) delete settings.groupId;
  if (settings.aspectRatio) settings.aspectRatio = Math.trunc(100 * settings.aspectRatio) / 100;

  log(`Camera active: ${track.label}`);
  log(`Camera settings: ${str(settings)}`);

  canvas.addEventListener('click', () => {
    if (video && video.readyState >= 2) {
      if (video.paused) {
        video.play();
        detectVideo(video, canvas);
      } else {
        video.pause();
      }
    }
    log(`Camera state: ${video.paused ? 'paused' : 'playing'}`);
  });

  return new Promise((resolve) => {
    video.onloadeddata = async () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      video.play();
      detectVideo(video, canvas);
      resolve(true);
    };
  });
}

async function setupFaceAPI() {
  await faceapi.nets.ssdMobilenetv1.load(modelPath);
  await faceapi.nets.ageGenderNet.load(modelPath);
  await faceapi.nets.faceLandmark68Net.load(modelPath);
  await faceapi.nets.faceRecognitionNet.load(modelPath);
  await faceapi.nets.faceExpressionNet.load(modelPath);
  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({ minConfidence: minScore, maxResults });
  log(`Models loaded: ${str(faceapi.tf.engine().state.numTensors)} tensors`);
}

async function main() {
  log('FaceAPI WebCam Test');
  await faceapi.tf.setBackend('webgl');
  await faceapi.tf.ready();

  if (faceapi.tf?.env().flagRegistry.CANVAS2D_WILL_READ_FREQUENTLY) faceapi.tf.env().set('CANVAS2D_WILL_READ_FREQUENTLY', true);
  if (faceapi.tf?.env().flagRegistry.WEBGL_EXP_CONV) faceapi.tf.env().set('WEBGL_EXP_CONV', true);

  log(`Version: FaceAPI ${str(faceapi?.version || '(not loaded)')} TensorFlow/JS ${str(faceapi.tf?.version_core || '(not loaded)')} Backend: ${str(faceapi.tf?.getBackend() || '(not loaded)')}`);

  await setupFaceAPI();
  await setupCamera();
}

window.onload = main;
