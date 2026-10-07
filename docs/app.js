// MediaPipe 웹캠 데모 — 파이썬 스크립트(webcam_*.py)를 브라우저용으로 옮긴 버전
import {
  FilesetResolver, HandLandmarker, GestureRecognizer, FaceLandmarker, DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const MODEL_URL = {
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
  gesture: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
};
const CUSTOM_MODEL_URL = "custom_gesture_model.json";  // export_web_model.py 로 생성

const MIN_GESTURE_SCORE = 0.5;  // 기본 제스처 표시 임계값
const MIN_PROB = 0.7;           // 커스텀 제스처: 이보다 낮으면 "?"
const SMOOTH_FRAMES = 5;        // 커스텀 제스처: 최근 N프레임 다수결
const BLINK_THRESHOLD = 0.5;
const MOUTH_THRESHOLD = 0.3;
const TOP_BLENDSHAPES = 8;

const GESTURE_TEXT = {
  Closed_Fist: "✊ Fist", Open_Palm: "🖐 Open Palm", Pointing_Up: "☝️ Pointing Up",
  Thumb_Down: "👎 Thumbs Down", Thumb_Up: "👍 Thumbs Up", Victory: "✌️ Victory", ILoveYou: "🤟 I Love You",
};
// 커스텀 제스처 라벨 → 이모지 (webcam_custom_gesture.py 의 EMOJI 와 동일)
const EMOJI = { korean_heart: "❤️", ok: "👌" };
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const draw = new DrawingUtils(ctx);
const $status = document.getElementById("status");
const $fps = document.getElementById("fps");
const $info = document.getElementById("info");
const $controls = document.getElementById("controls");
const $overlay = document.getElementById("overlay");

let fileset = null;
const tasks = {};          // 만든 모델 캐시 (탭 바꿀 때 재사용)
let customModel = null;
let mode = "hand";
let lastTs = -1;
let prevTime = performance.now();
const history = { Left: [], Right: [] };
const options = { mesh: false };

// ---------- 모델 로딩 ----------
async function createTask(TaskClass, opts) {
  // GPU가 안 되는 환경이면 CPU로 재시도
  for (const delegate of ["GPU", "CPU"]) {
    try {
      return await TaskClass.createFromOptions(fileset, {
        ...opts, baseOptions: { ...opts.baseOptions, delegate }, runningMode: "VIDEO",
      });
    } catch (e) {
      if (delegate === "CPU") throw e;
      console.warn("GPU delegate 실패, CPU로 재시도", e);
    }
  }
}

async function getTask(name) {
  if (tasks[name]) return tasks[name];
  $status.textContent = "모델 불러오는 중…";
  fileset ??= await FilesetResolver.forVisionTasks(WASM_URL);
  const conf = { minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 };
  if (name === "hand" || name === "custom") {
    tasks[name] = await createTask(HandLandmarker, { baseOptions: { modelAssetPath: MODEL_URL.hand }, numHands: 2, ...conf });
  } else if (name === "gesture") {
    tasks[name] = await createTask(GestureRecognizer, { baseOptions: { modelAssetPath: MODEL_URL.gesture }, numHands: 2, ...conf });
  } else if (name === "face") {
    tasks[name] = await createTask(FaceLandmarker, {
      baseOptions: { modelAssetPath: MODEL_URL.face }, numFaces: 1, outputFaceBlendshapes: true,
      minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
  }
  if (name === "custom" && !customModel) {
    customModel = await (await fetch(CUSTOM_MODEL_URL)).json();
  }
  $status.textContent = "실행 중";
  return tasks[name];
}

// ---------- 커스텀 제스처 (gesture_common.py / webcam_custom_gesture.py 와 같은 계산) ----------
function landmarksToFeatures(landmarks, handedness, w, h) {
  const pts = landmarks.map(p => [p.x * (w / h), p.y, p.z]);
  const [ox, oy, oz] = pts[0];
  for (const p of pts) { p[0] -= ox; p[1] -= oy; p[2] -= oz; }
  const scale = Math.max(...pts.map(([x, y]) => Math.hypot(x, y)));
  if (scale > 0) for (const p of pts) { p[0] /= scale; p[1] /= scale; p[2] /= scale; }
  if (handedness === "Left") for (const p of pts) p[0] *= -1;
  return pts.flat();
}

function predictProba(features) {
  const { mean, scale, layers } = customModel;
  let x = features.map((v, i) => (v - mean[i]) / scale[i]);
  layers.forEach(({ W, b }, li) => {
    const out = b.slice();
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = W[i];
      for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
    }
    x = li < layers.length - 1 ? out.map(v => Math.max(v, 0)) : out;  // 은닉층 ReLU
  });
  const m = Math.max(...x);
  const e = x.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);  // softmax
}

function mostCommon(arr) {
  const c = {};
  for (const v of arr) c[v] = (c[v] || 0) + 1;
  return Object.entries(c).sort((a, b) => b[1] - a[1])[0][0];
}

// ---------- 그리기 ----------
function drawHand(landmarks) {
  draw.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#ffffff", lineWidth: 2 });
  draw.drawLandmarks(landmarks, {
    color: "#00000000", lineWidth: 0, radius: 4,
    fillColor: d => (FINGERTIPS.has(d.index) ? "#ff3030" : "#30e030"),
  });
  const xs = landmarks.map(p => p.x * canvas.width);
  const ys = landmarks.map(p => p.y * canvas.height);
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys) };
}

function label(text, x, y, color = "#ffe040") {
  ctx.font = "bold 20px system-ui, 'Segoe UI', sans-serif";
  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(0,0,0,.7)";
  ctx.strokeText(text, x, Math.max(y, 24));
  ctx.fillStyle = color;
  ctx.fillText(text, x, Math.max(y, 24));
}

function bars(rows) {
  return rows.map(([name, score]) =>
    `<div class="bar-row"><span title="${name}">${name}</span><div class="bar"><i style="width:${(score * 100).toFixed(0)}%"></i></div><span>${score.toFixed(2)}</span></div>`,
  ).join("");
}

// ---------- 모드별 처리 ----------
function runHand(task, ts) {
  const r = task.detectForVideo(canvas, ts);
  const hd = r.handedness ?? r.handednesses;
  r.landmarks.forEach((lm, i) => {
    const box = drawHand(lm);
    const h = hd[i][0];
    label(`${h.categoryName} ${h.score.toFixed(2)}`, box.x0, box.y0 - 10);
  });
  $info.innerHTML = `<h3>검출된 손</h3><p class="big">${r.landmarks.length}개</p>
    <p>손마다 21개 관절 좌표를 찾습니다. 빨간 점은 손가락 끝입니다.</p>`;
}

function runGesture(task, ts) {
  const r = task.recognizeForVideo(canvas, ts);
  const hd = r.handedness ?? r.handednesses;
  const found = [];
  r.landmarks.forEach((lm, i) => {
    const box = drawHand(lm);
    const hand = hd[i][0].categoryName;
    const g = r.gestures[i]?.[0];
    const ok = g && GESTURE_TEXT[g.categoryName] && g.score >= MIN_GESTURE_SCORE;
    const text = ok ? `${hand}: ${GESTURE_TEXT[g.categoryName]} ${g.score.toFixed(2)}` : `${hand}: -`;
    label(text, box.x0, box.y0 - 10, ok ? "#ffe040" : "#bbbbbb");
    if (ok) found.push(GESTURE_TEXT[g.categoryName]);
  });
  $info.innerHTML = `<h3>인식 결과</h3><p class="big">${found.join(" ") || "-"}</p>
    <h3>인식 가능한 제스처</h3><ul>${Object.values(GESTURE_TEXT).map(t => `<li>${t}</li>`).join("")}</ul>`;
}

function runFace(task, ts) {
  const r = task.detectForVideo(canvas, ts);
  let panel = "<p>얼굴을 찾는 중…</p>";
  r.faceLandmarks.forEach((lm, i) => {
    if (options.mesh) draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#80808070", lineWidth: 1 });
    draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#dddddd", lineWidth: 1 });
    draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#ffc800", lineWidth: 2 });
    draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#ffc800", lineWidth: 2 });
    draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30e030", lineWidth: 1 });
    draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#30e030", lineWidth: 1 });
    draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#0080ff", lineWidth: 2 });
    draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#0080ff", lineWidth: 2 });
    draw.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#ff3030", lineWidth: 2 });

    const s = {};
    for (const c of r.faceBlendshapes?.[i]?.categories ?? []) s[c.categoryName] = c.score;
    const states = [];
    const bl = (s.eyeBlinkLeft ?? 0) > BLINK_THRESHOLD, br = (s.eyeBlinkRight ?? 0) > BLINK_THRESHOLD;
    if (bl && br) states.push("Blink"); else if (bl || br) states.push("Wink");
    if ((s.jawOpen ?? 0) > MOUTH_THRESHOLD) states.push("Mouth Open");
    if (((s.mouthSmileLeft ?? 0) + (s.mouthSmileRight ?? 0)) / 2 > 0.5) states.push("Smile");

    const x0 = Math.min(...lm.map(p => p.x)) * canvas.width;
    const y0 = Math.min(...lm.map(p => p.y)) * canvas.height;
    label(states.join(" | ") || "-", x0, y0 - 10);

    if (i === 0) {
      const top = Object.entries(s).filter(([k]) => k !== "_neutral").sort((a, b) => b[1] - a[1]).slice(0, TOP_BLENDSHAPES);
      panel = `<h3>상태</h3><p class="big">${states.join(" · ") || "-"}</p><h3>표정 점수 TOP ${TOP_BLENDSHAPES}</h3>${bars(top)}`;
    }
  });
  $info.innerHTML = panel;
}

function runCustom(task, ts) {
  const r = task.detectForVideo(canvas, ts);
  const hd = r.handedness ?? r.handednesses;
  const { labels } = customModel;
  let probsShown = null;
  r.landmarks.forEach((lm, i) => {
    const hand = hd[i][0].categoryName;
    const probs = predictProba(landmarksToFeatures(lm, hand, canvas.width, canvas.height));
    const best = probs.indexOf(Math.max(...probs));
    const pred = probs[best] >= MIN_PROB ? labels[best] : "?";
    const hist = history[hand];
    hist.push(pred);
    if (hist.length > SMOOTH_FRAMES) hist.shift();
    const smoothed = mostCommon(hist);

    const box = drawHand(lm);
    const dim = smoothed === "?" || smoothed === "none";
    label(`${hand}: ${smoothed} ${probs[best].toFixed(2)}`, box.x0, box.y0 - 10, dim ? "#bbbbbb" : "#ffe040");
    if (EMOJI[smoothed]) {
      ctx.font = "80px 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(EMOJI[smoothed], (box.x0 + box.x1) / 2, Math.max(box.y0 - 40, 80));
      ctx.textAlign = "start";
    }
    probsShown ??= probs;
  });
  if (!r.landmarks.length) { history.Left.length = 0; history.Right.length = 0; }
  $info.innerHTML = `<h3>학습한 제스처</h3>
    ${bars(labels.map((l, i) => [`${EMOJI[l] ?? ""} ${l}`, probsShown ? probsShown[i] : 0]))}
    <p>확률 ${MIN_PROB} 미만은 "?"로 표시합니다.</p>`;
}

const RUNNERS = { hand: runHand, gesture: runGesture, face: runFace, custom: runCustom };

// ---------- 루프 ----------
async function loop() {
  if (video.readyState >= 2) {
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    // 파이썬의 cv2.flip(frame, 1) 처럼 좌우 반전한 화면을 모델에 넣는다 (왼손/오른손 판정도 동일)
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    const current = mode;
    const task = tasks[current];
    if (task && (current !== "custom" || customModel)) {
      let ts = performance.now();
      if (ts <= lastTs) ts = lastTs + 1;  // VIDEO 모드는 타임스탬프가 계속 증가해야 함
      lastTs = ts;
      RUNNERS[current](task, ts);
    }
    const now = performance.now();
    $fps.textContent = `FPS ${(1000 / Math.max(now - prevTime, 1)).toFixed(0)}`;
    prevTime = now;
  }
  requestAnimationFrame(loop);
}

function renderControls() {
  $controls.innerHTML = mode === "face"
    ? `<label><input type="checkbox" id="mesh" ${options.mesh ? "checked" : ""}> 얼굴 메시(삼각망) 표시</label>`
    : "";
  document.getElementById("mesh")?.addEventListener("change", e => { options.mesh = e.target.checked; });
}

async function setMode(next) {
  mode = next;
  document.querySelectorAll(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.mode === next));
  renderControls();
  $info.innerHTML = "";
  if (video.srcObject) {
    try { await getTask(next); } catch (e) { showError(e); }
  }
}

function showError(e) {
  console.error(e);
  $status.textContent = `오류: ${e.message ?? e}`;
}

document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));

document.getElementById("start").addEventListener("click", async () => {
  try {
    $status.textContent = "카메라 여는 중…";
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    await video.play();
    $overlay.hidden = true;
    requestAnimationFrame(loop);
    await getTask(mode);
  } catch (e) {
    showError(e);
  }
});

renderControls();
