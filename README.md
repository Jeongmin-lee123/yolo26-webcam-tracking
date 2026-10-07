# YOLO26 Webcam Segmentation + Tracking

> 🌐 **웹 데모 (MediaPipe 손·제스처·얼굴·커스텀 제스처)**: https://jeongmin-lee123.github.io/yolo26-webcam-tracking/

[Ultralytics YOLO26](https://docs.ultralytics.com/) 모델로 웹캠 영상에서 실시간 instance segmentation과 object tracking을 수행합니다.

## 기능
- **Segmentation**: `yolo26n-seg.pt` 모델로 객체 마스크 표시
- **Tracking**: ByteTrack(기본) / BoT-SORT로 프레임 간 객체 ID 유지
- **이동 궤적**: 각 객체 중심의 최근 이동 경로 표시
- FPS, 현재 추적 수, 누적 ID 수 화면 표시

## 설치
```bash
pip install -U ultralytics opencv-python lap
```

## 실행
```bash
python webcam_detect.py
```
모델 가중치는 처음 실행 시 자동으로 다운로드됩니다.

## 단축키
| 키 | 기능 |
|---|---|
| `m` | 마스크 표시 on/off |
| `t` | 궤적 표시 on/off |
| `q` / `ESC` | 종료 |

## 설정
`webcam_detect.py` 상단 상수로 조정합니다.

| 상수 | 설명 | 기본값 |
|---|---|---|
| `MODEL_PATH` | 모델 (`yolo26{n,s,m,l,x}-seg.pt`) | `yolo26n-seg.pt` |
| `TRACKER` | `bytetrack.yaml` / `botsort.yaml` | `bytetrack.yaml` |
| `CAMERA_INDEX` | 웹캠 번호 | `0` |
| `CONF_THRESHOLD` | 신뢰도 임계값 | `0.5` |
| `TRAIL_LENGTH` | 궤적 길이(프레임) | `30` |

## 문제 해결
- Windows에서 `OSError: [WinError 1114] ... c10.dll` 오류가 나면 최신 [Visual C++ 재배포 패키지](https://aka.ms/vs/17/release/vc_redist.x64.exe)를 설치하세요.

---

# 강의노트 — YOLO26으로 웹캠 실시간 비전 만들기 (2026-10-07)

> 오늘 한 일: 웹캠으로 **Object Detection**을 만들고 → **Segmentation + Object Tracking**으로 확장한 뒤 → **GitHub**에 올리기까지

## 1. 개발 환경 준비

### 1-1. 패키지 설치
```bash
pip install -U ultralytics opencv-python lap
```
| 패키지 | 역할 |
|---|---|
| `ultralytics` | YOLO 모델 불러오기, 추론(predict), 추적(track). PyTorch도 함께 설치됨 |
| `opencv-python` | 웹캠 열기, 화면 출력, 도형/글자 그리기 |
| `lap` | 트래커(ByteTrack/BoT-SORT)가 내부에서 쓰는 매칭(할당 문제) 계산 라이브러리 |

### 1-2. 트러블슈팅: `WinError 1114` (c10.dll)
```
OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다.
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```
- **원인**: PC의 Visual C++ 재배포 패키지가 오래된 버전(14.29)이라 최신 PyTorch DLL을 불러오지 못함
- **해결**: 최신 [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) 설치 → 14.44로 업데이트
- **확인**: `python -c "import torch; print(torch.__version__)"` 가 오류 없이 출력되면 OK
- 참고: 설치된 PyTorch는 `+cpu` 버전 → GPU를 쓰려면 CUDA 버전 PyTorch를 따로 설치해야 함

## 2. 1단계 — Object Detection (객체 탐지)

**Detection**: 이미지 속 객체의 **위치(박스)** 와 **종류(클래스)**, **신뢰도**를 찾는 것

### 핵심 흐름
```
웹캠 프레임 읽기 → 모델에 넣기 → 결과(박스) 그리기 → 화면에 출력 → 반복
```

### 핵심 코드
```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")                     # 처음 실행 시 자동 다운로드
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)       # 0번 웹캠, Windows는 CAP_DSHOW가 빠름

while True:
    ok, frame = cap.read()                     # 프레임 1장 읽기
    results = model.predict(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()              # 박스+라벨이 그려진 이미지
    cv2.imshow("YOLO26", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):      # q 누르면 종료
        break
```

### 결과 값 꺼내 쓰기
```python
for box in results[0].boxes:
    cls_id = int(box.cls[0])                   # 클래스 번호 → model.names[cls_id] 로 이름
    conf   = float(box.conf[0])                # 신뢰도 (0~1)
    x1, y1, x2, y2 = map(int, box.xyxy[0])     # 박스 좌상단/우하단 좌표
```

### 알아둘 점
- **모델 크기**: `n`(nano) < `s` < `m` < `l` < `x` — 클수록 정확하지만 느림. 실시간 CPU에는 `n` 추천
- **conf (신뢰도 임계값)**: 낮추면 많이 잡지만 오탐↑, 높이면 확실한 것만 잡음
- **imgsz**: 모델 입력 크기(기본 640). 작게 하면 빨라지고 작은 물체는 놓치기 쉬움

## 3. 2단계 — Segmentation + Object Tracking

### 3-1. Segmentation (분할)
| | Detection | Instance Segmentation |
|---|---|---|
| 결과 | 사각형 박스 | 박스 + **픽셀 단위 마스크** |
| 모델 | `yolo26n.pt` | `yolo26n-seg.pt` |
| 속도 | 빠름 | 조금 느림 |

→ **모델 파일 이름만 `-seg`로 바꾸면** 같은 코드로 마스크까지 나온다.
`results[0].plot(masks=True/False)` 로 마스크 표시 여부를 조절.

### 3-2. Object Tracking (추적)
**Tracking**: 프레임마다 따로 탐지한 객체를 **같은 객체끼리 연결**해 고유 **ID**를 붙이는 것

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
ids = results[0].boxes.id                      # 객체별 트랙 ID (아무것도 없으면 None)
```
- `predict` → `track` 으로 바꾸기만 하면 됨
- **`persist=True`가 핵심**: 이전 프레임의 추적 상태를 기억해야 ID가 유지됨 (빼먹으면 매 프레임 ID가 새로 생김)
- `boxes.id`는 객체가 없으면 `None` → 반드시 `if boxes.id is not None:` 체크

| 트래커 | 특징 |
|---|---|
| `bytetrack.yaml` (기본) | 빠르고 가벼움. 신뢰도 낮은 박스도 매칭에 활용 |
| `botsort.yaml` | 카메라 움직임 보정 등으로 ID 유지가 더 안정적, 조금 느림 |

### 3-3. 이동 궤적(Trail) 그리기
- 트랙 ID별로 박스 **중심점**을 `deque(maxlen=30)`에 저장 → 최근 30프레임만 자동 유지
- 저장된 점들을 `cv2.line`으로 이어 그림 (최근일수록 두껍게)
- 화면에서 사라진 ID의 궤적은 삭제해 메모리 정리

```python
trails = defaultdict(lambda: deque(maxlen=30))
cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
trails[track_id].append((cx, cy))
```

### 3-4. 단축키로 기능 토글
`cv2.waitKey(1)`의 반환값으로 키 입력 처리: `m` 마스크, `t` 궤적, `q`/`ESC` 종료

## 4. 3단계 — GitHub에 올리기

```bash
git init -b main                 # 저장소 초기화
git add .                        # 파일 스테이징
git commit -m "메시지"            # 커밋
gh repo create yolo26-webcam-tracking --public --source=. --remote=origin --push
```
- **`.gitignore`로 모델 파일(`*.pt`) 제외**: 용량이 크고, 실행 시 자동 다운로드되므로 올릴 필요 없음
- 이후 수정 사항 반영: `git add .` → `git commit -m "..."` → `git push`

## 5. 오늘의 정리
1. `YOLO("모델.pt")` 한 줄로 모델 로드, `predict` 로 탐지
2. 모델만 `-seg`로 바꾸면 **Segmentation**
3. `predict` → `track(persist=True)` 로 바꾸면 **Tracking**
4. Windows에서 torch DLL 오류는 **VC++ 재배포 패키지 업데이트**로 해결
5. 코드는 GitHub에, 모델 가중치는 `.gitignore`로 제외

### 다음에 해볼 것
- CUDA 버전 PyTorch 설치해 GPU로 FPS 올리기
- 특정 클래스만 추적하기 (`classes=[0]` → 사람만)
- 선(line)을 넘은 객체 수 세기 (객체 카운팅)
- 결과 영상을 파일로 저장하기 (`cv2.VideoWriter`)

---

# 강의노트 — MediaPipe로 손 랜드마크 & 제스처 인식 (2026-10-07)

> 오늘 한 일 (2부): Google **MediaPipe**로 웹캠에서 **손 관절 21개 검출(Hand Landmarker)** → **손 제스처 인식(Gesture Recognizer)** 까지

## 1. MediaPipe란?
- Google이 만든 온디바이스 AI 프레임워크. 손·얼굴·포즈 등 **미리 학습된 모델(.task)** 을 바로 가져다 쓸 수 있음
- YOLO와 다른 점: 범용 객체 탐지가 아니라 **특정 작업(Task)에 특화**된 가볍고 빠른 모델 묶음 → CPU에서도 실시간

| | YOLO26 (1부) | MediaPipe (2부) |
|---|---|---|
| 하는 일 | 80종 객체 탐지/분할/추적 | 손 관절 위치, 제스처 분류 |
| 모델 파일 | `.pt` (자동 다운로드) | `.task` (**직접 다운로드**) |
| 불러오기 | `YOLO("모델.pt")` | `XXXOptions` → `XXX.create_from_options()` |

## 2. 준비

### 2-1. 설치
```bash
pip install -U mediapipe opencv-python
```

### 2-2. 모델(.task) 다운로드
MediaPipe 모델은 자동 다운로드가 안 되므로 공식 문서의 **Models** 표에서 받아 `.py`와 같은 폴더에 둔다.

| Task | 모델 파일 | 다운로드 |
|---|---|---|
| Hand Landmarker | `hand_landmarker.task` (7.8MB) | [링크](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |
| Gesture Recognizer | `gesture_recognizer.task` (8.4MB) | [링크](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task) |

- 문서: [Hand landmarks detection](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) / [Gesture recognition](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)
- 오늘은 **Claude in Chrome**으로 문서 페이지를 열어 다운로드 링크를 찾아 받았다.
  Chrome의 "다운로드 전에 저장 위치 확인" 설정이 켜져 있으면 저장 창에서 직접 **저장**을 눌러야 한다.

### 2-3. 실행
```bash
python webcam_hand.py      # 손 랜드마크
python webcam_gesture.py   # 손 제스처 인식
```
`q` / `ESC` 로 종료.

## 3. MediaPipe Tasks 공통 사용 패턴

```python
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ① 옵션 만들기
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
# ② 객체 생성 (with 문으로 자원 자동 해제)
with vision.HandLandmarker.create_from_options(options) as landmarker:
    # ③ OpenCV(BGR) → MediaPipe Image(RGB) 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # ④ 추론
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

### Running mode 3가지
| 모드 | 용도 | 호출 함수 |
|---|---|---|
| `IMAGE` | 사진 1장 | `detect()` / `recognize()` |
| `VIDEO` | 동영상·웹캠 프레임 (동기) | `detect_for_video()` / `recognize_for_video()` |
| `LIVE_STREAM` | 실시간 스트림 (비동기, 콜백으로 결과 받음) | `detect_async()` / `recognize_async()` |

- 오늘은 코드가 단순한 **VIDEO 모드** 사용
- VIDEO 모드는 **타임스탬프(ms)가 계속 증가**해야 함 → `time.monotonic()` 기준으로 계산
- **BGR → RGB 변환을 꼭 해야 함** (OpenCV는 BGR, MediaPipe는 RGB)

## 4. Hand Landmarker — 손 관절 21개 (`webcam_hand.py`)

### 21개 랜드마크 번호
```
0 손목(WRIST)
엄지 1~4 | 검지 5~8 | 중지 9~12 | 약지 13~16 | 새끼 17~20
손가락 끝(TIP) = 4, 8, 12, 16, 20
```

### 결과 구조
```python
result.hand_landmarks[i]       # i번째 손의 21개 점, 각 점은 x, y, z
result.handedness[i][0]        # .category_name = "Left"/"Right", .score = 신뢰도
result.hand_world_landmarks[i] # 실제 3D 좌표(미터 단위, 손 중심 기준)
```
- `x`, `y`는 **0~1로 정규화된 값** → 화면 좌표로 쓰려면 `int(lm.x * w), int(lm.y * h)`
- `z`는 손목 기준 깊이 (작을수록 카메라에 가까움)

### 그리기
- 21개 점을 `HAND_CONNECTIONS`(뼈대 연결 쌍 목록)대로 `cv2.line`으로 이음
- 손가락 끝은 빨간 점으로 강조, 손 위에 왼손/오른손 라벨 표시
- `cv2.flip(frame, 1)`로 **거울 모드** → 사용자가 보기 자연스러움

## 5. Gesture Recognizer — 손 제스처 (`webcam_gesture.py`)

Hand Landmarker + **제스처 분류 모델**이 합쳐진 것. 랜드마크 결과도 똑같이 나오고 `gestures`가 추가됨.

```python
result = recognizer.recognize_for_video(mp_image, timestamp_ms)
g = result.gestures[i][0]      # i번째 손의 가장 확률 높은 제스처
g.category_name, g.score       # 예: "Thumb_Up", 0.87
```

### 기본 제스처 7종
| 라벨 | 손 모양 |
|---|---|
| `Closed_Fist` | ✊ 주먹 |
| `Open_Palm` | 🖐 손바닥 |
| `Pointing_Up` | ☝️ 검지 위로 |
| `Thumb_Down` / `Thumb_Up` | 👎 / 👍 |
| `Victory` | ✌️ |
| `ILoveYou` | 🤟 |
| `None` | 손은 있지만 알 수 없는 동작 |

- 점수가 낮은 결과(< 0.5)나 `None`은 `-`로 표시하도록 필터링
- `Custom gesture classifier is not defined.` 로그는 직접 학습시킨 제스처가 없다는 안내일 뿐 → 무시
- Model Maker로 **나만의 제스처를 학습**시켜 추가할 수도 있음

## 6. 트러블슈팅: 한글 경로에서 모델 파일을 못 여는 문제
```
RuntimeError: Unable to open file at C:\Users\...\諛뷀깢 ?붾㈃\...\hand_landmarker.task, errno=-1
```
- **원인**: MediaPipe 내부(C++)가 경로를 열 때 **한글(비ASCII) 경로가 깨짐** (`바탕 화면`, `청년 피지컬ai`)
- **해결**: 경로 대신 **파일 내용을 바이트로 읽어서** 넘긴다
```python
# ❌ 한글 경로에서 실패
BaseOptions(model_asset_path=str(MODEL_PATH))
# ✅ 파이썬이 파일을 읽어서 바이트로 전달
BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes())
```
- 다른 해결책: 프로젝트를 `C:\work\` 처럼 영문 경로로 옮기기

## 7. 오늘의 정리 (2부)
1. MediaPipe는 `.task` 모델을 **직접 다운로드**해서 사용
2. 사용 패턴: `Options` → `create_from_options` → `mp.Image`(RGB) → `detect/recognize_for_video`
3. 손 랜드마크는 **21개 점, 0~1 정규화 좌표**
4. Gesture Recognizer = 랜드마크 + 제스처 7종 분류
5. 한글 경로 오류는 `model_asset_buffer`로 해결

### 다음에 해볼 것
- 펴진 손가락 개수 세기 (TIP과 PIP 관절의 y좌표 비교)
- 엄지–검지 거리로 **핀치** 감지 → 볼륨/밝기 조절
- 제스처로 동작 실행 (👍 캡처 저장, ✊ 종료 등)
- Model Maker로 커스텀 제스처 학습

---

# 강의노트 — 얼굴 랜드마크, 커스텀 제스처 학습, 웹 배포 (2026-10-07)

> 오늘 한 일 (3부): **Face Landmarker**로 얼굴 478점 + 표정 → **나만의 제스처 수집·학습·추론** → **GitHub Pages로 웹 배포**

## 1. Face Landmarker — 얼굴 478점 + 표정 (`webcam_face.py`)

모델: [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) (3.7MB) · [문서](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)

```python
options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,   # 표정 점수 52종
)
result = landmarker.detect_for_video(mp_image, timestamp_ms)
result.face_landmarks[i]            # 478개 점 (468 얼굴 + 10 홍채)
result.face_blendshapes[i]          # [Category(category_name="jawOpen", score=0.8), ...]
```

- **Blendshape**: 표정을 52개 점수(0~1)로 표현. 예: `eyeBlinkLeft`, `jawOpen`, `mouthSmileLeft`
- 점수에 임계값만 걸면 상태 판정 가능 → 눈 감음(`eyeBlink* > 0.5`), 입 벌림(`jawOpen > 0.3`), 미소(`mouthSmile* > 0.5`)
- 연결선 목록은 `vision.FaceLandmarksConnections`에 들어 있음 (`FACE_LANDMARKS_LIPS`, `..._LEFT_EYE`, `..._TESSELATION` 등)

| 키 | 기능 |
|---|---|
| `m` | 얼굴 메시(삼각망) on/off |
| `b` | 표정 점수 패널 on/off |

## 2. 커스텀 제스처 — 수집 → 학습 → 추론

### 2-1. 왜 Model Maker 대신 직접 만들었나
- MediaPipe 공식 커스텀 제스처 도구(`mediapipe-model-maker`)는 **TensorFlow + Python 3.11 이하** 필요 → 현재 Python 3.14에서 설치 불가
- 대신 **Hand Landmarker의 21개 점을 특징으로 쓰고 scikit-learn으로 분류기 학습** → 가볍고 CPU로 몇 초면 학습

```
웹캠 → Hand Landmarker → 21점(x,y,z) → 정규화(63차원) → MLP 분류기 → 제스처
```

### 2-2. 특징 정규화 (`gesture_common.py`) — 핵심 아이디어
같은 손 모양이면 **위치·크기·좌우와 상관없이 같은 숫자**가 나오도록 변환:
1. x에 (가로/세로) 곱하기 → 화면 비율 때문에 손 모양이 찌그러지지 않게
2. 손목(0번)을 원점으로 → 화면 어디에 있든 동일
3. 손목~가장 먼 점 거리로 나누기 → 카메라와의 거리와 무관
4. 왼손이면 x 뒤집기 → 오른손으로만 모아도 왼손 인식

### 2-3. 사용 방법
```bash
pip install -U scikit-learn pillow

# ① 수집: 라벨마다 실행, SPACE로 녹화 시작/정지, 300개 모이면 자동 종료
python collect_gesture.py --label none           # 아무 동작 아닌 손 (오인식 방지용, 꼭 넣기)
python collect_gesture.py --label korean_heart
python collect_gesture.py --label ok

# ② 학습: 20%로 검증 → 전체로 재학습 → custom_gesture_model.joblib 저장
python train_gesture.py

# ③ 실시간 추론 (인식되면 이모지 표시)
python webcam_custom_gesture.py

# ④ (웹 데모 갱신) 학습 모델을 JSON으로 변환
python export_web_model.py
```

| 파일 | 역할 |
|---|---|
| `collect_gesture.py` | 웹캠 → 정규화된 랜드마크를 `data/custom_gestures.csv`에 누적 저장 |
| `train_gesture.py` | `StandardScaler` + `MLPClassifier(64, 32)` 학습, 검증 리포트·혼동 행렬 출력 |
| `webcam_custom_gesture.py` | 실시간 추론, 확률 0.7 미만은 `?`, 최근 5프레임 다수결로 흔들림 감소, 이모지 표시 |
| `export_web_model.py` | 학습된 가중치를 `docs/custom_gesture_model.json`으로 변환 |

### 2-4. 오늘 학습 결과
| 라벨 | 샘플 | 검증 정확도 |
|---|---|---|
| `none` | 300 | 전체 **98.3%** |
| `korean_heart` ❤️ | 300 | (검증 180개 중 177개 정답) |
| `ok` 👌 | 300 | |

- 같은 녹화에서 나눈 검증이라 실제 사용 정확도는 이보다 낮을 수 있음 → 각도·거리를 다양하게 더 모으면 좋아짐
- `ok`와 `korean_heart`는 둘 다 엄지·검지를 맞대는 모양이라 헷갈리기 쉬움

### 2-5. OpenCV 화면에 이모지 그리기
- `cv2.putText`는 이모지/한글을 못 그림 → **Pillow**로 Windows 컬러 이모지 폰트(`seguiemj.ttf`)를 이용해 RGBA 이미지로 만든 뒤 **알파 합성**
```python
font = ImageFont.truetype("C:/Windows/Fonts/seguiemj.ttf", 96)
ImageDraw.Draw(img).text((0, 0), "👌", font=font, embedded_color=True)  # embedded_color=True가 컬러 이모지
```
- 새 라벨의 이모지는 `webcam_custom_gesture.py`의 `EMOJI` 딕셔너리에 추가

## 3. GitHub Pages로 웹 배포 (`docs/`)

파이썬은 GitHub Pages에서 실행할 수 없으므로 **MediaPipe JavaScript 버전**(`@mediapipe/tasks-vision`)으로 같은 기능을 웹에 옮김.

| 파일 | 역할 |
|---|---|
| `docs/index.html`, `style.css` | 페이지 (탭: 손 / 제스처 / 얼굴 / 커스텀 제스처) |
| `docs/app.js` | 웹캠 → MediaPipe 추론 → 캔버스 그리기 |
| `docs/custom_gesture_model.json` | 파이썬에서 학습한 MLP 가중치 (브라우저에서 직접 계산) |

- 영상은 **브라우저 안에서만 처리**(WebAssembly + GPU), 서버로 전송되지 않음
- 모델(`.task`)은 Google 저장소 URL에서 바로 불러옴 → 저장소에 모델 파일 불필요
- 파이썬과 같은 결과를 내도록 **좌우 반전 후 추론**, 정규화·MLP 계산도 동일하게 구현 (파이썬과 확률 일치 확인)
- 배포 설정: GitHub 저장소 → Settings → Pages → `main` 브랜치 `/docs` 폴더
- 웹캠은 **HTTPS**에서만 동작 → GitHub Pages는 기본 HTTPS라 OK

```bash
gh api repos/<사용자>/<저장소>/pages -X POST -f "source[branch]=main" -f "source[path]=/docs"
```

## 4. 오늘의 정리 (3부)
1. Face Landmarker는 **478점 + 표정 52종(blendshape)** → 임계값으로 눈 깜빡임·입 벌림·미소 판정
2. 커스텀 제스처 = **랜드마크 정규화 + 작은 분류기**로 충분 (수집 → 학습 → 추론)
3. `none` 클래스를 넣어야 아무 손이나 억지로 분류하지 않음
4. OpenCV에 이모지는 Pillow로 그려서 합성
5. 파이썬 모델도 가중치를 JSON으로 내보내면 **웹에서 그대로** 쓸 수 있음 → GitHub Pages로 배포
