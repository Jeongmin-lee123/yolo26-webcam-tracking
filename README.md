# YOLO26 Webcam Segmentation + Tracking

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
