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
