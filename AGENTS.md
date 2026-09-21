# Vision Guard 협업 및 구현 규정

이 파일은 이 저장소에서 작업하는 개발자와 코딩 에이전트가 따라야 하는 공통 기준이다. 저장소 전체에 적용한다.

## 1. 기준 문서와 변경 원칙

- 기능과 아키텍처의 최우선 기준은 `docs/VISION_GUARD_SPEC.md`다.
- 설계하거나 구현하기 전에 관련 명세와 현재 코드를 먼저 확인한다.
- 요구사항, 인터페이스, MVP 범위, 개발 우선순위 또는 Edge 제약이 바뀌면 기능 명세도 같은 변경에서 갱신한다.
- 명세와 코드가 충돌하면 임의로 기능을 바꾸지 말고 차이를 기록한 뒤 일관된 방향으로 수정한다.
- 기능 명세는 “무엇을 만들지”, 이 파일은 “어떤 규칙과 순서로 만들지”를 정의한다.

## 2. 프로젝트 목표

Vision Guard는 CCTV 영상을 실시간 분석하여 작업자와 산업 설비의 위험한 공간관계를 판단하는 Edge Vision 안전 시스템이다.

```text
CCTV / RTSP / Video
→ Frame 입력
→ YOLO Detection
→ ByteTrack Tracking
→ Track History / Trajectory
→ Equipment State Analysis
→ Zone / Spatial Analysis
→ Risk Engine
→ Risk Event
→ Alert / DB / Snapshot / Event Clip / Report
```

최종 배포 대상은 NVIDIA Jetson 계열 장치다. 정확도뿐 아니라 FPS, 평균 지연시간, P95 지연시간, CPU/GPU 사용량, RAM/VRAM 및 전력 사용량을 함께 고려한다.

## 3. 현재 구현 상태

이 절의 상태는 `main` 브랜치를 기준으로 한다. 다른 기능 브랜치에만 존재하는 코드는 병합 전까지 구현 완료로 간주하지 않는다.

### `main`에 구현 및 병합됨

- `app/inference/detector.py`
  - `Detection` 데이터 클래스와 `Detector` 추상 인터페이스
  - 원본 프레임 기준 `xyxy` bbox, confidence 및 프레임 검증
  - center와 bottom-center 계산
- `app/video/video_source.py`
  - OpenCV 기반 로컬 영상과 Webcam 입력
  - 불변 `FramePacket`, `CameraConfig`, YAML 설정 로드
  - open/read/reset/close, context manager 및 반복 재생
- `app/video/frame_processor.py`
  - Detection → Tracking → Trajectory 호출
  - Detection/Tracking bbox와 FPS/latency 렌더링
  - 원본 `FramePacket`을 변경하지 않고 `rendered_frame`을 별도 반환
  - 아직 Zone → Risk → Event → Alert 호출은 연결하지 않음
- `app/tracking/tracker.py`, `app/tracking/trajectory.py`
  - 공통 `Tracker` 계약, 단순 추적기와 ByteTrack adapter
  - `TrackedObject`, Track History, 이동거리, 방향 및 pixel 속도 계산
- `app/zones/`
  - 작업자 bottom-center와 bbox 포함/확대 geometry
  - 지게차·컨베이어·로봇팔 bbox 기반 Warning/Critical Zone 생성
  - `ZoneManager`가 프레임별 `ZoneFrameResult`를 생성
- `app/risk/`
  - `NORMAL/WARNING/CRITICAL` 위험등급
  - 작업자 bottom-center와 설비 bbox Zone, `is_active`를 조합한 순수 규칙
  - 모든 작업자-설비 조합에 대한 `RiskAssessment` 생성
- 설정과 테스트
  - `requirements.txt`와 `config/*.yaml`의 기본 골격 작성
  - `test_video_source.py`, `test_frame_processor.py`, `test_tracking.py`, `test_trajectory.py`, `test_zones.py`, `test_risk.py`
  - 2026-09-22 기준 `pytest`: 36 passed, `python -m compileall -q app`: 통과

### 모델 아티팩트 상태

- 실제 모델 파일은 Git LFS로 추적한다.
- `models/1model_20260909.pt`: 작업자와 산업 설비 탐지용 모델
  - class: `person`, `forklift`, `robot_arm`, `conveyor`
- `models/2model_20260921.pt`: PPE 탐지용 모델
  - class: `head`, `gloves`, `helmet`, `body`, `safety_vest`, `harness_body`
- `models/1model.best.onnx`, `models/2model.best.onnx`: ONNX Runtime 및 TensorRT 변환 입력용 FP32 모델
- `models/1model.best.engine`, `models/2model.best.engine`: 개발 PC에서 생성한 TensorRT engine
- TensorRT engine은 GPU, CUDA, TensorRT 버전에 종속되므로 Jetson 배포 장치에서 다시 빌드한다.
- `models/best.pt`는 0 byte placeholder이며 실행 모델로 사용하지 않는다.

### 아직 구현 또는 통합되지 않음

- `app/inference/pytorch_backend.py`, `tensorrt_backend.py`, `model_manager.py`는 docstring만 있는 상태다.
- `app/main.py`는 docstring만 있어 실제 영상 End-to-End 실행 진입점이 없다.
- `FrameProcessor`와 `ZoneManager`, `RiskEngine`, Event/Alert는 아직 하나의 실행 파이프라인으로 연결되지 않았다.
- `app/equipment/` 상태 분석 모듈은 아직 없고 `EquipmentZoneInfo.is_active: bool`을 임시 계약으로 사용한다.
- `config/zones.yaml`의 Static Polygon은 현재 `ZoneManager` 런타임에 로드되지 않는다.
- `app/alerts/`, `app/storage/`, `app/api/`, `app/video/rtsp_source.py`, Frontend는 실행 가능한 구현이 아니다.
- Event deduplication/cooldown, DB, Snapshot, Event Clip 및 보고서는 구현되지 않았다.
- 빈 파일이나 placeholder가 존재한다는 이유만으로 기능이 구현됐다고 판단하지 않는다.

### 현재 차단 요소와 기술 부채

- `config/model.yaml`은 아직 0 byte `models/best.pt`를 가리키며 두 모델의 서로 다른 class ID를 하나의 mapping으로 합쳐 놓았다.
- 실행 전에 산업설비 모델과 PPE 모델의 설정을 분리하고 실제 파일 경로 및 class mapping을 각각 일치시켜야 한다.
- 설비 상태 모듈이 없기 때문에 현재 Zone 생성 함수는 `DEFAULT_IS_ACTIVE = True`를 사용한다. 이는 보수적인 프로토타입 임시값이며 실제 상태 판정으로 교체해야 한다.
- 설비 상태는 최종적으로 Boolean 대신 `MOVING/STOPPED/UNKNOWN` 또는 `RUNNING/STOPPED/UNKNOWN` Enum과 Temporal Smoothing을 사용한다.
- bbox 기반 Zone과 Risk 단위 테스트는 통과하지만 실제 모델·실제 영상 기반 FPS, latency 및 경고 시나리오는 아직 검증하지 않았다.
- `origin/feat/tracking-tuning`의 PyTorch backend와 실행 조립 코드는 `main`에 병합되지 않았으므로 검토·정리 없이 복사하거나 완료 상태로 기록하지 않는다.


## 4. 반드시 지켜야 하는 아키텍처 규칙

1. Detection, Tracking, Equipment, Zone, Risk, Alert, Storage를 서로 강하게 결합하지 않는다.
2. 각 모듈은 명시적인 입력을 받고 표준 데이터 객체를 반환한다.
3. PyTorch와 TensorRT는 동일한 `Detector` 인터페이스와 `Detection` 출력 형식을 사용한다.
4. Risk Engine만 여러 분석 결과를 종합하여 최종 위험도를 계산한다.
5. Frontend와 API route에는 안전 판단 규칙을 구현하지 않는다.
6. 설비 상태와 위험도는 한 프레임만으로 확정하지 않고 Temporal Smoothing을 적용한다.
7. 설비 상태는 Boolean 대신 `MOVING/STOPPED/UNKNOWN`, `RUNNING/STOPPED/UNKNOWN`과 같은 명시적 상태를 사용한다.
8. 위험 이벤트는 매 프레임 생성하지 않고 Event State, Deduplication, Cooldown을 적용한다.
9. 작업자의 영상 내 대표 위치는 bbox의 bottom-center를 기본으로 한다.
10. bbox는 원본 영상 기준 `(x1, y1, x2, y2)` 픽셀 좌표로 통일한다.
11. 픽셀 거리와 실제 거리(m)를 혼용하지 않는다. 실제 거리와 속도에는 Calibration/Homography가 필요하다.
12. 모든 AI 모델을 상시 실행하지 않는다. 고비용 모델은 위험 접근 또는 이상 징후가 있을 때 Trigger 방식으로 실행한다.
13. `app/main.py`와 `app/video/frame_processor.py`는 조립과 호출 순서만 관리하며 세부 알고리즘을 포함하지 않는다.
14. UI, MQTT 또는 DB 장애가 영상 추론 루프 전체를 중단시키지 않도록 경계를 분리한다.
15. 먼저 하나의 위험 시나리오를 End-to-End로 완성한 뒤 다른 설비와 고도화 기능으로 확장한다.

## 5. 모듈별 파일 책임

### 실행과 영상 처리

- `app/main.py`: 설정 로드, 의존성 생성, 처리 시작·종료, signal과 자원 해제를 담당한다.
- `app/video/video_source.py`: 로컬 동영상과 Webcam의 공통 입력 및 `FramePacket` 생성을 담당한다.
- `app/video/rtsp_source.py`: RTSP 연결, timeout, buffering, stale frame 제거 및 자동 재접속을 담당한다.
- `app/video/frame_processor.py`: Detection → Tracking → Equipment → Zone → Risk 호출 순서를 담당한다.

### 객체 탐지

- `app/inference/detector.py`: `Detection`과 backend 공통 `Detector` 계약을 정의한다.
- `app/inference/pytorch_backend.py`: `.pt` 모델 로딩, YOLO 추론, 필터링 및 표준 출력 변환을 담당한다.
- `app/inference/tensorrt_backend.py`: `.engine` 로딩, TensorRT 전처리·추론·후처리를 담당한다.
- `app/inference/model_manager.py`: backend 선택, 모델 검증, warm-up, 클래스와 threshold 설정을 담당한다.

### 객체 추적

- `app/tracking/tracker.py`: ByteTrack 실행, Track ID 유지 및 `TrackedObject` 생성을 담당한다.
- `app/tracking/trajectory.py`: Track별 최근 위치, 이동거리, 방향, 속도, 체류시간과 미래경로 기반 데이터를 관리한다.

### 설비 상태

명세에는 있지만 현재 실제 디렉터리가 없으므로 구현 시 다음 파일을 생성한다.

- `app/equipment/forklift_state.py`: Track History로 지게차의 `MOVING/STOPPED/UNKNOWN`을 판단한다.
- `app/equipment/robot_state.py`: Robot ROI의 Frame Difference 또는 Optical Flow로 `RUNNING/STOPPED/UNKNOWN`을 판단한다.
- `app/equipment/conveyor_state.py`: Belt ROI의 Flow Magnitude와 방향 일관성으로 `RUNNING/STOPPED/UNKNOWN`을 판단한다.

### Zone과 공간 분석

- `app/zones/geometry.py`: center, bottom-center, point-in-polygon, 거리, 교차와 trajectory buffer 계산을 담당한다.
- `app/zones/zone_manager.py`: 카메라별 Zone의 생성·조회·수정·삭제·로드·저장을 담당한다.
- `app/zones/worker_zone.py`: 작업자 위치와 Zone 진입·이탈·체류시간을 계산한다.
- `app/zones/forklift.py`: 초기 안전 buffer와 향후 이동방향 기반 Dynamic Zone을 생성한다.
- `app/zones/conveyor.py`: 컨베이어 Static Danger Polygon과 접근 판정을 담당한다.
- `app/zones/robot_arm.py`: 로봇팔 Safety Zone을 생성한다.
- `app/zones/calibration.py`: Homography, Perspective Transform, Bird's-Eye View 및 Pixel→World 변환을 담당한다.

로봇팔 Zone 단계:

```text
초기: YOLO Robot Arm bbox + margin
중기: Temporal Smoothing + Camera별 Static Polygon 또는 사전 정의 작업 반경
고도화: 작동상태 + 작업반경 + Motion + 선택적 PLC 정보를 결합한 Dynamic Zone
```

현재 초기 로봇팔 Zone은 Tracking된 `TrackedObject.bbox`를 입력으로 받고, bbox 자체를 Critical Zone으로, 확대 bbox를 Warning Zone으로 반환한다. `app/zones/robot_arm.py`가 생성하고 `ZoneManager`가 프레임 결과에 포함한다. 기본 확대 비율은 아직 Python 기본값이므로 `config/zones.yaml` 로딩과 Temporal Smoothing은 중기 단계에서 연결한다.

### 위험 판단

- `app/risk/risk_rules.py`: 설비별 조건, 거리와 시간 threshold 및 위험등급 매핑을 순수 규칙으로 정의한다.
- `app/risk/risk_engine.py`: Worker, Equipment State, Zone, Distance, Trajectory, PPE 결과를 종합해 `RiskAssessment`를 반환한다.
- 위험등급은 `NORMAL`, `WARNING`, `CRITICAL`로 통일한다.

### 이벤트와 경고

- `app/alerts/event_manager.py`: Risk Assessment를 Risk Event로 전환하고 ACTIVE/RESOLVED, 중복 제거, cooldown과 등급 상승을 관리한다.
- `app/alerts/alert_manager.py`: 화면 경고, 객체 강조, 경고 문구, 경고음과 외부 경고장치를 담당한다.
- `app/alerts/mqtt_client.py`: 중앙 관제 또는 외부 장치로 이벤트와 장치 상태를 Publish한다. MVP에서는 선택사항이다.

### 저장소

- `app/storage/models.py`: SQLite Event 테이블과 영속 데이터 모델을 정의한다.
- `app/storage/database.py`: DB 연결, 테이블 초기화, transaction과 session 수명주기를 담당한다.
- `app/storage/event_repository.py`: 이벤트 저장·조회·수정·종료 인터페이스를 제공한다.
- `app/storage/media_repository.py` 신규: Snapshot과 Event Clip 파일 저장 및 경로 반환을 담당한다.
- `app/video/ring_buffer.py` 신규: 위험 발생 전후 프레임을 보관해 Event Clip 생성을 지원한다.

### API와 Frontend

- `app/api/server.py`: FastAPI 앱 생성, router와 WebSocket 등록 및 수명주기를 담당한다.
- `app/api/routes/cameras.py`: 카메라 조회·등록·수정과 연결 상태 API를 담당한다.
- `app/api/routes/zones.py`: Zone CRUD와 관리자 Polygon 반영 API를 담당한다.
- `app/api/routes/events.py`: 이벤트 목록·상세·필터 조회 API를 담당한다.
- `app/api/routes/system.py`: FPS, latency, backend, 모델과 장치 상태 API를 담당한다.
- `app/api/websocket.py`: 실시간 Alert와 시스템 상태를 관리자 화면에 전달한다.
- `app/api/schemas.py` 신규: API Request/Response와 WebSocket 메시지 스키마를 정의한다.
- `frontend/`: 결과 시각화와 관리 기능만 담당하며 위험 판단 로직을 포함하지 않는다.

## 6. 표준 데이터 계약

모듈 간에는 임의의 dict 대신 type hint가 있는 dataclass, Enum 또는 명시적 모델을 사용한다.

### `Detection`

정의 위치: `app/inference/detector.py`

```text
class_id: int
class_name: str
confidence: float (0.0~1.0)
bbox: tuple[float, float, float, float]  # xyxy, original-frame pixels
```

### `FramePacket`

정의 위치: `app/video/video_source.py`

```text
camera_id: str
frame: numpy.ndarray
frame_index: int
timestamp: float
fps: float
width: int
height: int
```

### `TrackedObject`

정의 위치: `app/tracking/tracker.py`

```text
track_id
class_id
class_name
bbox
center
bottom_center
confidence
timestamp
```

### `EquipmentStateResult`

목표 정의 위치: `app/equipment/`의 공통 모델. 아직 구현되지 않았으며 현재 `EquipmentZoneInfo.is_active: bool`을 임시 사용한다.

```text
track_id
equipment_type
state
motion_score 또는 speed
direction
confidence
timestamp
```

### `ZoneFrameResult`와 `EquipmentZoneInfo`

정의 위치: `app/zones/models.py`

```text
ZoneFrameResult:
  timestamp
  frame_index
  camera_id
  workers: list[WorkerZoneInfo]
  equipments: list[EquipmentZoneInfo]

EquipmentZoneInfo:
  equipment_id
  equipment_type
  equipment_bbox
  warning_zone
  critical_zone
  is_active  # prototype-only bool, EquipmentState Enum으로 교체 예정
```

### `RiskAssessment`

정의 위치: `app/risk/models.py`

```text
timestamp
frame_index
camera_id
person_id
equipment_id
equipment_type
risk_level
```

`RiskEvent`는 아직 정의되지 않았다. Event 구현 시 `RiskAssessment`와 생명주기 상태, deduplication key, 발생·갱신·해제 시각을 분리하여 정의한다.

ORM 객체를 Detection, Tracking 또는 Risk 계층의 데이터 계약으로 사용하지 않는다.

## 7. `main` 기준 다음 개발 순서

현재 완료된 Detection 계약, 영상 입력, Tracking, Trajectory, bbox Zone, Risk 단위 로직을 기반으로 아래 순서대로 진행한다. 각 단계의 완료 조건을 만족한 후 다음 단계로 이동한다.

1. **모델 설정 분리**
   - `config/model.yaml`을 산업설비 모델과 PPE 모델 설정으로 분리
   - `1model_20260909.pt`, `2model_20260921.pt`의 class ID와 실제 label 순서를 각각 기록
   - 0 byte `models/best.pt` 의존 제거
2. **PyTorch YOLO backend**
   - `app/inference/pytorch_backend.py`에서 `1model_20260909.pt` 로딩·추론·`Detection` 변환
   - `app/inference/model_manager.py`에서 설정 검증, backend 선택 및 warm-up
   - 실제 프레임에서 원본 픽셀 bbox와 confidence 검증
3. **실행 진입점**
   - `app/main.py`에서 CameraConfig, VideoSource, Detector, Tracker, Trajectory를 조립
   - 종료 signal, VideoCapture와 창 자원 해제
4. **Zone과 Risk 파이프라인 연결**
   - `app/video/frame_processor.py`에 `ZoneManager`와 `RiskEngine` 호출 추가
   - 반환 객체에 `ZoneFrameResult`와 `RiskAssessment`를 명시적으로 포함
5. **프로토타입 터미널 경고**
   - `app/alerts/alert_manager.py`에 WARNING/CRITICAL 로그 출력 구현
   - 같은 객체 조합의 매 프레임 중복 출력은 최소한의 상태 추적으로 억제
6. **첫 End-to-End 시현 검증**
   - 로컬 영상 또는 Webcam → Detection → Tracking → Zone → Risk → 터미널 Alert
   - 작업자 진입 시 WARNING/CRITICAL, 이탈 시 정상 복귀 확인
   - FPS, 평균 latency와 P95 latency 기록
7. **설비 상태 계약 정상화**
   - `app/equipment/forklift_state.py`와 공통 `EquipmentStateResult` 생성
   - `is_active: bool`을 `MOVING/STOPPED/UNKNOWN`으로 교체
   - jitter, hysteresis와 Temporal Smoothing 테스트
8. **Static Polygon Zone**
   - `config/zones.yaml` 로더와 카메라별 필터링 구현
   - bbox Zone과 관리자 Polygon의 우선순위·결합 규칙 명시
9. **Event 생명주기**
   - `app/alerts/event_manager.py`에 ACTIVE/RESOLVED, deduplication, cooldown 및 등급 상승 구현
10. **저장과 증적**
    - `app/storage/models.py`, `database.py`, `event_repository.py`
    - `app/storage/media_repository.py`, `app/video/ring_buffer.py`로 Snapshot/Event Clip 생성
11. **로봇팔 상태 확장**
    - `app/equipment/robot_state.py`에서 Robot ROI Motion 기반 `RUNNING/STOPPED/UNKNOWN`
    - bbox + margin 초기 Zone에서 Static Polygon 중기, 상태 기반 Dynamic Zone 고도화 순으로 진행
12. **컨베이어 상태 확장**
    - `app/equipment/conveyor_state.py`에서 ROI Optical Flow와 방향 일관성 분석
13. **PPE Trigger 추론**
    - 작업자가 위험 Zone에 접근할 때만 `2model_20260921.pt` 실행
    - 산업설비 모델과 PPE 모델의 class space 및 결과 계약을 분리
14. **API와 Frontend**
    - FastAPI, WebSocket, Zone CRUD, 이벤트 조회 및 시스템 상태
15. **TensorRT와 Jetson 배포**
    - ONNX 결과 일치 검증 후 Jetson에서 FP16 engine 재생성
    - PyTorch/ONNX/TensorRT 정확도, FPS, latency, RAM/VRAM과 전력 비교
16. **후속 고도화**
    - TTC, 미래경로 Dynamic Zone, Near-Miss, Pose, Fall, Adaptive Inference, Heatmap 및 Risk Report

## 8. Edge 실행 정책

| 기능 | 기본 실행 방식 | 비용 기준 |
|---|---|---|
| 영상 입력 | 상시 | 낮음~중간 |
| 기본 YOLO Detection | 상시 또는 frame skip | 높음 |
| ByteTrack | Detection 갱신 시 | 낮음 |
| Track History | 매 프레임 | 낮음 |
| Geometry / Zone / Risk | 매 프레임 | 낮음 |
| 지게차 상태 | Tracking 갱신 시 | 낮음 |
| 로봇팔 Frame Difference | Robot ROI에서 N프레임마다 | 중간 |
| 컨베이어 Optical Flow | Belt ROI에서 N프레임마다 | 중간~높음 |
| PPE Detection | PPE 확인 Zone 접근 시 Trigger | 높음 |
| 고해상도 ROI 재추론 | 낮은 confidence일 때 Trigger | 높음 |
| Pose | 접근·침입·쓰러짐 후보 발생 시 Trigger | 높음 |
| Snapshot | 신규 또는 등급 상승 이벤트 시 | I/O |
| Event Clip | 이벤트 발생 시 | 메모리·I/O |
| Report | 이벤트 확정 후 비동기 | 중간 |

새 고비용 기능을 추가할 때는 입력 ROI를 줄일 수 있는지, N프레임 주기로 실행할 수 있는지, 위험상황에서만 활성화할 수 있는지 먼저 검토한다.

## 9. 설정 파일 책임

- `config/cameras.yaml`: camera ID, 입력 경로/URL, 해상도, FPS, reconnect와 buffer 설정
- `config/model.yaml`: 모델 경로, backend, device, 입력 크기, confidence, IoU와 class mapping
- `config/zones.yaml`: 카메라별 Polygon, 설비별 margin, normalized 좌표와 smoothing 설정
- `config/system.yaml`: 로그, DB, 저장 경로, cooldown, tracking history와 성능 설정

threshold와 환경별 경로를 Python 코드 곳곳에 하드코딩하지 않는다. 설정 로딩 전인 임시 값은 이름에 `TEMP_`를 붙이고 제거 조건을 문서에 남긴다. RTSP 인증정보, 비밀번호와 토큰은 커밋하지 않는다.

## 10. Python 구현 규정

- Python 3.10 이상에서 동작하는 type hint를 사용한다.
- 공개 데이터 계약은 가능한 한 dataclass와 Enum으로 정의한다.
- 전달 중 변경되면 안 되는 결과 객체에는 `frozen=True`를 고려한다. NumPy 배열은 frozen dataclass 내부에서도 변경 가능하다.
- 함수와 클래스 이름은 역할이 드러나게 작성하고 한 함수가 여러 계층의 책임을 갖지 않게 한다.
- 설명 없는 dict, magic number와 전역 mutable state를 피한다.
- 파일 경로는 `pathlib.Path`를 사용한다.
- 운영 로그는 `logging`을 사용한다. 라이브러리 모듈에서 `print()`를 사용하지 않는다.
- 주석은 알고리즘의 이유, 좌표계, 성능상 선택처럼 코드만으로 드러나지 않는 내용에만 짧게 작성한다.
- 외부 자원(VideoCapture, DB, MQTT, TensorRT Context)은 명시적으로 종료하거나 context manager를 제공한다.
- 광범위한 `except Exception: pass`를 사용하지 않는다. 복구 가능한 장애만 경계에서 처리하고 context와 함께 로그를 남긴다.
- Backend별 코드는 공통 인터페이스 밖으로 누출하지 않는다.

## 11. 테스트 규정

- 테스트 도구는 `pytest`를 기본으로 한다.
- 정상 입력, 경계값, 빈 입력, 잘못된 입력 및 자원 해제를 검증한다.
- 단위 테스트는 가능한 한 실제 GPU나 대형 모델 없이 실행할 수 있어야 한다.
- 모델 또는 실제 영상이 필요한 테스트는 단위 테스트와 분리하고 marker를 사용한다.
- Geometry와 Risk Rule은 순수 함수 중심으로 만들고 deterministic test를 작성한다.
- 시간 기반 로직은 실제 sleep 대신 주입 가능한 timestamp 또는 clock을 사용한다.
- 비동기 저장이나 이벤트 처리는 종료 시 flush 여부를 검증한다.

권장 테스트 파일:

```text
tests/test_detector.py
tests/test_video_source.py
tests/test_tracking.py
tests/test_trajectory.py
tests/test_equipment_state.py
tests/test_zones.py
tests/test_risk.py
tests/test_event_manager.py
tests/test_event_repository.py
tests/test_ring_buffer.py
tests/test_frame_processor.py
tests/test_api.py
```

## 12. 기능별 완료 조건

1. 입력과 출력 타입이 명확하다.
2. 다른 계층의 내부 구현을 직접 참조하지 않는다.
3. 정상·경계·실패 흐름이 테스트됐다.
4. 영상이나 모델 자원이 정상적으로 해제된다.
5. 로그에 camera ID와 필요한 track/event context가 포함된다.
6. 기존 파이프라인과 연결되는 인터페이스가 문서화됐다.
7. Edge 실행 비용과 상시/Trigger 실행 여부가 결정됐다.
8. 관련 명세와 설정 예시가 갱신됐다.
9. 관련 테스트가 통과한다.
10. 실제 영상 통합 단계라면 FPS와 latency 변화가 기록된다.

## 13. Git 및 팀 협업 규정

- 작업 시작 전 branch, 변경 파일과 충돌 가능성을 확인한다.
- 다른 팀원의 미완료 변경을 임의로 되돌리거나 덮어쓰지 않는다.
- 한 커밋에는 하나의 기능 또는 하나의 명확한 수정 목적만 담는다.
- 커밋 메시지는 `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:` 형식을 권장한다.
- 코드 변경과 무관한 대용량 영상, 모델, 로그, Snapshot, DB 파일을 함께 커밋하지 않는다.
- 새 대용량 모델이나 영상을 추가하기 전에 필요성과 라이선스를 확인하고 필요하면 Git LFS를 사용한다.
- `.env`, RTSP 비밀번호, API key, MQTT credential 등 비밀정보를 커밋하지 않는다.
- Pull Request에는 구현 기능, 변경 파일, 테스트 결과, 성능 영향과 남은 제한사항을 기록한다.
- 인터페이스를 변경하면 모든 호출부와 테스트를 같은 변경에서 수정한다.
- 명세를 변경한 커밋에는 변경 이유를 설명한다.

## 14. 새 기능 작업 전 체크리스트

1. 어느 모듈과 정확히 어느 파일에 위치하는가?
2. 입력 데이터와 좌표계·시간 단위는 무엇인가?
3. 출력 데이터와 상태값은 무엇인가?
4. 다른 모듈과 어떤 인터페이스로 연결되는가?
5. Jetson에서 예상되는 연산·메모리·I/O 비용은 어느 정도인가?
6. 상시 실행, N프레임 주기 또는 Trigger 방식 중 무엇인가?
7. Temporal Smoothing과 cooldown이 필요한가?
8. 어떤 단위·통합 테스트로 완료를 증명할 것인가?
9. 장애 발생 시 중단, 건너뛰기 또는 재시도 중 무엇을 적용할 것인가?
10. 기능 명세 또는 설정 파일 갱신이 필요한가?
