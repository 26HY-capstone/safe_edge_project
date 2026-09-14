# Vision Guard 프로젝트 구조

이 문서는 Vision Guard 저장소가 어떻게 나뉘어 있는지 설명한다.
기능과 아키텍처 기준은 `docs/VISION_GUARD_SPEC.md`에 두고, 여기서는 코드와 설정 파일의 위치를 정리한다.

## 전체 구조

```text
.
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── app/
├── config/
├── data/
├── docs/
├── frontend/
├── models/
├── requirements.txt
└── tests/
```

- `AGENTS.md`: 저장소에서 지켜야 할 개발 규칙이다.
- `CLAUDE.md`: Claude 계열 도구가 참고하는 지침이다.
- `README.md`: 프로젝트 소개와 실행 방법을 적는 문서다.
- `requirements.txt`: Python 패키지 목록이다.
- `docker-compose.yml`: 로컬 서비스나 배포 환경을 묶을 때 쓰는 구성 파일이다.

## 문서

```text
docs/
├── PROJECT_STRUCTURE.md
└── VISION_GUARD_SPEC.md
```

- `docs/VISION_GUARD_SPEC.md`: Vision Guard가 무엇을 만들고 어떤 기준으로 동작해야 하는지 적은 명세다.
- `docs/PROJECT_STRUCTURE.md`: 저장소 안의 파일과 폴더가 어떤 역할인지 설명하는 문서다.

## 애플리케이션 코드

```text
app/
├── alerts/
├── api/
├── inference/
├── main.py
├── risk/
├── storage/
├── tracking/
├── video/
└── zones/
```

`app/`은 실제 실행 코드가 들어가는 곳이다.
영상 입력, 객체 탐지, 추적, 공간 분석, 위험 판단, 저장, API 계층을 역할별로 나눈다.

## 실행 진입점

### `app/main.py`

프로그램을 시작하는 파일이다.
설정을 읽고, 필요한 객체를 만들고, 영상 처리 루프를 실행하고, 종료 시 자원을 정리한다.

## 영상 처리

```text
app/video/
├── frame_processor.py
├── rtsp_source.py
└── video_source.py
```

- `app/video/video_source.py`: 로컬 영상이나 웹캠에서 프레임을 읽어 `FramePacket`으로 만든다.
- `app/video/frame_processor.py`: 프레임을 읽고 detector를 호출한 뒤, bbox와 FPS/지연시간 표시를 그린다.
- `app/video/rtsp_source.py`: RTSP 카메라 입력을 다룬다. 연결, timeout, buffering, 재접속 같은 처리가 여기에 들어간다.

## 객체 탐지

```text
app/inference/
├── detector.py
├── model_manager.py
├── pytorch_backend.py
└── tensorrt_backend.py
```

- `app/inference/detector.py`: 탐지 결과 형식인 `Detection`과 공통 detector 인터페이스가 있는 파일이다.
- `app/inference/pytorch_backend.py`: PyTorch YOLO 모델을 불러와 추론하고 `Detection` 목록으로 바꾼다.
- `app/inference/tensorrt_backend.py`: Jetson에서 쓸 TensorRT engine 추론 코드가 들어간다.
- `app/inference/model_manager.py`: 어떤 backend를 쓸지 고르고, 모델 경로와 threshold, class mapping을 관리한다.

## 객체 추적

```text
app/tracking/
├── tracker.py
└── trajectory.py
```

- `app/tracking/tracker.py`: ByteTrack으로 객체 ID를 유지하고 `TrackedObject`를 만든다.
- `app/tracking/trajectory.py`: track별 위치 이력을 쌓고 이동거리, 방향, 속도, 체류시간을 계산한다.

## 공간 분석과 Zone

```text
app/zones/
├── calibration.py
├── conveyor.py
├── forklift.py
├── geometry.py
├── robot_arm.py
├── worker_zone.py
└── zone_manager.py
```

- `app/zones/geometry.py`: 좌표, polygon 포함 여부, 거리, 교차 같은 기본 계산을 모아둔다.
- `app/zones/zone_manager.py`: 카메라별 Zone을 만들고 읽고 저장하는 관리 코드다.
- `app/zones/worker_zone.py`: 작업자의 bottom-center 위치로 Zone 진입, 이탈, 체류시간을 계산한다.
- `app/zones/forklift.py`: 지게차 주변 안전 buffer와 이동방향 기반 danger zone을 만든다.
- `app/zones/robot_arm.py`: 로봇팔 bbox를 기준으로 safety zone을 만들고 흔들림을 완화한다.
- `app/zones/conveyor.py`: 컨베이어 위험구역과 작업자 접근 여부를 계산한다.
- `app/zones/calibration.py`: homography, bird's-eye view, pixel-to-world 변환을 다룬다.

## 위험 판단

```text
app/risk/
├── risk_engine.py
└── risk_rules.py
```

- `app/risk/risk_rules.py`: 설비별 위험 조건, 거리 기준, 시간 기준, 위험등급 규칙이 들어간다.
- `app/risk/risk_engine.py`: 작업자, 설비 상태, Zone, 거리, 이동 경로, PPE 결과를 모아 최종 위험도를 계산한다.

## 이벤트와 경고

```text
app/alerts/
├── alert_manager.py
├── event_manager.py
└── mqtt_client.py
```

- `app/alerts/event_manager.py`: 위험 판단 결과를 이벤트로 바꾸고 중복 생성, cooldown, 등급 상승을 처리한다.
- `app/alerts/alert_manager.py`: 화면 표시, 객체 강조, 경고 문구, 경고음 같은 사용자 알림을 다룬다.
- `app/alerts/mqtt_client.py`: 중앙 관제나 외부 장치로 이벤트와 장치 상태를 보낼 때 쓴다.

## 저장소

```text
app/storage/
├── database.py
├── event_repository.py
└── models.py
```

- `app/storage/models.py`: SQLite에 저장할 이벤트 테이블 구조가 들어간다.
- `app/storage/database.py`: DB 연결, 테이블 생성, transaction 처리를 맡는다.
- `app/storage/event_repository.py`: 이벤트 저장, 조회, 수정, 종료 처리를 제공한다.

## API

```text
app/api/
├── routes/
│   ├── cameras.py
│   ├── events.py
│   ├── system.py
│   └── zones.py
├── server.py
└── websocket.py
```

- `app/api/server.py`: FastAPI 앱을 만들고 route와 websocket을 연결한다.
- `app/api/websocket.py`: 실시간 경고와 시스템 상태를 프론트엔드로 보낸다.
- `app/api/routes/cameras.py`: 카메라 목록, 등록, 수정, 연결 상태 API다.
- `app/api/routes/zones.py`: Zone 조회, 생성, 수정, 삭제 API다.
- `app/api/routes/events.py`: 이벤트 목록, 상세, 필터 조회 API다.
- `app/api/routes/system.py`: FPS, 지연시간, 모델, backend, 장치 상태 API다.

## 설정 파일

```text
config/
├── cameras.yaml
├── model.yaml
├── system.yaml
└── zones.yaml
```

- `config/cameras.yaml`: 카메라 ID, 영상 경로, 웹캠 번호, RTSP 주소, loop, 해상도, FPS 설정이다.
- `config/model.yaml`: 모델 경로, backend, device, input size, confidence/IoU threshold, class mapping 설정이다.
- `config/zones.yaml`: 카메라별 polygon, 로봇팔 margin, 지게차 buffer, 작업자 체류시간 설정이다.
- `config/system.yaml`: 로그, DB 경로, snapshot 저장 위치, tracking 이력, risk cooldown, 목표 FPS 설정이다.

코드에는 로직을 두고, YAML에는 환경마다 바뀌는 값을 둔다.
영상 경로, threshold, 실행 주기, 저장 경로처럼 바뀔 수 있는 값은 설정 파일에서 관리한다.

## 데이터

```text
data/
├── events.db
├── logs/
├── samples/
│   └── factory_floor_demo.mp4
└── snapshots/
```

- `data/samples/factory_floor_demo.mp4`: 샘플 공장 CCTV 영상이다. 초기 파이프라인 확인에 쓴다.
- `data/events.db`: SQLite 이벤트 DB 파일이다.
- `data/logs/`: 실행 로그가 쌓이는 위치다.
- `data/snapshots/`: 위험 이벤트 snapshot이 저장되는 위치다.

## 모델

```text
models/
├── best.engine
├── best.onnx
└── best.pt
```

- `models/best.pt`: 로컬 개발과 기본 검증에 쓰는 PyTorch YOLO 모델이다.
- `models/best.onnx`: ONNX로 변환한 모델이다.
- `models/best.engine`: Jetson TensorRT 실행에 쓰는 engine 모델이다.

## 프론트엔드

```text
frontend/
├── package.json
├── public/
└── src/
```

프론트엔드는 실시간 영상, 탐지 박스, 위험구역, 경고, 이벤트 기록을 보여주는 관리자 화면이다.
위험 판단 로직은 프론트엔드가 아니라 백엔드와 엣지 파이프라인에 둔다.

## 테스트

```text
tests/
├── test_risk.py
├── test_tracking.py
└── test_zones.py
```

- `tests/test_detector.py`: `Detection` 형식과 detector 입력 검증 테스트다.
- `tests/test_video_source.py`: 로컬 영상과 웹캠 입력 테스트다.
- `tests/test_frame_processor.py`: `VideoSource`와 `Detector` 연결 테스트다.
- `tests/test_tracking.py`: 객체 ID 유지와 추적 결과 테스트다.
- `tests/test_zones.py`: geometry와 Zone 판정 테스트다.
- `tests/test_risk.py`: 위험 규칙과 위험등급 테스트다.

## 개발 순서

1. 설정 파일과 Python 패키지 기반을 정리한다.
2. 영상 입력과 `Detection` 형식을 검증한다.
3. PyTorch YOLO backend를 연결한다.
4. 샘플 영상과 노트북 카메라로 detection 파이프라인을 확인한다.
5. ByteTrack과 trajectory 이력을 연결한다.
6. 지게차 상태 판단, Zone 분석, Risk Engine을 붙인다.
7. Event, Alert, DB, Snapshot까지 하나의 MVP 흐름을 만든다.
8. 로봇팔, 컨베이어, PPE crop, pose, API, frontend로 확장한다.
