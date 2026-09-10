# Vision Guard 기능 명세 및 개발 프롬프트

> 문서 상태: Living Specification  
> 프로젝트명: Vision Guard  
> 캐치프레이즈: 사고말고경고

이 문서는 Vision Guard의 설계·개발에서 가장 중요한 기준으로 사용한다. 요구사항, 아키텍처, 인터페이스, 개발 우선순위 또는 Edge 제약이 변경될 때 이 문서를 함께 갱신한다.

## 역할과 목표

당신은 산업용 Edge Vision AI 시스템을 설계하고 개발하는 시니어 AI/백엔드 엔지니어다.

아래 기능 명세를 기준으로 **Vision Guard** 산업현장 안전 모니터링 시스템을 설계 및 개발해야 한다.

프로젝트의 핵심은 단순 YOLO 객체 탐지가 아니라 다음 과정까지 이어지는 실시간 Edge Vision 안전 시스템을 구축하는 것이다.

```text
CCTV 영상
→ 객체 탐지
→ 객체 추적
→ 설비 작동상태 분석
→ 작업자-설비 공간관계 분석
→ 위험 판단
→ 경고
→ 로그 및 위험상황 보고서 생성
```

## 1. 프로젝트 개요

산업현장의 CCTV 영상을 실시간 분석하여 작업자와 산업 설비를 탐지·추적하고, 설비의 작동 상태와 작업자의 접근 상태를 분석하여 위험상황을 판단한다.

위험상황 발생 시 다음을 수행한다.

- 실시간 경고
- 위험 이벤트 로그 저장
- 스냅샷 및 영상 저장
- 위험상황 보고서 생성

최종적으로 NVIDIA Jetson 계열 Edge Device에서 동작하는 것을 목표로 한다.

## 2. 주요 탐지 객체

YOLO 기반 Object Detection을 사용한다.

### 작업자

- person

### 산업 설비

- forklift
- conveyor belt
- robot arm

### PPE

- helmet
- head
- gloves
- body
- safety_vest
- harness_body
- 필요 시 safety vest / safety shoes 추가 가능

## 3. 전체 처리 파이프라인

```text
CCTV / RTSP / Video
        ↓
Frame 입력
        ↓
YOLO Object Detection
        ↓
ByteTrack Object Tracking
        ↓
Track History / Trajectory
        ↓
Equipment State Analysis
        ↓
Zone / Spatial Analysis
        ↓
Risk Engine
        ↓
Risk Event
        ↓
 ┌─────────────┬──────────────┬───────────────┐
 ↓             ↓              ↓
Alert        Event Log     Risk Report
                               +
                         Snapshot / Clip
```

## 4. 영상 입력

지원 입력:

- 로컬 동영상
- Webcam
- CCTV
- RTSP Stream

`video_source.py`를 통해 공통 인터페이스를 제공하고, RTSP 관련 연결·재접속·timeout·buffering 처리는 `rtsp_source.py`에서 처리한다.

향후 Jetson 환경에서는 OpenCV + GStreamer 사용을 고려한다.

## 5. YOLO 객체 탐지

YOLO 추론 결과는 backend에 관계없이 동일한 `Detection` 객체 형태로 반환한다.

```python
Detection(
    class_id,
    class_name,
    confidence,
    bbox=(x1, y1, x2, y2),
)
```

PyTorch `.pt` 모델과 TensorRT `.engine` 모델을 동일 인터페이스에서 사용할 수 있도록 한다.

- Local/Server: PyTorch
- Jetson: TensorRT

## 6. PPE 탐지

작업자의 PPE 착용 여부를 판단한다.

```text
Person Detection
→ Worker ROI Crop
→ PPE Detection
→ Worker와 PPE Spatial Matching
```

작은 객체인 helmet, gloves, safety shoes 등의 탐지 성능 향상을 위해 전체 프레임보다 Person ROI 기반 탐지를 우선 고려한다.

향후 선택적으로 다음 기능을 사용할 수 있도록 구조를 설계한다.

- Person ROI Crop
- ROI Margin
- High Resolution Re-Inference
- Confidence 기반 재추론
- Temporal Voting

PPE 모델을 반드시 모든 프레임에서 실행할 필요는 없다.

## 7. 객체 추적

ByteTrack을 이용하여 작업자와 이동 설비의 Track ID를 유지한다.

주요 추적 대상:

- Worker
- Forklift
- 필요한 경우 Robot Arm

Tracked Object는 다음 정보를 가진다.

- track_id
- class
- bbox
- center
- bottom_center
- confidence
- timestamp

각 Track ID별 최근 N frame의 위치 정보를 저장한다. Track History를 이용하여 다음을 계산한다.

- 이동 여부
- 이동 방향
- 이동 속도
- trajectory
- zone 체류시간
- 접근 여부

## 8. 지게차 작동상태 판단

지게차는 Tracking 기반으로 이동 여부를 판단한다.

```text
Forklift Detection
→ ByteTrack
→ 최근 N frame 좌표
→ 이동거리 및 속도 계산
→ 상태 판단
```

상태:

- MOVING
- STOPPED
- UNKNOWN

Detection jitter를 방지하기 위해 한 프레임이 아니라 최근 일정 프레임의 평균 이동량을 사용한다.

향후 Homography가 적용되면 Pixel 단위가 아니라 m/s 단위 실제 속도 계산을 지원한다.

## 9. 로봇팔 작동상태 판단

로봇팔은 전체 Bounding Box가 이동하지 않을 가능성이 높으므로 BBox Tracking만으로 작동 여부를 판단하지 않는다.

```text
Robot Arm Detection
→ Robot ROI Crop
→ Frame Difference 또는 Optical Flow
→ Motion Score
→ 상태 판단
```

상태:

- RUNNING
- STOPPED
- UNKNOWN

초기 MVP에서는 Frame Difference를 사용하고, 필요한 경우 Farneback Optical Flow 등으로 개선한다.

향후 실제 산업현장 적용 시 PLC / Robot Controller 상태정보를 받을 수 있다면 Vision 결과와 상태정보를 결합할 수 있도록 설계한다.

## 10. 컨베이어 벨트 작동상태 판단

컨베이어의 Bounding Box 자체는 이동하지 않기 때문에 벨트 내부 영역의 Motion을 분석한다.

```text
Conveyor Detection
→ Belt ROI
→ Optical Flow
→ Flow Magnitude
  + Flow Direction Consistency
→ RUNNING / STOPPED
```

보조 방법으로 다음을 사용할 수 있다.

- Belt Marker Tracking
- Conveyor 위 물체 Tracking
- PLC Run Signal

상태:

- RUNNING
- STOPPED
- UNKNOWN

## 11. 위험구역 관리

설비별 Danger Zone을 관리한다.

`ZoneManager`가 다음 기능을 담당한다.

- Zone 생성
- 수정
- 삭제
- Camera별 Zone 관리
- Config Load / Save
- Polygon 관리

초기에는 Static Polygon을 우선 구현한다.

### Conveyor

Static Danger Polygon을 사용한다. Conveyor가 RUNNING인 상태에서 작업자가 Danger Zone에 진입하면 위험도를 증가시킨다.

### Robot Arm

- 초기: YOLO가 탐지한 Robot Arm Bounding Box를 기준으로 Safety Zone을 생성한다.
  - 입력: Robot Arm `Detection.bbox`
  - 처리: Bounding Box를 설정된 margin만큼 확장하고 영상 경계 안으로 제한하여 사각형 Polygon으로 변환한다.
  - 출력: `zone_id`, `camera_id`, `equipment_track_id`, `polygon`, `timestamp`를 가진 Robot Safety Zone
  - 인터페이스: `app/zones/robot_arm.py`에서 Zone을 생성하고 `app/zones/zone_manager.py`를 통해 Risk Engine에 전달한다.
  - 설정: Bounding Box 확장 비율 또는 pixel margin은 `config/zones.yaml`에서 관리한다.
  - 안정화: Detection jitter로 Zone이 흔들리지 않도록 최근 N frame의 Bounding Box에 Temporal Smoothing을 적용한다.
  - Edge 비용: 단순 좌표 확장 및 Polygon 변환이므로 낮으며, 유효한 Robot Arm Detection이 있는 프레임마다 실행한다.
- 중기: Camera별 Static Safety Polygon 또는 사전에 정의한 작업 반경 기반 Zone을 사용한다.
- 고도화: Robot 작동상태, 작업반경, Vision Motion 정보 및 선택적인 PLC / Robot Controller 상태정보를 결합한 Dynamic Safety Zone을 사용한다.

### Forklift

이동 객체이므로 고정 Polygon보다 Tracking 기반 Dynamic Zone을 적용할 수 있도록 한다.

```text
Forklift Track
→ 이동방향
→ 미래 이동경로
→ Buffer
→ Dynamic Danger Zone
```

## 12. 공간 분석

`geometry.py`에서는 다음 기능을 담당한다.

- bbox center
- bbox bottom-center
- point in polygon
- point-to-polygon distance
- object distance
- line intersection
- trajectory buffer

작업자 위치는 기본적으로 Bounding Box의 bottom-center를 사용한다. CCTV 환경에서 작업자의 바닥 위치를 나타내는 데 center보다 적합하기 때문이다.

## 13. Camera Calibration / Homography

향후 다음 기능을 지원할 수 있도록 calibration 모듈을 분리한다.

- Homography
- Perspective Transform
- Bird's-Eye View
- Pixel → World Coordinate
- 실제 거리 계산
- 실제 속도 계산

초기 MVP에서는 필수가 아니지만, 지게차 거리 및 TTC를 정확하게 계산하기 위해 추후 적용한다.

## 14. 위험 판단 엔진

Risk Engine은 다음 정보를 종합한다.

- Worker 위치
- Equipment 위치
- Equipment State
- Danger Zone
- Worker 이동 방향
- Equipment 이동 방향
- Distance
- PPE
- Zone 체류시간

Risk Rule은 별도 `risk_rules.py`에서 관리한다.

예시 1:

```text
IF Conveyor == RUNNING
AND Worker inside Conveyor Danger Zone
THEN Risk = HIGH
```

예시 2:

```text
IF RobotArm == RUNNING
AND Worker inside Robot Safety Zone
THEN Risk = HIGH
```

예시 3:

```text
IF Forklift == MOVING
AND Worker close to Forklift
THEN Risk = WARNING / CRITICAL
```

위험등급:

- NORMAL
- CAUTION
- WARNING
- CRITICAL

## 15. 실시간 경고

Risk Event 발생 시 Alert Manager를 호출한다.

지원 기능:

- 화면 Warning
- 위험 객체 Highlight
- Warning Text
- 경고음

예시:

```text
CRITICAL
Worker #7
Forklift #3 접근 위험
```

동일 이벤트가 매 프레임 반복 발생하지 않도록 Alert Cooldown / Event Deduplication을 적용한다.

## 16. 위험 이벤트 로그

Risk Event 발생 시 DB에 기록한다.

최소 저장 정보:

- event_id
- timestamp
- camera_id
- worker_track_id
- equipment_track_id
- equipment_type
- equipment_state
- risk_type
- risk_level
- distance
- zone_id
- PPE 상태
- snapshot_path
- video_clip_path

개발 초기에는 SQLite 사용을 권장한다.

## 17. 위험 상황 Snapshot

위험 이벤트 발생 시 해당 Frame을 이미지로 저장한다.

예시:

```text
20260905_142311_forklift_worker17.jpg
```

DB에서 Snapshot 파일 위치를 참조할 수 있도록 한다.

## 18. 위험상황 영상 저장

Edge Device에서 Ring Buffer를 운영한다.

예시:

- 위험 발생 전 5초
- 위험 발생 후 5초
- 총 10초 영상을 Event Clip으로 저장

전체 CCTV 영상을 계속 저장하는 것보다 저장공간과 네트워크 사용량을 줄이는 방향을 우선한다.

## 19. 위험상황 보고서

Risk Event 데이터를 이용하여 자동 보고서를 생성한다.

보고서에 포함할 내용:

- 발생 시각
- Camera
- Worker ID
- Equipment ID
- Equipment 종류
- Equipment 상태
- 위험 유형
- 위험등급
- 최소거리
- PPE 상태
- Snapshot
- Event Clip
- 위험 상황 요약

초기에는 Template 기반으로 생성하고, LLM/VLM 기반 자연어 보고서는 향후 고도화 기능으로 처리한다.

## 20. 관리자 기능

관리자 Dashboard에서 다음 기능을 제공한다.

- 실시간 CCTV 영상
- Detection Bounding Box
- Track ID
- Equipment State
- Danger Zone
- 현재 위험등급
- 실시간 Alert
- 위험 이벤트 기록
- 위험상황 보고서
- Camera 설정
- Zone 설정
- 시스템 상태

Frontend는 위험판단 로직을 포함하지 않는다. 모든 안전 판단은 Backend / Edge AI에서 수행한다.

## 21. API

FastAPI를 사용한다.

예상 API:

```text
GET /events
GET /events/{id}

GET /cameras
POST /cameras

GET /zones
POST /zones
PUT /zones/{id}
DELETE /zones/{id}

GET /system/status
```

실시간 Alert와 시스템 상태 전달에는 WebSocket을 사용할 수 있다.

## 22. Edge Device

최종 목표는 NVIDIA Jetson 계열 디바이스에서 동작하는 것이다.

개발 순서:

```text
PyTorch .pt
→ ONNX
→ TensorRT FP16
→ Jetson
```

FP16을 우선 적용하고, 필요한 경우 INT8을 추가 검토한다.

다음 성능을 측정해야 한다.

- FPS
- Average Latency
- P95 Latency
- GPU Usage
- CPU Usage
- RAM / VRAM
- Power Consumption

## 23. Edge Adaptive Inference

본 프로젝트의 주요 고도화 방향 중 하나이다. 모든 AI 기능을 항상 실행하지 않고 현재 상황에 따라 필요한 분석 기능만 활성화한다.

### NORMAL 상태

```text
YOLO + Tracking
```

### Worker가 설비에 접근

```text
Distance / Zone 분석 활성화
```

### PPE 확인이 필요한 Zone 접근

```text
Person ROI Crop → PPE Model ON
```

### 작은 PPE Confidence가 낮음

```text
High Resolution ROI Re-Inference
```

### 작업자가 Conveyor에 접근

```text
Pose Model ON → Hand / Arm 위치 분석
```

### 작업자 자세 급변

```text
Fall Detection Module ON
```

### Forklift 접근

```text
Trajectory Prediction + TTC 분석 ON
```

위험상황이 종료되면 추가 분석 모듈을 비활성화한다.

목적:

- Edge GPU 부하 감소
- 평균 Latency 감소
- 불필요한 추론 방지
- 위험상황에 연산 자원 집중

## 24. 고도화 후보 기능

다음 기능은 1차 MVP 완성 이후 개발한다.

### 24.1 Forklift Trajectory Prediction

Track History를 이용하여 0.5초 / 1초 / 2초 후 지게차 위치를 예측한다.

### 24.2 TTC

Worker와 Forklift의 이동경로를 이용하여 Time To Collision을 계산한다.

```text
TTC < 1 sec → CRITICAL
1~2 sec     → WARNING
2~3 sec     → CAUTION
```

### 24.3 Dynamic Safety Zone

Forklift 속도와 이동방향에 따라 Danger Zone의 크기와 방향을 동적으로 변경한다.

### 24.4 Near-Miss Detection

실제 충돌은 발생하지 않았지만 최소거리, TTC, 접근속도 등이 위험 Threshold를 초과한 사건을 Near-Miss로 저장한다.

### 24.5 위험 Heatmap

Near-Miss 및 위험 이벤트의 좌표를 누적하여 공장 내 반복적으로 위험이 발생하는 위치를 시각화한다.

### 24.6 Pose 기반 Hand / Arm Intrusion

Conveyor 또는 Robot Arm 근처 작업자에게만 Pose를 실행한다. Wrist / Elbow / Shoulder 좌표를 이용하여 신체 일부가 설비 Danger Zone에 접근하거나 침범하는지 판단한다.

### 24.7 Fall / Collapse Detection

항상 Pose를 실행하지 않는다. 다음과 같은 Trigger 발생 시 Pose 분석을 수행한다.

- Person bbox height 급감
- Aspect Ratio 급변
- 중심점 급격한 하강
- 일정 시간 움직이지 않음

Pose 분석을 통해 실제 쓰러짐 여부를 최종 판단한다.

### 24.8 Abnormal Immobility

Track ID의 위치가 일정 시간 이상 거의 변하지 않으면 작업자의 상태를 확인한다.

## 25. 프로젝트 구조

다음 구조를 기본으로 사용한다.

```text
edge-safety-system/
│
├── app/
│   ├── main.py
│   ├── inference/
│   │   ├── detector.py
│   │   ├── pytorch_backend.py
│   │   ├── tensorrt_backend.py
│   │   └── model_manager.py
│   ├── tracking/
│   │   ├── tracker.py
│   │   └── trajectory.py
│   ├── equipment/
│   │   ├── forklift_state.py
│   │   ├── robot_state.py
│   │   └── conveyor_state.py
│   ├── zones/
│   │   ├── conveyor.py
│   │   ├── robot_arm.py
│   │   ├── forklift.py
│   │   ├── worker_zone.py
│   │   ├── zone_manager.py
│   │   ├── geometry.py
│   │   └── calibration.py
│   ├── risk/
│   │   ├── risk_engine.py
│   │   └── risk_rules.py
│   ├── video/
│   │   ├── video_source.py
│   │   ├── rtsp_source.py
│   │   └── frame_processor.py
│   ├── alerts/
│   │   ├── alert_manager.py
│   │   ├── event_manager.py
│   │   └── mqtt_client.py
│   ├── storage/
│   │   ├── database.py
│   │   ├── event_repository.py
│   │   └── models.py
│   └── api/
│       ├── server.py
│       ├── routes/
│       │   ├── cameras.py
│       │   ├── zones.py
│       │   ├── events.py
│       │   └── system.py
│       └── websocket.py
├── frontend/
├── config/
├── models/
├── data/
└── tests/
```

## 26. 개발 우선순위

다음 순서로 개발한다.

1. Phase 1 — YOLO Detection
2. Phase 2 — ByteTrack
3. Phase 3 — Track History / Trajectory
4. Phase 4 — Forklift MOVING / STOPPED
5. Phase 5 — Robot Arm RUNNING / STOPPED
6. Phase 6 — Conveyor RUNNING / STOPPED
7. Phase 7 — Static Polygon Zone
8. Phase 8 — Worker-Zone Spatial Analysis
9. Phase 9 — Risk Engine
10. Phase 10 — Alert
11. Phase 11 — DB Event Log
12. Phase 12 — Snapshot / Event Clip
13. Phase 13 — Risk Report
14. Phase 14 — Forklift Trajectory / TTC
15. Phase 15 — Dynamic Zone / Near-Miss
16. Phase 16 — Adaptive Inference
17. Phase 17 — Pose / Fall / Hand Intrusion
18. Phase 18 — FastAPI / WebSocket
19. Phase 19 — Admin Dashboard
20. Phase 20 — TensorRT / Jetson Optimization

## 27. 개발 원칙

1. Detection, Tracking, Zone, Risk를 서로 강하게 결합하지 않는다.
2. 각 기능을 독립적인 모듈로 개발한다.
3. Risk Engine만 여러 모듈의 결과를 통합한다.
4. UI에는 위험 판단 로직을 넣지 않는다.
5. 모델의 출력 구조를 표준화한다.
6. PyTorch와 TensorRT가 동일한 Detector Interface를 사용하도록 한다.
7. 모든 AI 모델을 동시에 실행하지 않는다.
8. Edge 환경에서는 위험도에 따라 AI 모듈 실행 여부를 동적으로 결정한다.
9. 한 프레임만으로 설비의 작동상태나 위험여부를 판단하지 않는다. Temporal Smoothing을 적용한다.
10. 설비 상태값은 Boolean만 사용하지 말고 RUNNING / STOPPED / UNKNOWN 등의 상태를 사용한다.
11. 위험 이벤트는 매 프레임 생성하지 않고 Event State와 Cooldown을 사용한다.
12. 처음부터 모든 기능을 개발하지 않는다. 먼저 하나의 위험 시나리오를 End-to-End로 완성한 후 확장한다.

## 28. 1차 MVP 완료 조건

최소 다음 흐름이 실제 영상에서 End-to-End로 동작해야 한다.

```text
CCTV / Video
↓
YOLO
↓
ByteTrack
↓
Equipment State Detection
↓
Danger Zone
↓
Worker Intrusion
↓
Risk Engine
↓
Alert
↓
Event Log
↓
Snapshot 저장
```

1차 MVP가 완성된 이후에 Trajectory, TTC, Pose, Fall Detection, Adaptive Inference 등을 추가한다.

## 설계·구현 시 필수 검토사항

새로운 기능을 제안하거나 코드를 작성할 때는 반드시 다음을 함께 판단한다.

1. 기존 모듈 중 어디에 위치하는가
2. 입력은 무엇인가
3. 출력은 무엇인가
4. 다른 모듈과 어떤 인터페이스로 연결되는가
5. Edge 환경에서 계산 비용이 어느 정도인가
6. 항상 실행해야 하는가, Trigger 방식으로 실행할 수 있는가

모든 설계는 최종적으로 Jetson Edge Device에서 실시간으로 동작할 수 있다는 조건을 우선한다.
