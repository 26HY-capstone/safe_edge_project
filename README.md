# Vision Guard

산업현장 CCTV 영상에서 작업자와 설비를 탐지·추적하고, 위험구역 침입을 판단해 경고하는 Edge Vision 안전 시스템입니다.

## 현재 구현 범위

현재 브랜치에는 영상 입력, Detection 계약, Tracking/Trajectory, Polygon Zone, Risk Engine, Event 중복 제거와 로그 Alert 연결이 구현되어 있습니다. 실제 YOLO 실행을 위한 PyTorch/TensorRT backend와 모델 파일, 설비 상태 분석, DB/API는 후속 구현 대상입니다.

## 개발 환경

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

대용량 영상과 모델은 Git에 포함하지 않습니다. 로컬 영상은 `data/samples/`에 두고 `config/cameras.yaml`의 `source`를 맞추며, 모델은 `models/`에 두고 `config/model.yaml`에서 경로를 지정합니다.

## 2x2 입력 확인 데모

```powershell
python -m app.demo.multi_camera_viewer --camera-config config/cameras.yaml
```

metadata JSON 기반 ceiling/eye 샘플을 함께 표시하려면 `--sample-dir`과 선택적인 `--seed`를 지정합니다. 창에서 `q`를 누르면 종료됩니다.

상세 기능 기준은 `docs/VISION_GUARD_SPEC.md`, 파일별 역할은 `docs/PROJECT_STRUCTURE.md`, 협업 규정은 `AGENTS.md`를 참고합니다.
