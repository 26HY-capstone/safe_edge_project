# Windows NVIDIA GPU 가속 실행

이 브랜치는 ONNX Runtime의 `CUDAExecutionProvider`를 우선 사용한다.
CUDA provider를 사용할 수 없으면 `CPUExecutionProvider`로 자동 전환한다.
경고음은 Windows 기본 `winsound` 모듈로 비동기 재생한다.

## 준비

- Windows 10 또는 Windows 11
- Python 3.10 이상
- NVIDIA GPU와 최신 드라이버
- ONNX Runtime 버전에 맞는 CUDA 및 cuDNN
- `models/model.onnx` 로컬 배치

CPU용 ONNX Runtime과 GPU용 패키지를 동시에 설치하지 않는다. 기존 환경에
CPU 패키지가 있다면 제거한 뒤 이 브랜치의 의존성을 설치한다.

```powershell
cd C:\work\safe_edge_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install -r requirements.txt
```

설치 후 provider 목록을 확인한다.

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

출력에 `CUDAExecutionProvider`가 있어야 NVIDIA GPU 추론을 사용할 수 있다.

## 실행

```powershell
python -m app.main
```

영상 타일 오른쪽 위의 `OFF` 버튼을 눌러 입력을 활성화한다. 종료 키는 `q`다.

## 확인 사항

- 시작 로그의 `active` provider 첫 항목이 `CUDAExecutionProvider`인지 확인한다.
- CPU fallback 로그가 출력되면 GPU 가속이 적용되지 않은 상태다.
- `nvidia-smi`로 실행 중 GPU 메모리와 사용률을 확인한다.
- CUDA와 cuDNN 버전은 설치한 `onnxruntime-gpu` 버전의 호환표와 일치해야 한다.
- 실제 FPS와 latency를 CPU 설정과 비교해 기록한다.
