# macOS GPU 가속 실행

이 브랜치는 ONNX Runtime의 `CoreMLExecutionProvider`를 우선 사용한다.
CoreML은 지원되는 Mac에서 CPU, GPU와 Apple Neural Engine 중 모델 연산에
적합한 장치를 선택한다. CoreML provider를 사용할 수 없으면
`CPUExecutionProvider`로 자동 전환한다.

## 준비

- macOS 12 이상 권장
- Python 3.10 이상
- Apple Silicon 권장
- `models/model.onnx` 로컬 배치

```bash
cd /path/to/safe_edge_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

공식 macOS `onnxruntime` wheel에는 CoreML Execution Provider가 포함된다.
설치 후 provider 목록을 확인한다.

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

출력에 `CoreMLExecutionProvider`가 있으면 CoreML 가속을 사용할 수 있다.

## 실행

```bash
python -m app.main
```

영상 타일 오른쪽 위의 `OFF` 버튼을 눌러 입력을 활성화한다. 종료 키는 `q`다.
경고음은 macOS 기본 명령인 `afplay`로 재생한다.

## 확인 사항

- 시작 로그의 `active` provider에 `CoreMLExecutionProvider`가 표시되는지 확인한다.
- 첫 실행은 CoreML 모델 변환 때문에 이후 실행보다 오래 걸릴 수 있다.
- CPU fallback 로그가 출력되면 설치된 wheel의 provider 목록을 먼저 확인한다.
- 실제 가속 효과는 모델 연산의 CoreML 지원 범위에 따라 달라지므로 FPS와
  latency를 CPU 설정과 비교해 기록한다.
