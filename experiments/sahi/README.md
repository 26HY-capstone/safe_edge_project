# SAHI Sliced Inference 실험

## 1. 목적

현재 서비스는 YOLOv9t ONNX 모델(`app/inference/onnx_backend.py`, 입력 640x640,
10 class, NMS는 모델 밖에서 직접 수행)을 그대로 쓴다. 이 실험의 목적은 **기존
YOLOv9t 모델에 [SAHI](https://github.com/obss/sahi) 방식의 sliced inference를
적용했을 때, 특히 작은 PPE 객체(head/gloves/helmet/safety_vest/harness_body)의
탐지 성능이 개선되는지를 확인**하는 것이다.

SAHI는 **추가 학습(fine-tuning) 방식이 아니라 inference 방식**이다. 즉 기존
weight를 그대로 두고, "이미지를 어떻게 모델에 넣는가"만 바꾼다 — 원본 이미지를
여러 overlapping slice로 잘라 각각 추론한 뒤, 좌표를 원본 기준으로 복원하고
global NMS로 중복을 정리한다. 작은 객체가 상대적으로 더 크게 보이는 효과가 있다.

실험은 **1차 파일럿**과 **2차 정식**으로 나눈다.

## 2. 1차 파일럿 실험

현재 존재하는 기존 데이터셋과 기존 YOLOv9t 모델을 그대로 쓴다. 목적은 최종
성능 수치 확정이 아니라 **"SAHI를 적용할 가치가 있는가?"**를 확인하는 것이다
(코드 정상 동작 여부, 작은 PPE 성능 개선 경향, 큰 객체 성능 하락 여부, latency
증가량 등). 이 데이터셋은 향후 폐기/교체될 수 있으므로 1차 결과는 방법 선택을
위한 파일럿으로만 쓴다.

| 비교군 | 설명 | weight |
|---|---|---|
| **A. Standard 640** | 기존 YOLOv9t 640으로 전체 이미지 1회 추론 (baseline) | 기존 640 weight |
| **B. Standard 960** | `imgsz=960`으로 별도 학습한 모델로 전체 이미지 1회 추론 | 별도 960 weight (팀원이 학습) |
| **C. SAHI 640** | **A와 동일한 640 weight**를 그대로 쓰고, 추론만 overlapping slice로 수행 | 기존 640 weight (추가 학습 없음) |
| **D. Full+SAHI 640** | 같은 이미지에 A(full-image)와 C(slice)를 모두 수행해 결과를 합침 | 기존 640 weight (추가 학습 없음) |

비교 관점: `A vs B`(해상도 증가 자체의 효과), `A vs C`(SAHI slicing 자체의
효과), `C vs D`(full-image 추가가 큰 객체 유지에 도움이 되는지), `B vs D`(960
단일 추론과 Hybrid SAHI의 정확도/latency trade-off).

## 3. 2차 정식 실험

팀에서 새로 구축하는 **v2 데이터셋**(기존 bbox 재작업 + 작은 PPE crop 추가)이
완성된 뒤 수행한다. v2로 YOLOv9t 640을 **한 번만 새로 학습**하고, 그 동일 weight를
E/F/G 세 비교군이 공유한다.

| 비교군 | 설명 | weight |
|---|---|---|
| **E. v2 Standard 640** | v2로 학습한 YOLOv9t 640 + v2 test set + full-image 추론 (2차 baseline) | v2 640 weight |
| **F. v2 SAHI 640** | **E와 동일한 v2 640 weight** + v2 test set + sliced 추론 | v2 640 weight (추가 학습 없음) |
| **G. v2 Full+SAHI 640** | E와 동일한 weight + v2 test set + full-image와 slice 결과 통합 | v2 640 weight (추가 학습 없음) |

**중요**: E/F/G는 반드시 **동일한 v2 test set**으로 생성한 결과만 비교한다.
old dataset과 v2 dataset 사이의 성능 비교(old model vs new model, old test vs
new test)는 이 실험 범위에서 다루지 않는다 — 이 실험이 보고 싶은 것은 오직
"v2로 학습된 동일 모델에서 추론 방식 차이가 얼마나 효과를 만드는가"이다.

## 4. Data leakage 주의사항 (v2 데이터셋 전용)

v2 데이터셋에는 원본 이미지 외에 작은 PPE crop 이미지가 추가될 예정이다. **원본
이미지와 그 이미지에서 파생된 모든 crop은 반드시 동일 split(train/val/test
중 하나)에 배치**해야 한다.

```text
잘못된 예:
  A.jpg            -> train
  A_helmet_crop.jpg -> test   # A.jpg와 거의 같은 장면이 test에 노출됨 (leakage)
```

가능하면 같은 영상에서 연속으로 추출된 유사 frame들도 동일 split으로 묶는 것이
좋다. 이 규칙은 권장사항이며, split 생성 자체는 이 코드에서 자동화하지 않는다.

## 5. 필요한 모델/데이터

- `--model`: 평가할 `.onnx` 모델 경로. A/C/D는 기존 640 weight, B는 별도 960
  weight, E/F/G는 v2로 새로 학습한 640 weight.
- `--images`: YOLO 데이터셋의 이미지 디렉터리 (jpg/jpeg/png/bmp).
- `--labels` (evaluate.py에서만 필요): 같은 파일명(확장자만 `.txt`)의 YOLO
  format label 디렉터리. `class_id cx cy w h` (0~1 정규화).

이 코드는 특정 절대경로/데이터셋 이름에 의존하지 않는다 — 모든 경로는 실행 시
CLI로 받는다. 1차의 기존 데이터셋과 2차의 v2 데이터셋을 동일한 코드로 처리할 수
있다.

## 6. Dependency 설치

이 실험은 프로젝트 `requirements.txt`에 이미 있는 `numpy`, `opencv-python`,
`PyYAML`, `onnxruntime`만 사용한다 — **추가 설치가 필요 없다**. mAP 계산은
`pycocotools` 같은 외부 라이브러리 대신 `experiments/sahi/metrics.py`에서
numpy로 직접 구현했다(이유: 로컬에 COCO JSON 변환 파이프라인이 없고, YOLO
label을 바로 쓸 수 있게 하기 위함). 자세한 내용은 7절 참고.

## 7. 실행 명령

### 1차 실험

```bash
# A. Standard 640 (baseline)
python experiments/sahi/run_standard.py \
  --model /path/to/yolov9t_640.onnx \
  --images /path/to/test/images \
  --input-size 640 \
  --output experiments/sahi/outputs/A_standard_640

# B. Standard 960 (960으로 별도 학습한 모델이 있을 때만)
python experiments/sahi/run_standard.py \
  --model /path/to/yolov9t_960.onnx \
  --images /path/to/test/images \
  --input-size 960 \
  --output experiments/sahi/outputs/B_standard_960

# C. SAHI 640 (A와 동일한 640 weight, 추가 학습 없음)
python experiments/sahi/run_sahi.py \
  --model /path/to/yolov9t_640.onnx \
  --images /path/to/test/images \
  --slice-size 640 \
  --overlap 0.2 \
  --output experiments/sahi/outputs/C_sahi_640

# D. Full + SAHI 640 (A와 동일한 640 weight, 추가 학습 없음)
python experiments/sahi/run_hybrid.py \
  --model /path/to/yolov9t_640.onnx \
  --images /path/to/test/images \
  --slice-size 640 \
  --overlap 0.2 \
  --output experiments/sahi/outputs/D_hybrid_640

# 평가 (A/B/C/D 각각에 대해 동일하게 실행)
python experiments/sahi/evaluate.py \
  --predictions experiments/sahi/outputs/C_sahi_640/predictions.json \
  --labels /path/to/test/labels \
  --images /path/to/test/images

# 비교표 생성 (B가 없으면 --result B=... 줄만 빼면 된다)
python experiments/sahi/compare.py \
  --result A=experiments/sahi/outputs/A_standard_640 \
  --result B=experiments/sahi/outputs/B_standard_960 \
  --result C=experiments/sahi/outputs/C_sahi_640 \
  --result D=experiments/sahi/outputs/D_hybrid_640 \
  --output experiments/sahi/outputs/comparison_pilot
```

### 2차 실험 (v2 데이터셋 확보 후)

아래 명령은 1차와 **완전히 동일한 코드**이고, 모델 경로와 이미지/레이블 경로만
v2 데이터로 바꾼 것이다.

```bash
# E. v2 Standard 640
python experiments/sahi/run_standard.py \
  --model /path/to/yolov9t_v2_640.onnx \
  --images /path/to/v2_test/images \
  --input-size 640 \
  --output experiments/sahi/outputs/E_v2_standard_640

# F. v2 SAHI 640 (E와 동일한 v2 640 weight)
python experiments/sahi/run_sahi.py \
  --model /path/to/yolov9t_v2_640.onnx \
  --images /path/to/v2_test/images \
  --slice-size 640 \
  --overlap 0.2 \
  --output experiments/sahi/outputs/F_v2_sahi_640

# G. v2 Full + SAHI 640 (E와 동일한 v2 640 weight)
python experiments/sahi/run_hybrid.py \
  --model /path/to/yolov9t_v2_640.onnx \
  --images /path/to/v2_test/images \
  --slice-size 640 \
  --overlap 0.2 \
  --output experiments/sahi/outputs/G_v2_hybrid_640

python experiments/sahi/evaluate.py \
  --predictions experiments/sahi/outputs/F_v2_sahi_640/predictions.json \
  --labels /path/to/v2_test/labels \
  --images /path/to/v2_test/images

# E/F/G는 반드시 동일한 v2 test set 결과만 비교한다.
python experiments/sahi/compare.py \
  --result E=experiments/sahi/outputs/E_v2_standard_640 \
  --result F=experiments/sahi/outputs/F_v2_sahi_640 \
  --result G=experiments/sahi/outputs/G_v2_hybrid_640 \
  --output experiments/sahi/outputs/comparison_v2
```

빠른 동작 확인만 하고 싶다면 `--limit 20` 같은 옵션으로 이미지 수를 제한할 수
있고, `--save-visualizations`를 추가하면 `<output>/visualizations/`에 bbox를
그린 이미지가 저장된다(기본값은 저장 안 함 — 대량 생성 방지).

## 8. 결과 파일 구조

```text
experiments/sahi/outputs/<실험이름>/
├── predictions.json        # 이미지별 detection + latency_ms
├── summary.json            # 이미지 수, latency 통계(mean/p50/p95), 실행 설정
├── metrics.json            # evaluate.py 실행 후 생성: class별/전체 AP
└── visualizations/         # --save-visualizations를 줬을 때만 생성
    └── <image>.jpg

experiments/sahi/outputs/<비교이름>.csv
experiments/sahi/outputs/<비교이름>.md
```

`predictions.json`의 이미지별 스키마:

```json
{
  "image": "example.jpg",
  "width": 1920,
  "height": 1080,
  "detections": [
    {"class_id": 3, "class_name": "helmet", "confidence": 0.81, "bbox_xyxy": [100, 200, 150, 260]}
  ],
  "latency_ms": 25.3
}
```

## 9. 주요 설정값

- `slice.width` / `slice.height` / `slice.overlap_width_ratio` /
  `slice.overlap_height_ratio` (`config.yaml`, 기본 640/640/0.2/0.2): CLI
  `--slice-size`/`--slice-width`/`--slice-height`/`--overlap`/`--overlap-width`/
  `--overlap-height`가 주어지면 그 값이 항상 config.yaml보다 우선한다.
- `--confidence-threshold`(기본 0.25), `--iou-threshold`(기본 0.45):
  `config/model.yaml`의 기존 기본값과 동일하게 맞췄다.
- `--input-size`: standard(run_standard.py)는 전체 이미지를 이 크기로
  letterbox resize, SAHI/Hybrid(run_sahi.py/run_hybrid.py)는 **각 slice를**
  이 크기로 letterbox resize한다(원본 전체가 아니라 slice 단위로 적용됨에 유의).

## 10. mAP/AP_small 계산 방식과 한계

`evaluate.py`는 COCO 스타일 101-point interpolation으로 AP를 직접 계산한다
(`experiments/sahi/metrics.py`). 면적 기준(AP_small/medium/large)은 COCO 공식
정의를 그대로 썼다: bbox 면적이 `32x32=1024px²` 미만이면 small, `1024~9216px²`
(=`96x96`)이면 medium, 그 이상이면 large다. GT를 그 버킷으로만 필터링해 AP를
계산하는 방식이라, **pycocotools의 ignore 플래그/crowd 처리 같은 세부 구현을
그대로 재현하지는 않는다** — 작은 PPE 성능 경향을 보는 1차 파일럿 목적에는
충분하지만, 논문/정식 보고용 수치가 필요하면 pycocotools 기반 재검증을 권장한다.

Precision/Recall은 confidence를 스윕하는 AP 곡선과 달리, **이미 설정된
`--confidence-threshold`에서 걸러진 predictions.json 전체**를 그대로 써서 계산한
"그 운영 지점(operating point)"의 값이다.

데이터셋/모델이 없는 현재 로컬 환경에서는 실제 수치를 만들어낼 수 없으므로,
이 README와 코드에는 임의의 mock 성능 숫자를 넣지 않았다.

## 11. 테스트

```bash
.venv/bin/python -m pytest tests/test_sahi_slicing.py tests/test_sahi_utils.py tests/test_sahi_metrics.py -v
```

실제 모델 weight나 이미지 데이터셋 없이도 다음을 검증한다:

- slice가 이미지 전체를 빠짐없이 덮는지, 경계가 누락되지 않는지, overlap이
  적용되는지, 이미지가 slice보다 작아도 동작하는지 (`test_sahi_slicing.py`)
- slice 좌표 -> 원본 좌표 복원이 정확한지, 경계 밖 bbox가 올바르게 clip되는지
  (`test_sahi_slicing.py`)
- overlap 영역에서 중복 탐지된 같은 class bbox가 global NMS로 하나로 정리되는지
  (`test_sahi_slicing.py`)
- latency 통계/이미지 목록 나열 등 공통 유틸리티 (`test_sahi_utils.py`)
- AP/Precision/Recall 계산이 synthetic GT/prediction에 대해 올바른 값을 내는지
  (`test_sahi_metrics.py`)

`run_standard.py`/`run_sahi.py`/`run_hybrid.py`는 실제 `.onnx` weight가 있어야
동작하므로, 이 환경에서는 import/문법 검사(`python -m compileall`,
`pyflakes`)만 수행했다. 실제 추론 동작은 weight와 데이터셋을 가진 팀원이
위 "실행 명령"을 그대로 실행해서 확인해야 한다.
