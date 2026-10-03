"""ONNX Runtime execution provider 선택 정책을 검증한다."""

import pytest

from app.inference.onnx_backend import (
    _providers_for_device,
    load_onnx_detector_config,
)
from app.video.video_source import PROJECT_ROOT


def test_cpu_device_uses_cpu_provider() -> None:
    providers = _providers_for_device(
        device="cpu",
        available_providers=["CPUExecutionProvider"],
    )

    assert providers == ["CPUExecutionProvider"]


def test_cuda_device_uses_selected_gpu_then_cpu_fallback() -> None:
    providers = _providers_for_device(
        device="cuda:1",
        available_providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    assert providers == [
        ("CUDAExecutionProvider", {"device_id": "1"}),
        "CPUExecutionProvider",
    ]


def test_coreml_device_enables_all_compute_units() -> None:
    providers = _providers_for_device(
        device="coreml",
        available_providers=[
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    assert providers == [
        (
            "CoreMLExecutionProvider",
            {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "1",
            },
        ),
        "CPUExecutionProvider",
    ]


@pytest.mark.parametrize("device", ["cuda", "coreml"])
def test_accelerator_can_fall_back_to_cpu(device: str) -> None:
    providers = _providers_for_device(
        device=device,
        available_providers=["CPUExecutionProvider"],
        allow_cpu_fallback=True,
    )

    assert providers == ["CPUExecutionProvider"]


@pytest.mark.parametrize("device", ["cuda", "coreml"])
def test_accelerator_can_require_strict_provider(device: str) -> None:
    with pytest.raises(RuntimeError, match="is unavailable"):
        _providers_for_device(
            device=device,
            available_providers=["CPUExecutionProvider"],
            allow_cpu_fallback=False,
        )


def test_invalid_cuda_device_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid CUDA device"):
        _providers_for_device(
            device="cuda:primary",
            available_providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )


def test_branch_model_config_prefers_first_cuda_gpu_with_cpu_fallback() -> None:
    config = load_onnx_detector_config(PROJECT_ROOT / "config" / "model.yaml")

    assert config.device == "cuda:0"
    assert config.allow_cpu_fallback is True
