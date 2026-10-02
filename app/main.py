"""Vision Guard 실행 진입점."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import cv2
import numpy as np

from app.alerts.alert_renderer import draw_alert_overlay
from app.alerts.event_manager import EventManager
from app.alerts.models import CameraAlertUpdate
from app.alerts.sound import LocalSoundPlayer
from app.inference.detector import Detector
from app.inference.model_manager import create_detector_from_model_config
from app.risk.risk_engine import RiskEngine
from app.risk.risk_logger import RiskLogger
from app.tracking.tracker import Tracker, create_tracker_from_system_config
from app.tracking.trajectory import TrajectoryAnalyzer
from app.video.frame_processor import FrameProcessor
from app.zones.zone_manager import ZoneManager
from app.video.sample_dataset import create_random_ceiling_eye_video_sources
from app.video.video_source import (
    CameraConfig,
    PROJECT_ROOT,
    VideoSource,
    VideoSourceError,
    create_video_source_from_config,
    load_camera_configs,
)

WINDOW_NAME = "Vision Guard"
TILE_WIDTH = 640
TILE_HEIGHT = 360
MAX_VIEW_COUNT = 4
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data" / "samples" / "forklift_human_nearmiss"
# cam_03(ceiling)/cam_04(eye)에 쓸 샘플 영상을 고정하고 싶을 때만 정수로 지정한다.
# None이면 실행마다 무작위로 run/ceiling/eye가 선택된다. 값을 바꿔가며 실행해 터미널의
# "[sample video] cam_03: ..." 출력으로 원하는 파일이 나오는 seed를 찾으면 된다.
# 현재 테스트 가능한 seed 번호: 4, 7, 
SAMPLE_VIDEO_SEED: int | None = 14
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"
SYSTEM_CONFIG_PATH = PROJECT_ROOT / "config" / "system.yaml"

# 화면 슬롯 0~3에 대응하는 고정 camera_id 목록 ("cam_01".."cam_04").
# 실제 video_sources 개수가 4보다 적어도 Risk Log 파일은 이 네 개로 항상 유지한다.
DISPLAY_CAMERA_IDS = [f"cam_{slot_index + 1:02d}" for slot_index in range(MAX_VIEW_COUNT)]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessingComponents:
    detector: Detector | None
    trackers: dict[str, Tracker]
    trajectory_analyzers: dict[str, TrajectoryAnalyzer]
    zone_manager: ZoneManager
    risk_engine: RiskEngine


@dataclass(slots=True)
class CameraViewState:
    video_source: VideoSource
    processor: FrameProcessor | None = None
    enabled: bool = False


@dataclass(frozen=True, slots=True)
class ButtonBounds:
    camera_id: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        """마우스 좌표가 버튼 영역 안에 있는지 확인한다."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(slots=True)
class DisplayState:
    camera_views: list[CameraViewState]
    button_bounds: list[ButtonBounds]


@dataclass(slots=True)
class DisplayFrameResult:
    """_read_display_frame() 한 번의 결과: 화면에 쓸 frame과 Sound 판단용 Alert 결과.

    frame은 camera OFF/분석 불가 상태에서도 항상 채워진다(blank tile 또는 raw frame).
    alert_update는 이번 cycle에 Risk 분석이 실제로 성공해 EventManager가 그 camera의
    Alert 상태를 갱신한 경우에만 채워지고, 그 외에는 None이다. main loop는 이 None 여부로
    "Sound 판단에 포함할 camera"를 걸러낸다.
    """

    frame: np.ndarray
    alert_update: CameraAlertUpdate | None


def main() -> None:
    """설정된 영상 입력을 최대 4개까지 한 창의 2x2 화면으로 표시한다."""
    camera_configs = load_camera_configs()
    video_sources = _create_display_video_sources(camera_configs)
    _assign_display_camera_ids(video_sources)

    # cam_03/04는 실행마다 샘플 ceiling/eye 영상이 무작위로 바뀌어
    # 영상 확인을 위해 디버깅용으로 실행 시 한 번 터미널에 출력한다.
    for video_source in video_sources[2:4]:
        print(f"[sample video] {video_source.camera_id}: {video_source.source}")

    processing_components = _create_processing_components(video_sources)

    # Risk Log: 화면 슬롯 4개에 대응하는 로그 파일을 시작 시 한 번만 초기화한다.
    # 프레임 처리 중에는 이 인스턴스 하나의 log()만 호출되고 다시 초기화하지 않는다.
    risk_logger = RiskLogger()
    _prepare_risk_logger(risk_logger)

    # Alert UI: EventManager는 camera_id별 hold 상태를 들고 있는 stateful 객체라
    # 프레임마다 새로 만들면 3초 hold가 매번 초기화돼버린다. 실행 중 단 하나만
    # 만들어 계속 재사용한다.
    event_manager = EventManager()

    # Sound Alert: LocalSoundPlayer는 현재 재생 중인 process/RiskLevel을 들고 있는
    # stateful 객체라 프레임마다 새로 만들면 재생 중인 소리를 추적할 수 없다.
    # EventManager와 마찬가지로 실행 중 단 하나만 만들어 계속 재사용한다.
    sound_player = LocalSoundPlayer()

    display_state = DisplayState(
        camera_views=[
            CameraViewState(
                video_source=video_source,
                processor=_create_frame_processor(
                    video_source=video_source,
                    processing_components=processing_components,
                ),
            )
            for video_source in video_sources
        ],
        button_bounds=[],
    )

    try:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, _handle_mouse_event, display_state)

        while True:
            # 한 번의 2x2 화면 갱신에서는 모든 camera가 같은 now를 기준으로
            # blink를 계산하도록 루프당 한 번만 시각을 읽는다.
            now = time.monotonic()
            results = [
                _read_display_frame(camera_view, risk_logger, event_manager, now)
                for camera_view in display_state.camera_views
            ]
            frames = [result.frame for result in results]

            # Sound Alert: 이번 cycle에서 분석이 성공한 camera들의 CameraAlertUpdate만
            # 모아(개별 camera 처리 즉시가 아니라) 한 번만 Sound 판단을 수행한다.
            alert_updates = [
                result.alert_update
                for result in results
                if result.alert_update is not None
            ]
            _dispatch_sound_alert(alert_updates, sound_player, now)

            display_frame, button_bounds = _compose_2x2_grid(
                frames=frames,
                camera_views=display_state.camera_views,
            )
            display_state.button_bounds = button_bounds

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        sound_player.close()
        for camera_view in display_state.camera_views:
            camera_view.video_source.close()
        cv2.destroyAllWindows()


def _handle_mouse_event(
    event: int,
    x: int,
    y: int,
    flags: int,
    display_state: DisplayState | None,
) -> None:
    """OpenCV 창의 버튼 클릭을 카메라 enable 상태로 반영한다."""
    del flags
    if event != cv2.EVENT_LBUTTONDOWN or display_state is None:
        return

    for button_bound in display_state.button_bounds:
        if button_bound.contains(x, y):
            _toggle_camera_view(display_state.camera_views, button_bound.camera_id)
            return


def _toggle_camera_view(
    camera_views: list[CameraViewState],
    camera_id: str,
) -> None:
    """camera_id에 해당하는 입력을 켜거나 끈다."""
    for camera_view in camera_views:
        if camera_view.video_source.camera_id != camera_id:
            continue

        camera_view.enabled = not camera_view.enabled
        if not camera_view.enabled:
            camera_view.video_source.close()
        return


def _create_processing_components(
    video_sources: list[VideoSource],
) -> ProcessingComponents:
    """입력별 tracking 상태와 trajectory analyzer, 공용 Zone/Risk 컴포넌트를 생성한다."""
    detector = _create_detector()
    trackers = {
        video_source.camera_id: _create_tracker()
        for video_source in video_sources
    }
    trajectory_analyzers = {
        video_source.camera_id: TrajectoryAnalyzer()
        for video_source in video_sources
    }
    return ProcessingComponents(
        detector=detector,
        trackers=trackers,
        trajectory_analyzers=trajectory_analyzers,
        # ZoneManager와 RiskEngine은 프레임 단위 순수 계산만 수행하고 내부 상태를
        # 유지하지 않으므로, tracker/trajectory_analyzer와 달리 카메라별로 나누지 않고
        # 모든 FrameProcessor가 공유하는 인스턴스 하나만 둔다.
        zone_manager=ZoneManager(),
        risk_engine=RiskEngine(),
    )


def _create_detector() -> Detector | None:
    """model.yaml 기반 detector를 생성하고 실패 시 화면 출력만 계속 가능하게 한다."""
    try:
        return create_detector_from_model_config(MODEL_CONFIG_PATH)
    except Exception as exc:
        logger.warning("Detector is disabled: %s", exc)
        return None


def _create_tracker() -> Tracker:
    """system.yaml 기반 tracker를 생성한다."""
    return create_tracker_from_system_config(SYSTEM_CONFIG_PATH)


def _create_frame_processor(
    video_source: VideoSource,
    processing_components: ProcessingComponents,
) -> FrameProcessor | None:
    """detector가 준비된 입력에 FrameProcessor를 연결한다."""
    if processing_components.detector is None:
        return None

    tracker = processing_components.trackers.get(video_source.camera_id)
    trajectory_analyzer = processing_components.trajectory_analyzers.get(
        video_source.camera_id
    )
    if tracker is None or trajectory_analyzer is None:
        return None

    return FrameProcessor(
        video_source=video_source,
        detector=processing_components.detector,
        tracker=tracker,
        trajectory_analyzer=trajectory_analyzer,
        zone_manager=processing_components.zone_manager,
        risk_engine=processing_components.risk_engine,
        draw_bbox=True,
        draw_metrics=True,
    )


def _create_display_video_sources(
    camera_configs: list[CameraConfig],
) -> list[VideoSource]:
    """샘플 ceiling/eye 영상과 설정 기반 입력을 4분할 화면 입력으로 구성한다."""
    sample_sources = _create_sample_video_sources()
    camera_source_count = MAX_VIEW_COUNT - len(sample_sources)
    camera_sources = _create_video_sources(camera_configs[:camera_source_count])
    return [*camera_sources, *sample_sources][:MAX_VIEW_COUNT]


def _prepare_risk_logger(risk_logger: RiskLogger) -> None:
    """화면 슬롯 4개(DISPLAY_CAMERA_IDS)에 대응하는 로그 파일을 항상 준비한다.

    실제 video_sources 개수와 무관하게 risk_log_cam_01~04.jsonl 네 개를 고정으로
    초기화한다. 이번 실행에서 입력이 없는 슬롯의 로그 파일은 빈 파일로 유지된다.
    """
    risk_logger.prepare(DISPLAY_CAMERA_IDS)


def _assign_display_camera_ids(video_sources: list[VideoSource]) -> None:
    """화면 슬롯 순서를 기준으로 camera_id를 cam_01~04로 고정한다.

    샘플 ceiling/eye 영상은 실행마다 무작위로 다른 run_id가 선택되어
    (app/video/sample_dataset.py) camera_id도 매번 달라진다. 이 상태로 두면
    RiskLogger가 실행마다 새 로그 파일을 계속 만들게 된다. 실제 영상 내용과
    무관하게 화면에 표시되는 슬롯 순서(0번=laptop-camera, 1번=factory-floor-demo,
    2~3번=샘플 ceiling/eye)를 기준으로 camera_id를 덮어써서, 로그 파일이 항상
    cam_01~04 네 개로만 유지되게 한다.
    """
    for slot_index, video_source in enumerate(video_sources):
        video_source.camera_id = f"cam_{slot_index + 1:02d}"


def _create_sample_video_sources() -> list[VideoSource]:
    """샘플 metadata에서 ceiling/eye 영상 쌍을 생성한다."""
    if not DEFAULT_SAMPLE_DIR.is_dir():
        return []

    try:
        return create_random_ceiling_eye_video_sources(
            sample_dir=DEFAULT_SAMPLE_DIR,
            loop=True,
            seed=SAMPLE_VIDEO_SEED,
        )
    except (FileNotFoundError, ValueError):
        return []


def _create_video_sources(camera_configs: list[CameraConfig]) -> list[VideoSource]:
    """지원되는 카메라 설정만 VideoSource로 변환한다."""
    video_sources: list[VideoSource] = []

    for camera_config in camera_configs:
        try:
            video_sources.append(create_video_source_from_config(camera_config))
        except ValueError:
            # RTSP처럼 아직 별도 source 구현이 필요한 입력은 현재 4분할 화면에서 제외한다.
            continue

    return video_sources


def _read_display_frame(
    camera_view: CameraViewState,
    risk_logger: RiskLogger,
    event_manager: EventManager,
    now: float,
) -> DisplayFrameResult:
    """단일 입력에서 프레임을 읽고 4분할 타일 크기로 변환한다.

    alert_update는 이번 호출에서 Risk 분석이 실제로 성공해 EventManager가 그
    camera의 Alert 상태를 갱신한 경우에만 채워진다(Sound 판단에서 이 camera를
    포함시키기 위함). camera OFF, 입력 실패, processor 없음 등 분석이 아예
    이뤄지지 않은 경우에는 항상 None이다.
    """
    video_source = camera_view.video_source
    if not camera_view.enabled:
        return DisplayFrameResult(
            frame=_make_blank_tile(f"{video_source.camera_id}: off"),
            alert_update=None,
        )

    if camera_view.processor is not None:
        try:
            processed_frame = camera_view.processor.process_next()
        except (FileNotFoundError, VideoSourceError):
            return DisplayFrameResult(
                frame=_make_blank_tile(f"{video_source.camera_id}: no input"),
                alert_update=None,
            )

        if processed_frame is None:
            return DisplayFrameResult(
                frame=_make_blank_tile(f"{video_source.camera_id}: no frame"),
                alert_update=None,
            )

        # Risk Log: Zone/Risk 계산 결과를 프레임마다 JSONL로 남긴다(디버깅/검증용).
        # FrameProcessor 내부에서는 파일 I/O를 하지 않고, 호출부인 여기서 기록한다.
        risk_logger.log(
            zone_result=processed_frame.zone_result,
            risk_assessments=processed_frame.risk_assessments,
        )

        frame = processed_frame.rendered_frame
        if frame is None:
            frame = processed_frame.frame_packet.frame

        frame = _resize_tile(frame)
        frame = _draw_camera_label(frame=frame, label=video_source.camera_id)

        # Alert UI: 분석이 실제로 성공한 경우(이 분기)에만 camera의 Alert 상태를
        # 갱신하고 renderer에게 그리기를 맡긴다. camera OFF나 처리 실패 시에는 이
        # 분기에 들어오지 않으므로 Alert UI도 자연히 표시되지 않는다.
        camera_alert_update = event_manager.update(
            camera_id=video_source.camera_id,
            risk_assessments=processed_frame.risk_assessments,
        )
        draw_alert_overlay(frame=frame, camera_alert_update=camera_alert_update, now=now)

        return DisplayFrameResult(frame=frame, alert_update=camera_alert_update)

    # processor가 없는 입력(detector 비활성 등)은 Risk 분석 결과 자체가 없으므로
    # Alert badge/border를 그리지 않는다. NORMAL로 대체 표시하지 않는 것이 중요하다
    # (NORMAL은 "분석 결과가 정상"이라는 뜻이고, 여기는 분석이 아예 안 되는 상태다).
    try:
        frame_packet = video_source.read()
    except (FileNotFoundError, VideoSourceError):
        return DisplayFrameResult(
            frame=_make_blank_tile(f"{video_source.camera_id}: no input"),
            alert_update=None,
        )

    if frame_packet is None:
        return DisplayFrameResult(
            frame=_make_blank_tile(f"{video_source.camera_id}: no frame"),
            alert_update=None,
        )

    frame = _resize_tile(frame_packet.frame)
    frame = _draw_camera_label(frame=frame, label=video_source.camera_id)
    return DisplayFrameResult(frame=frame, alert_update=None)


def _dispatch_sound_alert(
    alert_updates: list[CameraAlertUpdate],
    sound_player: LocalSoundPlayer,
    now: float,
) -> None:
    """이번 display cycle에서 모은 CameraAlertUpdate 전체로 Sound 상태를 한 번 갱신한다.

    "최초 진입/escalation만 즉시 반응", "위험 유지 시 3초마다 반복 알림" 같은 실제
    판단은 LocalSoundPlayer.update()가 전담한다. 이 함수는 orchestration 계층에서
    그 호출을 한 번만 하도록 묶어주는 역할만 한다.
    """
    sound_player.update(alert_updates, now)


def _compose_2x2_grid(
    frames: list[np.ndarray],
    camera_views: list[CameraViewState] | None = None,
) -> tuple[np.ndarray, list[ButtonBounds]]:
    """최대 4개의 프레임을 2x2 격자 이미지로 합친다."""
    tiles = [_resize_tile(frame) for frame in frames[:MAX_VIEW_COUNT]]
    visible_camera_views = list((camera_views or [])[:MAX_VIEW_COUNT])

    while len(tiles) < MAX_VIEW_COUNT:
        tiles.append(_make_blank_tile("empty"))
    while len(visible_camera_views) < MAX_VIEW_COUNT:
        visible_camera_views.append(
            CameraViewState(video_source=VideoSource(source=0, camera_id="empty"))
        )

    button_bounds: list[ButtonBounds] = []
    for tile_index, camera_view in enumerate(visible_camera_views[:MAX_VIEW_COUNT]):
        if camera_view.video_source.camera_id == "empty":
            continue
        offset_x = (tile_index % 2) * TILE_WIDTH
        offset_y = (tile_index // 2) * TILE_HEIGHT
        button_bounds.append(
            _draw_power_button(
                frame=tiles[tile_index],
                camera_id=camera_view.video_source.camera_id,
                enabled=camera_view.enabled,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        )

    top_row = np.hstack([tiles[0], tiles[1]])
    bottom_row = np.hstack([tiles[2], tiles[3]])
    return np.vstack([top_row, bottom_row]), button_bounds


def _resize_tile(frame: np.ndarray) -> np.ndarray:
    """입력 프레임을 고정 크기 타일로 맞춘다."""
    return cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT))


def _make_blank_tile(label: str) -> np.ndarray:
    """프레임이 없는 입력을 표시하기 위한 빈 타일을 만든다."""
    frame = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    return _draw_camera_label(frame=frame, label=label)


def _draw_camera_label(frame: np.ndarray, label: str) -> np.ndarray:
    """타일 왼쪽 위에 카메라 식별자를 표시한다."""
    # 임시 비활성화: 화면 확인 시 카메라 이름이 detection 라벨과 겹치지 않도록 표시하지 않는다.
    # cv2.putText(
    #     frame,
    #     label,
    #     (20, 35),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.8,
    #     (0, 255, 255),
    #     2,
    # )
    return frame


def _draw_power_button(
    frame: np.ndarray,
    camera_id: str,
    enabled: bool,
    offset_x: int,
    offset_y: int,
) -> ButtonBounds:
    """타일 오른쪽 위에 카메라 ON/OFF 버튼을 그린다."""
    x1 = TILE_WIDTH - 104
    y1 = 16
    x2 = TILE_WIDTH - 18
    y2 = 52
    color = (40, 160, 40) if enabled else (70, 70, 70)
    label = "ON" if enabled else "OFF"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(
        frame,
        label,
        (x1 + 18, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return ButtonBounds(
        camera_id=camera_id,
        x1=offset_x + x1,
        y1=offset_y + y1,
        x2=offset_x + x2,
        y2=offset_y + y2,
    )


if __name__ == "__main__":
    main()
