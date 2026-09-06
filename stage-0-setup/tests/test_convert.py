"""커밋된 mock 픽스처 전부를 실제로 변환해 본다.

여기서 지키려는 것은 "돌아간다"가 아니라 **두 세대를 옳게 갈라 읽는가**다.
proto3는 틀린 스키마로 읽어도 크래시하지 않으므로, 회전 기저가 정규직교라는 것만이
세대를 옳게 골랐다는 증거다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from movin_clip import convert, decode, motion, store

known_kinds = {
    "DH/ObjsFrame.packets.bin": "objs_frame/motion9",
    "v3/ObjsFrame.packets.bin": "objs_frame/motion6",
    "v3.bin": "objs_frame/motion6",
    "objsframe_subscriber.bin": "objs_frame/motion6",
    "t_pose_5frames.packets.bin": "objs_frame/motion6",
    "arms_forward_palms_in_5frames.packets.bin": "objs_frame/motion6",
    "mocap_idle.bin": "mocap_topic",
    "mocap_idle_20fps.bin": "mocap_topic",
    "mocap_entering.bin": "mocap_topic",
    "synced_mocap_subscriber.bin": "mocap_topic",
}


@pytest.mark.parametrize("name", sorted(known_kinds), ids=lambda n: n)
def test_detects_the_right_generation(mock_root: Path, name: str):
    assert store.detect_kind(store.read_payloads(mock_root / name)).label() == known_kinds[name]


def test_wrong_generation_is_detectable(mock_root: Path):
    """옛 스토어를 새 스키마로 읽으면 조용히 모션이 사라진다 — 그래서 길이로 판정한다."""
    payloads = store.read_payloads(mock_root / "t_pose_5frames.packets.bin")
    current = decode.schema.objs_frame_class(decode.schema.variant_by_name("motion9"))
    frame = current()
    frame.ParseFromString(payloads[0])
    assert frame.objs and all(entry.has_motion for entry in frame.objs)
    assert all(len(entry.motion) == 0 for entry in frame.objs)  # 크래시가 아니라 침묵이다


@pytest.mark.parametrize("name", sorted(known_kinds), ids=lambda n: n)
def test_converts_and_rotations_are_orthonormal(mock_root: Path, tmp_path: Path, name: str):
    meta = convert.convert(mock_root / name, tmp_path / "clip")
    assert meta["rows"] > 0

    check = meta["rotation_check"]
    if meta["rows_with_valid_motion"] > 0:
        assert max(check.values()) < 1e-3
    else:
        assert check is None  # mocap_entering: zone state가 PLAYING이 아니라 전부 0

    frames = pl.read_parquet(tmp_path / "clip/frames.parquet")
    assert len(frames) == meta["rows"]
    assert json.loads((tmp_path / "clip/meta.json").read_text())["source"]["sha256"] == meta["source"]["sha256"]


def test_motion_valid_marks_the_zero_filled_frames(mock_root: Path, tmp_path: Path):
    meta = convert.convert(mock_root / "mocap_entering.bin", tmp_path / "clip")
    frames = pl.read_parquet(tmp_path / "clip/frames.parquet")
    assert meta["rows_with_valid_motion"] == 0
    assert not frames["motion_valid"].any()
    assert frames["zone_state"].unique().to_list() == [1]  # ENTERED
    assert np.abs(np.stack(frames["rotation_6d"].to_numpy())).max() == 0


def test_pointclouds_are_one_file_per_sensor_frame(mock_root: Path, tmp_path: Path):
    """센서 20 Hz, 모션 60 Hz. 서브프레임 셋이 같은 클라우드를 공유한다."""
    meta = convert.convert(mock_root / "DH/ObjsFrame.packets.bin", tmp_path / "clip")
    frames = pl.read_parquet(tmp_path / "clip/frames.parquet")

    assert frames["total_sub_frames"].unique().to_list() == [3]
    assert meta["pointcloud"]["files"] == frames["timestamp_sensor_ns"].n_unique()

    # 서브프레임 셋이 파일 하나를 공유한다. 녹화가 서브프레임 중간에서 시작·종료하므로
    # 양 끝 센서 프레임만 셋이 안 될 수 있다(이 클립은 첫 행이 sub_frame_index=2다).
    per_file = frames.group_by("pointcloud_file").len()["len"].to_list()
    assert set(per_file) <= {1, 2, 3}
    assert sum(count == 3 for count in per_file) >= len(per_file) - 2
    assert sum(per_file) == meta["pointcloud"]["rows_referencing"] == len(frames)

    written = sorted(path.name for path in (tmp_path / "clip/pointcloud").iterdir())
    assert written == sorted(set(frames["pointcloud_file"].to_list()))

    # 스트림이 세는 점 개수와 실제 payload가 맞는지
    assert (frames["num_points"] == frames["pointcloud_points"]).all()
    sample = np.load(tmp_path / "clip/pointcloud" / written[0])
    assert sample.ndim == 2 and sample.shape[1] == 4 and sample.dtype == np.float32


def test_reference_columns_have_the_documented_widths(mock_root: Path, tmp_path: Path):
    convert.convert(mock_root / "DH/ObjsFrame.packets.bin", tmp_path / "clip")
    frames = pl.read_parquet(tmp_path / "clip/frames.parquet")
    assert {len(row) for row in frames["body_keypoints_2d"].to_list()} == {51}  # COCO-17 (x, y, conf)
    assert {len(row) for row in frames["body_bbox"].to_list()} == {4}
    assert {len(row) for row in frames["hand_left_keypoints_2d"].to_list()} <= {0, 42}  # 21 × (x, y)


def test_contact_is_continuous_not_binary(mock_root: Path, tmp_path: Path):
    """Stage 1의 전제: 스트림의 발 접촉은 클래스가 아니라 회귀된 실수이고 [0, 1]을 살짝 넘는다."""
    convert.convert(mock_root / "DH/ObjsFrame.packets.bin", tmp_path / "clip")
    contact = pl.read_parquet(tmp_path / "clip/frames.parquet")["contact_l"].to_numpy()
    assert len(np.unique(contact)) > 2
    assert contact.min() < 0.0 or contact.max() > 1.0


def test_destinations_keep_colliding_names_apart(tmp_path: Path):
    sources = [Path("/a/DH/ObjsFrame.packets.bin"), Path("/a/v3/ObjsFrame.packets.bin")]
    mapped = convert.destinations(sources, tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in mapped] == ["DH/ObjsFrame", "v3/ObjsFrame"]


def test_basis_matches_the_generation(mock_root: Path):
    """패킹을 바꿔 읽으면 직교성이 깨진다 — 이 신호가 세대 판정의 근거다."""
    payloads = store.read_payloads(mock_root / "mocap_idle.bin")
    value = decode.decode(payloads, store.detect_kind(payloads))
    rotation = motion.frames(value.columns["rotation_6d"], value.layout.bone_count, 6)

    right = motion.basis_residual(rotation, value.layout)
    wrong = motion.basis_residual(rotation, decode.schema.motion_layout(22, "interleaved"))
    assert max(right.values()) < 1e-3
    assert wrong["max_orthogonality_error"] > 0.5
