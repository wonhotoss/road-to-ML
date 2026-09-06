"""패킷 → 프레임 열(column). 값은 와이어 그대로 둔다.

행의 알갱이는 **(패킷, obj 엔트리)** 하나다. ObjsFrame 한 패킷에 사람이 여럿일 수
있고, 서브프레임 때문에 같은 `frame_id`가 여러 번 나온다. 둘 다 열로 남겨 두면
이후 단계에서 원하는 대로 접을 수 있다.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import schema, store


class decode_error(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class pointcloud:
    name: str
    points: np.ndarray  # (n, 4) float32 — PointXYZI, OpenGL


@dataclasses.dataclass(frozen=True)
class decoded:
    kind: store.kind
    layout: schema.motion_layout
    columns: dict[str, list]
    pointclouds: list[pointcloud]

    @property
    def row_count(self) -> int:
        return len(next(iter(self.columns.values())))


def floats(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<f4")


def split_motion(payload: bytes, layout: schema.motion_layout, where: str) -> dict[str, object]:
    values = floats(payload)
    if values.size != layout.float_count:
        raise decode_error(f"{where}: expected {layout.float_count} motion floats, got {values.size}")
    return {
        "position": values[layout.position_offset : layout.rotation_offset].tolist(),
        "rotation_6d": values[layout.rotation_offset : layout.velocity_offset].tolist(),
        "velocity": values[layout.velocity_offset : layout.angular_velocity_offset].tolist(),
        "angular_velocity": values[layout.angular_velocity_offset : layout.contact_offset].tolist(),
        "contact_l": float(values[layout.contact_offset]),
        "contact_r": float(values[layout.contact_offset + 1]),
        "skeleton_offset": values[layout.skeleton_offset :].tolist(),
        # 백엔드는 zone state가 PLAYING이 아닌 동안 회전·속도·오프셋을 0으로 채운 버퍼를
        # 그대로 보낸다(레거시 `subscriber.cs`는 그때 `create_frame`을 아예 건너뛰었다).
        # 값은 지우지 않고 쓸 수 있는 행인지만 표시한다.
        "motion_valid": bool(
            np.abs(values[layout.rotation_offset : layout.velocity_offset]).max() > 0
        ),
    }


empty_motion = {
    "position": None,
    "rotation_6d": None,
    "velocity": None,
    "angular_velocity": None,
    "contact_l": None,
    "contact_r": None,
    "skeleton_offset": None,
    "motion_valid": None,
}

objs_frame_columns = (
    "packet_index",
    "frame_id",
    "sub_frame_index",
    "total_sub_frames",
    "timestamp_sensor_ns",
    "timestamp_pub_ns",
    "obj_id",
    "obj_state",
    "valid",
    "has_motion",
    "num_points",
    "obj_pos_x",
    "obj_pos_y",
    "obj_pos_z",
    *empty_motion,
    "seg_bbox",
    "seg_mask_polygon_points",
    "seg_mask_polygon",
    "seg_mask_ring_lengths",
    "body_bbox",
    "body_keypoints_2d",
    "hand_left_tracked",
    "hand_left_held",
    "hand_left_status",
    "hand_left_keypoints_2d",
    "hand_left_box",
    "hand_right_tracked",
    "hand_right_held",
    "hand_right_status",
    "hand_right_keypoints_2d",
    "hand_right_box",
    "pointcloud_points",
    "pointcloud_file",
)

mocap_topic_columns = (
    "packet_index",
    "frame_num",
    "lidar_timestamp_ns",
    "send_timestamp_ns",
    "obj_id",
    "real",
    "zone_state",
    *empty_motion,
)


def hand_columns(side: str, hand) -> dict[str, object]:
    return {
        f"hand_{side}_tracked": bool(hand.tracked),
        f"hand_{side}_held": bool(hand.held),
        f"hand_{side}_status": int(hand.status),
        f"hand_{side}_keypoints_2d": floats(hand.keypoints_2d).tolist(),
        f"hand_{side}_box": [float(hand.box_cx), float(hand.box_cy), float(hand.box_side)],
    }


def decode_objs_frame(payloads: list[bytes], kind: store.kind) -> decoded:
    variant = kind.variant
    message_class = schema.objs_frame_class(variant)
    layout = schema.objs_frame_layout
    columns: dict[str, list] = {name: [] for name in objs_frame_columns}
    pointclouds: list[pointcloud] = []
    clouds_by_name: dict[str, np.ndarray] = {}

    for packet_index, payload in enumerate(payloads):
        frame = message_class()
        frame.ParseFromString(payload)
        for entry in frame.objs:
            where = f"packet {packet_index} obj {entry.obj_id}"
            motion = split_motion(bytes(entry.motion), layout, where) if entry.has_motion else empty_motion

            points = floats(entry.pointcloud)
            if points.size % schema.pointcloud_floats_per_point != 0:
                raise decode_error(f"{where}: pointcloud has {points.size} floats, not a multiple of 4")
            point_count = points.size // schema.pointcloud_floats_per_point

            # 한 센서 프레임의 서브프레임들은 **같은** 클라우드를 그대로 다시 싣는다
            # (측정: DH 클립 182 프레임 전부 바이트 동일). 센서는 20 Hz, 모션은 60 Hz다.
            # 그래서 파일은 센서 프레임당 하나만 쓰고 세 행이 그것을 함께 가리킨다.
            name = None
            if point_count > 0:
                name = f"{frame.timestamp_sensor_ns}_{entry.obj_id}.npy"
                shaped = points.reshape(point_count, schema.pointcloud_floats_per_point)
                seen = clouds_by_name.get(name)
                if seen is None:
                    clouds_by_name[name] = shaped
                    pointclouds.append(pointcloud(name=name, points=shaped))
                elif not np.array_equal(seen, shaped):
                    raise decode_error(
                        f"{where}: sub-frames of sensor frame {frame.timestamp_sensor_ns} "
                        f"carry different pointclouds; the 20 Hz assumption is wrong"
                    )

            row = {
                "packet_index": packet_index,
                "frame_id": int(frame.frame_id),
                "sub_frame_index": int(frame.sub_frame_index),
                "total_sub_frames": int(frame.total_sub_frames),
                "timestamp_sensor_ns": int(frame.timestamp_sensor_ns),
                "timestamp_pub_ns": int(frame.timestamp_pub_ns),
                "obj_id": int(entry.obj_id),
                "obj_state": int(entry.state),
                "valid": bool(entry.valid),
                "has_motion": bool(entry.has_motion),
                "num_points": int(entry.num_points),
                # motion6 세대에는 `pos_x/y/z`가 아예 없다 — 0이 아니라 null로 남긴다.
                "obj_pos_x": float(entry.pos_x) if variant.has_object_position else None,
                "obj_pos_y": float(entry.pos_y) if variant.has_object_position else None,
                "obj_pos_z": float(entry.pos_z) if variant.has_object_position else None,
                **motion,
                "seg_bbox": floats(entry.seg.bbox).tolist(),
                "seg_mask_polygon_points": int(entry.seg.mask_polygon_points),
                "seg_mask_polygon": floats(entry.seg.mask_polygon).tolist(),
                "seg_mask_ring_lengths": [int(value) for value in entry.seg.mask_polygon_ring_lengths],
                "body_bbox": floats(entry.body.bbox).tolist(),
                "body_keypoints_2d": floats(entry.body.keypoints_2d).tolist(),
                **hand_columns("left", entry.hand_left),
                **hand_columns("right", entry.hand_right),
                "pointcloud_points": point_count,
                "pointcloud_file": name,
            }
            for key, value in row.items():
                columns[key].append(value)

    return decoded(kind=kind, layout=layout, columns=columns, pointclouds=pointclouds)


def decode_mocap_topic(payloads: list[bytes], kind: store.kind) -> decoded:
    message_class = schema.mocap_topic_class()
    layout = schema.mocap_topic_layout
    columns: dict[str, list] = {name: [] for name in mocap_topic_columns}

    for packet_index, payload in enumerate(payloads):
        topic = message_class()
        topic.ParseFromString(payload)
        where = f"packet {packet_index}"
        motion = split_motion(bytes(topic.mocap_data), layout, where) if topic.mocap_data else empty_motion
        row = {
            "packet_index": packet_index,
            "frame_num": int(topic.header.frame_num),
            "lidar_timestamp_ns": int(topic.header.lidar_timestamp),
            "send_timestamp_ns": int(topic.header.send_timestamp),
            "obj_id": int(topic.metadata.id),
            "real": bool(topic.metadata.real),
            "zone_state": int(topic.metadata.zone_state),
            **motion,
        }
        for key, value in row.items():
            columns[key].append(value)

    return decoded(kind=kind, layout=layout, columns=columns, pointclouds=[])


def decode(payloads: list[bytes], kind: store.kind) -> decoded:
    if kind.name == "objs_frame":
        return decode_objs_frame(payloads, kind)
    if kind.name == "mocap_topic":
        return decode_mocap_topic(payloads, kind)
    raise decode_error(f"unknown packet kind: {kind.name}")
