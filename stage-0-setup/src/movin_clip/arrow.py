"""디코드 결과 → Arrow 테이블. 열 타입을 명시해 파일 간 스키마가 흔들리지 않게 한다."""

from __future__ import annotations

import pyarrow as pa

from . import decode, schema

f32 = pa.float32()
u32 = pa.uint32()
u64 = pa.uint64()


def motion_fields(layout: schema.motion_layout) -> dict[str, pa.DataType]:
    bones = layout.bone_count
    return {
        "position": pa.list_(f32, bones * 3),
        "rotation_6d": pa.list_(f32, bones * 6),
        "velocity": pa.list_(f32, bones * 3),
        "angular_velocity": pa.list_(f32, bones * 3),
        "contact_l": f32,
        "contact_r": f32,
        "skeleton_offset": pa.list_(f32, layout.offset_count * 3),
        "motion_valid": pa.bool_(),
    }


def objs_frame_schema(layout: schema.motion_layout) -> pa.Schema:
    fields = {
        "packet_index": pa.int32(),
        "frame_id": u32,
        "sub_frame_index": u32,
        "total_sub_frames": u32,
        "timestamp_sensor_ns": u64,
        "timestamp_pub_ns": u64,
        "obj_id": u32,
        "obj_state": pa.int32(),
        "valid": pa.bool_(),
        "has_motion": pa.bool_(),
        "num_points": u32,
        "obj_pos_x": f32,
        "obj_pos_y": f32,
        "obj_pos_z": f32,
        **motion_fields(layout),
        "seg_bbox": pa.list_(f32),
        "seg_mask_polygon_points": u32,
        "seg_mask_polygon": pa.list_(f32),
        "seg_mask_ring_lengths": pa.list_(u32),
        "body_bbox": pa.list_(f32),
        "body_keypoints_2d": pa.list_(f32),
        "hand_left_tracked": pa.bool_(),
        "hand_left_held": pa.bool_(),
        "hand_left_status": u32,
        "hand_left_keypoints_2d": pa.list_(f32),
        "hand_left_box": pa.list_(f32, 3),
        "hand_right_tracked": pa.bool_(),
        "hand_right_held": pa.bool_(),
        "hand_right_status": u32,
        "hand_right_keypoints_2d": pa.list_(f32),
        "hand_right_box": pa.list_(f32, 3),
        "pointcloud_points": u32,
        "pointcloud_file": pa.string(),
    }
    return pa.schema([pa.field(name, kind) for name, kind in fields.items()])


def mocap_topic_schema(layout: schema.motion_layout) -> pa.Schema:
    fields = {
        "packet_index": pa.int32(),
        "frame_num": u32,
        "lidar_timestamp_ns": u64,
        "send_timestamp_ns": u64,
        "obj_id": u32,
        "real": pa.bool_(),
        "zone_state": pa.int32(),
        **motion_fields(layout),
    }
    return pa.schema([pa.field(name, kind) for name, kind in fields.items()])


def table(value: decode.decoded) -> pa.Table:
    builder = objs_frame_schema if value.kind.name == "objs_frame" else mocap_topic_schema
    arrow_schema = builder(value.layout)
    if list(value.columns) != arrow_schema.names:
        raise decode.decode_error(
            f"column mismatch: decoded {list(value.columns)}, schema {arrow_schema.names}"
        )
    return pa.table(value.columns, schema=arrow_schema)
