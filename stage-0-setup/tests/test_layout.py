"""모션 float 배치가 Studio 소비자 코드와 같은 숫자인지 못 박는다.

기대값은 `scene/mocap+recreiver.cs`의 `create_frame`이 소비하는 순서에서 손으로 센 것이고,
`tools/motion_parity/bvh_parity.py`의 상수와도 일치한다.
"""

from __future__ import annotations

import pytest

from movin_clip import schema


def test_objs_frame_layout():
    layout = schema.objs_frame_layout
    assert layout.bone_count == 55
    assert layout.offset_count == 54
    assert layout.float_count == 989
    assert layout.byte_count == 3956
    assert (layout.position_offset, layout.rotation_offset) == (0, 165)
    assert (layout.velocity_offset, layout.angular_velocity_offset) == (495, 660)
    assert (layout.contact_offset, layout.skeleton_offset) == (825, 827)
    assert layout.rotation_packing == "interleaved"


def test_mocap_topic_layout():
    layout = schema.mocap_topic_layout
    assert layout.bone_count == 22
    assert layout.float_count == 395  # mocap.proto 주석의 "395 floats"와 일치
    assert layout.rotation_packing == "consecutive"


def test_layout_blocks_tile_exactly():
    for layout in (schema.objs_frame_layout, schema.mocap_topic_layout):
        bounds = [
            layout.position_offset,
            layout.rotation_offset,
            layout.velocity_offset,
            layout.angular_velocity_offset,
            layout.contact_offset,
            layout.skeleton_offset,
            layout.float_count,
        ]
        assert bounds == sorted(bounds)
        assert bounds[-1] - bounds[-2] == layout.offset_count * 3


@pytest.mark.parametrize("variant", schema.objs_frame_variants, ids=lambda v: v.name)
def test_vendored_descriptors_load(variant):
    message = schema.objs_frame_class(variant)()
    fields = {field.name: field.number for field in message.DESCRIPTOR.fields}
    assert fields["objs"] == 6
    entry = message.DESCRIPTOR.fields_by_name["objs"].message_type
    motion = entry.fields_by_name["motion"].number
    assert motion == (9 if variant.has_object_position else 6)


def test_mocap_descriptor_assembles():
    message = schema.mocap_topic_class()()
    assert {field.name for field in message.DESCRIPTOR.fields} == {"header", "metadata", "mocap_data"}
