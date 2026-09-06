"""와이어 스키마와 모션 float 배치.

두 세대의 클립이 있다:

- `movin.stream.ObjsFrame`  — 현재. 55본 989 float 모션 + 포인트클라우드 + 세그/2D.
  스키마는 Studio가 생성한 C#에 base64로 박혀 있어 `.desc`로 뽑아 커밋해 둔다.
  **필드 번호가 한 번 바뀌었다.** `e0f486e8`(2026-06-30)에서 `ObjEntry`에 `pos_x/y/z`가
  6~8번으로 들어오면서 `motion`이 6 → 9, 그 뒤가 전부 3씩 밀렸다. 그 이전에 녹화된
  스토어(커밋된 `mock/*.packets.bin`이 그렇다)는 옛 번호로 적혀 있고, 새 스키마로 읽으면
  proto3가 모르는 필드를 조용히 삼켜 **모션이 빈 채로 통과한다.** 그래서 두 세대를 모두
  들고 다니며 모션 바이트 수로 판정한다.
- `movin.MocapTopic`        — 레거시. 22본 395 float 모션만. `mocap.proto` + `common.proto`.
  descriptor를 여기서 직접 조립한다(protoc 불필요).

모션 float 배치는 Studio 소비자 코드가 진실의 원본이다:
`MOVIN_Studio_Unity/Assets/Scripts/scene/mocap+recreiver.cs` `create_frame`.

    [0]                  위치        bone_count * 3
    [b*3]                6D 회전     bone_count * 6
    [b*9]                선속도      bone_count * 3
    [b*12]               각속도      bone_count * 3
    [b*15]               발 접촉     2                (L, R)
    [b*15 + 2]           스켈레톤 오프셋 (bone_count - 1) * 3
    총                               bone_count * 18 - 1

6D 회전의 **본당 6 float 배치가 두 세대에서 서로 다르다.** 어느 쪽이든 두 기저벡터
xB, yB를 정규화하고 zB = cross(xB, yB)로 프레임을 세우지만,

    objs_frame   `interleaved`  (xBx, yBx, xBy, yBy, xBz, yBz)  — mocap+recreiver.cs
    mocap_topic  `consecutive`  (xBx, yBx, zBx, xBy, yBy, zBy)  — 옛 backend/subscriber.cs

이다. 틀린 쪽으로 읽어도 크래시하지 않고 그럴듯한 숫자가 나오므로(정규화가 다 삼킨다)
`movin_clip.motion.basis`를 거쳐 읽고, 검증은 |xB| = |yB| = 1, xB·yB = 0으로 한다.

좌표계는 **와이어 그대로(OpenGL)** 보존한다. Studio는 읽는 순간
`Utility.ConvertOpenGLToUnity` (위치 `x → -x`, 회전 `(x, -y, -z, w)`)를 적용한다.
스켈레톤 오프셋은 현재 세대가 변환하지 않고 레거시는 변환했다 — 이런 규약 차이 때문에
여기서는 아무 변환도 하지 않고 `meta.json`에 무엇으로 적혀 있는지만 남긴다.
"""

from __future__ import annotations

import base64
import dataclasses
import re
from pathlib import Path
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

package_root = Path(__file__).resolve().parent

studio_descriptor_source = "MOVIN_Studio_Unity/Assets/Scripts/backend/proto/ObjsFrame.cs"


@dataclasses.dataclass(frozen=True)
class objs_frame_variant:
    name: str
    descriptor_file: str
    studio_ref: str
    has_object_position: bool

    @property
    def descriptor_path(self) -> Path:
        return package_root / self.descriptor_file


# 새것부터. `detect`가 이 순서로 시도한다.
objs_frame_variants = (
    objs_frame_variant(
        name="motion9",
        descriptor_file="objs_frame_motion9.desc",
        studio_ref="e0f486e8",
        has_object_position=True,
    ),
    objs_frame_variant(
        name="motion6",
        descriptor_file="objs_frame_motion6.desc",
        studio_ref="775556a0",
        has_object_position=False,
    ),
)


def variant_by_name(name: str) -> objs_frame_variant:
    match = next((value for value in objs_frame_variants if value.name == name), None)
    if match is None:
        raise schema_error(f"unknown ObjsFrame variant: {name}")
    return match

objs_frame_message_name = "movin.stream.ObjsFrame"
mocap_topic_message_name = "movin.MocapTopic"

pointcloud_floats_per_point = 4  # PointXYZI


class schema_error(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class motion_layout:
    bone_count: int
    rotation_packing: str  # "interleaved" | "consecutive"

    @property
    def offset_count(self) -> int:
        return self.bone_count - 1

    @property
    def position_offset(self) -> int:
        return 0

    @property
    def rotation_offset(self) -> int:
        return self.bone_count * 3

    @property
    def velocity_offset(self) -> int:
        return self.bone_count * 9

    @property
    def angular_velocity_offset(self) -> int:
        return self.bone_count * 12

    @property
    def contact_offset(self) -> int:
        return self.bone_count * 15

    @property
    def skeleton_offset(self) -> int:
        return self.bone_count * 15 + 2

    @property
    def float_count(self) -> int:
        return self.bone_count * 18 - 1

    @property
    def byte_count(self) -> int:
        return self.float_count * 4


objs_frame_layout = motion_layout(bone_count=55, rotation_packing="interleaved")
mocap_topic_layout = motion_layout(bone_count=22, rotation_packing="consecutive")


def descriptor_from_studio(studio_root: Path) -> descriptor_pb2.FileDescriptorProto:
    """Studio가 생성한 C#에 박힌 base64 FileDescriptorProto를 꺼낸다."""
    path = studio_root / studio_descriptor_source
    source = path.read_text(encoding="utf-8")
    match = re.search(r"FromBase64String\(\s*string\.Concat\((.*?)\)\)", source, re.DOTALL)
    if match is None:
        raise schema_error(f"{path}: embedded descriptor not found")
    encoded = "".join(re.findall(r'"([A-Za-z0-9+/=]+)"', match.group(1)))
    if not encoded:
        raise schema_error(f"{path}: embedded descriptor is empty")
    return descriptor_pb2.FileDescriptorProto.FromString(base64.b64decode(encoded))


def objs_frame_descriptor(variant: objs_frame_variant) -> descriptor_pb2.FileDescriptorProto:
    return descriptor_pb2.FileDescriptorProto.FromString(variant.descriptor_path.read_bytes())


def mocap_topic_descriptors() -> list[descriptor_pb2.FileDescriptorProto]:
    """레거시 `common.proto` + `mocap.proto`를 손으로 조립한다.

    두 파일은 합쳐 6개 필드뿐이고 protoc 의존을 지우는 편이 재현에 낫다.
    원본은 `MOVIN_Studio_Unity/Assets/Scripts/backend/proto/{common,mocap}.proto`.
    """
    label_optional = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    type_of = descriptor_pb2.FieldDescriptorProto

    common = descriptor_pb2.FileDescriptorProto(name="common.proto", package="movin", syntax="proto3")
    header = common.message_type.add(name="Header")
    header.field.add(name="frame_num", number=1, label=label_optional, type=type_of.TYPE_UINT32)
    header.field.add(name="lidar_timestamp", number=2, label=label_optional, type=type_of.TYPE_UINT64)
    header.field.add(name="send_timestamp", number=3, label=label_optional, type=type_of.TYPE_UINT64)

    mocap = descriptor_pb2.FileDescriptorProto(
        name="mocap.proto", package="movin", syntax="proto3", dependency=["common.proto"]
    )
    zone_state = mocap.enum_type.add(name="ZoneState")
    for index, name in enumerate(("NONE", "ENTERED", "PREPARING", "CALIBRATING", "PLAYING")):
        zone_state.value.add(name=name, number=index)

    metadata = mocap.message_type.add(name="MocapMetadata")
    metadata.field.add(name="id", number=1, label=label_optional, type=type_of.TYPE_UINT32)
    metadata.field.add(name="real", number=2, label=label_optional, type=type_of.TYPE_BOOL)
    metadata.field.add(
        name="zone_state", number=3, label=label_optional, type=type_of.TYPE_ENUM, type_name=".movin.ZoneState"
    )

    topic = mocap.message_type.add(name="MocapTopic")
    topic.field.add(
        name="header", number=1, label=label_optional, type=type_of.TYPE_MESSAGE, type_name=".movin.Header"
    )
    topic.field.add(
        name="metadata", number=2, label=label_optional, type=type_of.TYPE_MESSAGE, type_name=".movin.MocapMetadata"
    )
    topic.field.add(name="mocap_data", number=3, label=label_optional, type=type_of.TYPE_BYTES)

    return [common, mocap]


def message_class(descriptors: list[descriptor_pb2.FileDescriptorProto], name: str) -> Any:
    pool = descriptor_pool.DescriptorPool()
    for descriptor in descriptors:
        pool.Add(descriptor)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(name))


def objs_frame_class(variant: objs_frame_variant) -> Any:
    return message_class([objs_frame_descriptor(variant)], objs_frame_message_name)


def mocap_topic_class() -> Any:
    return message_class(mocap_topic_descriptors(), mocap_topic_message_name)
