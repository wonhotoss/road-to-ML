"""패킷 스토어 컨테이너.

Studio가 쓰는 형식(`Extensions/Utility.cs` `packet_store`):

    int32  packet_count
    반복:  int32 payload_length, payload_length 바이트

payload는 protobuf 메시지 하나다. 어느 메시지인지는 파일이 말해주지 않으므로
`detect_kind`가 두 스키마로 파싱을 시도해 판정한다.
"""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

from . import schema

max_packet_bytes = 256 * 1024 * 1024


class store_error(Exception):
    pass


def read_payloads(path: Path) -> list[bytes]:
    data = path.read_bytes()
    if len(data) < 4:
        raise store_error(f"{path}: missing packet count header")

    count = struct.unpack_from("<i", data, 0)[0]
    if count < 0:
        raise store_error(f"{path}: negative packet count {count}")

    offset = 4
    payloads = []
    for index in range(count):
        if offset + 4 > len(data):
            raise store_error(f"{path}: truncated length for packet {index}")
        length = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        if length < 0 or length > max_packet_bytes:
            raise store_error(f"{path}: invalid packet {index} length {length}")
        if offset + length > len(data):
            raise store_error(f"{path}: truncated payload for packet {index}")
        payloads.append(data[offset : offset + length])
        offset += length

    if offset != len(data):
        raise store_error(f"{path}: {len(data) - offset} trailing bytes")
    return payloads


@dataclasses.dataclass(frozen=True)
class kind:
    name: str  # "objs_frame" | "mocap_topic"
    variant: schema.objs_frame_variant | None

    def label(self) -> str:
        return self.name if self.variant is None else f"{self.name}/{self.variant.name}"


def detect_kind(payloads: list[bytes]) -> kind:
    """어떤 메시지·어떤 세대인지 **모션 바이트 수로** 가른다.

    "파싱 성공"은 판정이 못 된다. proto3는 모르는 필드를 조용히 보관하므로 틀린 스키마로도
    파싱은 늘 통과하고, 모션만 빈 채로 나온다. 실제로 커밋된 `mock/*.packets.bin`은
    옛 필드 번호(motion=6)라서 현재 스키마로 읽으면 전 프레임이 모션 없는 것처럼 보인다.

    그래서 유일하게 믿을 수 있는 신호를 쓴다: 모션 payload가 그 스키마가 약속한 길이
    (55본 3956 B, 22본 1580 B)로 나오는가.
    """
    if not payloads:
        raise store_error("empty packet store")

    candidates = [
        (kind("objs_frame", variant), schema.objs_frame_class(variant))
        for variant in schema.objs_frame_variants
    ]
    mocap_topic = schema.mocap_topic_class()

    for payload in payloads:
        for candidate, message_class in candidates:
            frame = message_class()
            frame.ParseFromString(payload)
            if any(len(entry.motion) == schema.objs_frame_layout.byte_count for entry in frame.objs):
                return candidate

        topic = mocap_topic()
        topic.ParseFromString(payload)
        if len(topic.mocap_data) == schema.mocap_topic_layout.byte_count:
            return kind("mocap_topic", None)

    raise store_error("no packet carries a motion payload of any known schema")
