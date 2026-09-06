"""Studio 자체 디코더(`tools/motion_parity/bvh_parity.py`)와 값이 같은지 대조한다.

우리 디코더가 float 989개를 옳게 가르는지 확인하는 유일한 독립 근거다.
`MOVIN_STUDIO_ROOT`가 있어야 돈다(머신마다 경로가 다르다).

`bvh_parity`는 현 세대 스키마(motion=9)를 전제하지만 `decode_stream_record`가 만지는 것은
`objs`/`has_motion`/`motion`뿐이라, 옛 세대(motion=6)로 파싱한 메시지를 그대로 먹여도 된다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from movin_clip import decode, motion, schema, store

studio_root = os.environ.get("MOVIN_STUDIO_ROOT")

pytestmark = pytest.mark.skipif(studio_root is None, reason="MOVIN_STUDIO_ROOT is not set")


def bvh_parity_module():
    sys.path.insert(0, str(Path(studio_root) / "tools/motion_parity"))
    import bvh_parity

    return bvh_parity


def fixtures() -> list[Path]:
    if studio_root is None:
        return []
    return sorted((Path(studio_root) / "MOVIN_Studio_Unity/mock").glob("*.packets.bin"))


class fake_key:
    def label(self) -> str:
        return "fixture"


class fake_record:
    def __init__(self, index: int, message):
        self.index = index
        self.message = message
        self.key = fake_key()


@pytest.mark.parametrize("path", fixtures(), ids=lambda p: p.name)
def test_matches_studio_decoder(path: Path):
    parity = bvh_parity_module()

    payloads = store.read_payloads(path)
    kind = store.detect_kind(payloads)
    ours = decode.decode(payloads, kind)

    message_class = schema.objs_frame_class(kind.variant)
    theirs = []
    for index, payload in enumerate(payloads):
        message = message_class()
        message.ParseFromString(payload)
        frame = parity.decode_stream_record(fake_record(index, message), "studio")
        if frame is not None:
            theirs.append(frame)

    assert len(theirs) == ours.row_count

    layout = ours.layout
    position = motion.frames(ours.columns["position"], layout.bone_count, 3)
    rotation = motion.basis(motion.frames(ours.columns["rotation_6d"], layout.bone_count, 6), layout)
    offsets = motion.frames(ours.columns["skeleton_offset"], layout.offset_count, 3)

    for index, frame in enumerate(theirs):
        # bvh_parity는 root를 positions[0] + rot[0] * positions[1]로 합성한다.
        root = position[index, 0] + rotation[index, 0] @ position[index, 1]
        assert np.allclose(root, frame.root_position, atol=1e-5), f"frame {index} root"

        # 그리고 bone 0, 1의 회전을 하나로 합쳐 54개를 만든다.
        combined = np.concatenate([(rotation[index, 0] @ rotation[index, 1])[None], rotation[index, 2:]])
        assert np.allclose(combined, np.asarray(frame.rotations), atol=1e-5), f"frame {index} rotations"

        assert np.allclose(offsets[index], np.asarray(frame.offsets), atol=1e-6), f"frame {index} offsets"
