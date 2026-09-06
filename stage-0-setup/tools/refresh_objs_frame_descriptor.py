#!/usr/bin/env python3
"""Studio의 생성 C#에서 ObjsFrame 스키마를 다시 뽑아 커밋한다.

    uv run tools/refresh_objs_frame_descriptor.py <studio-root> [variant ...]

variant를 안 주면 전부 갱신한다. 각 variant는 자기 Studio 커밋(`studio_ref`)을
`git show`로 꺼내 읽으므로 작업 트리 상태와 무관하다.
Studio가 스키마를 또 바꾸면 `schema.objs_frame_variants`에 새 variant를 추가하고 여기를 돌린다.
"""

from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from google.protobuf import descriptor_pb2

from movin_clip import schema


def descriptor_at_ref(studio_root: Path, ref: str) -> descriptor_pb2.FileDescriptorProto:
    source = subprocess.run(
        ["git", "show", f"{ref}:{schema.studio_descriptor_source}"],
        cwd=studio_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"FromBase64String\(\s*string\.Concat\((.*?)\)\)", source, re.DOTALL)
    if match is None:
        raise schema.schema_error(f"{ref}: embedded descriptor not found")
    encoded = "".join(re.findall(r'"([A-Za-z0-9+/=]+)"', match.group(1)))
    return descriptor_pb2.FileDescriptorProto.FromString(base64.b64decode(encoded))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    studio_root = Path(sys.argv[1])
    names = sys.argv[2:]
    variants = (
        [schema.variant_by_name(name) for name in names] if names else list(schema.objs_frame_variants)
    )

    for variant in variants:
        payload = descriptor_at_ref(studio_root, variant.studio_ref).SerializeToString()
        path = variant.descriptor_path
        previous = path.read_bytes() if path.exists() else b""
        path.write_bytes(payload)
        verdict = "unchanged" if payload == previous else f"updated ({len(previous)} -> {len(payload)} bytes)"
        print(f"{variant.name} @ {variant.studio_ref}: {path.name} {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
