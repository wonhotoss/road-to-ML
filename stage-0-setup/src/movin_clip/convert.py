"""클립 `.bin` → `<out>/<clip>/{frames.parquet, pointcloud/*.npy, meta.json}`.

    uv run movin-clip-convert <clip.bin> [...] --out <dir>

값은 와이어 그대로다(OpenGL 좌표계, 변환 없음). 규약은 `meta.json`에 적어 둔다.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

from . import arrow, decode, motion, schema, store


rotation_note = {
    "interleaved": "per bone (xBx, yBx, xBy, yBy, xBz, yBz); read via movin_clip.motion.basis",
    "consecutive": "per bone (xBx, yBx, zBx, xBy, yBy, zBy); read via movin_clip.motion.basis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, destination: Path) -> dict:
    payloads = store.read_payloads(source)
    kind = store.detect_kind(payloads)
    value = decode.decode(payloads, kind)

    destination.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.table(value), destination / "frames.parquet", compression="zstd")

    points_total = 0
    if value.pointclouds:
        cloud_root = destination / "pointcloud"
        cloud_root.mkdir(exist_ok=True)
        for cloud in value.pointclouds:
            np.save(cloud_root / cloud.name, cloud.points)
            points_total += cloud.points.shape[0]

    # 패킹을 잘못 골랐으면 여기서 드러난다. 맞으면 셋 다 0에 붙는다.
    rotation = [
        row
        for row, valid in zip(value.columns["rotation_6d"], value.columns["motion_valid"])
        if valid
    ]
    residual = (
        motion.basis_residual(motion.frames(rotation, value.layout.bone_count, 6), value.layout)
        if rotation
        else None
    )
    tolerance = 1e-3
    if residual is not None and max(residual.values()) > tolerance:
        raise decode.decode_error(
            f"{source}: rotation_6d is not an orthonormal basis under "
            f"{value.layout.rotation_packing} packing ({residual}); the schema generation is wrong"
        )

    meta = {
        "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": sha256(source)},
        "kind": kind.name,
        "variant": None if kind.variant is None else kind.variant.name,
        "packets": len(payloads),
        "rows": value.row_count,
        "rows_with_valid_motion": len(rotation),
        "bone_count": value.layout.bone_count,
        "motion_float_count": value.layout.float_count,
        "pointcloud": {
            "files": len(value.pointclouds),
            "rows_referencing": sum(
                name is not None for name in value.columns.get("pointcloud_file", [])
            ),
            "points": points_total,
            "floats_per_point": schema.pointcloud_floats_per_point,
            "format": "PointXYZI",
        },
        "convention": {
            "coordinates": "OpenGL, as received on the wire; no ConvertOpenGLToUnity applied",
            "motion_layout": "position, rotation_6d, velocity, angular_velocity, contact(L,R), skeleton_offset",
            "rotation_packing": value.layout.rotation_packing,
            "rotation_6d": rotation_note[value.layout.rotation_packing],
        },
        "rotation_check": residual,
        "converted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def destinations(sources: list[Path], out: Path) -> list[Path]:
    """소스의 디렉터리 구조를 출력에 그대로 편다.

    녹화 디렉터리마다 파일 이름이 `ObjsFrame.packets.bin`으로 같으므로 basename만
    쓰면 서로를 덮어쓴다. 공통 부모를 기준으로 상대 경로를 살린다.
    """
    root = Path(os.path.commonpath([source.resolve().parent for source in sources]))
    mapped = [
        out / source.resolve().parent.relative_to(root) / source.name.removesuffix(".bin").removesuffix(".packets")
        for source in sources
    ]
    if len(set(mapped)) != len(mapped):
        raise decode.decode_error(f"two sources map to the same destination: {mapped}")
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path, help="클립 `.bin` 경로")
    parser.add_argument("--out", required=True, type=Path, help="출력 루트")
    arguments = parser.parse_args()

    for source, destination in zip(arguments.sources, destinations(arguments.sources, arguments.out)):
        meta = convert(source, destination)
        cloud = meta["pointcloud"]
        print(
            f"{destination.relative_to(arguments.out)}: {meta['kind']}"
            f"{'' if meta['variant'] is None else '/' + meta['variant']}"
            f" packets={meta['packets']} rows={meta['rows']} "
            f"bones={meta['bone_count']} clouds={cloud['files']} points={cloud['points']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
