"""클립 인벤토리 — 무엇이 몇 분어치 있는지 센다.

    uv run movin-clip-inventory <dir-or-bin> [...] [--json out.json]

Stage 0의 판단 재료다. 모델 선택은 데이터 규모가 정한다: 프레임 수, 실제 초, 사람 수,
모션이 유효한 비율, 포인트클라우드가 실린 프레임 비율.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import decode, store


def timestamps(value: decode.decoded) -> np.ndarray:
    name = "timestamp_sensor_ns" if value.kind.name == "objs_frame" else "lidar_timestamp_ns"
    return np.asarray(value.columns[name], dtype=np.uint64)


unrecognized = {
    "bone_count": 0,
    "packets": 0,
    "rows": 0,
    "distinct_frames": 0,
    "seconds": 0.0,
    "sensor_hz": None,
    "motion_hz": None,
    "subjects": [],
    "valid_motion_rows": 0,
    "pointcloud_rows": 0,
    "pointcloud_files": 0,
    "pointcloud_points": 0,
}


def survey(path: Path, label: str) -> dict:
    """모르는 파일은 크래시가 아니라 결과다 — 인벤토리는 "뭐가 있나"를 답하는 도구다.

    변환기(`convert`)는 반대로 크래시한다. 거기서는 모르는 파일이 곧 오류다.
    """
    payloads = store.read_payloads(path)
    try:
        kind = store.detect_kind(payloads)
    except store.store_error as error:
        return {"clip": label, "bytes": path.stat().st_size, "kind": "unrecognized",
                "reason": str(error), "packets": len(payloads), **{k: v for k, v in unrecognized.items() if k != "packets"}}
    value = decode.decode(payloads, kind)

    stamps = timestamps(value)
    unique = np.unique(stamps)
    span_ns = int(unique.max() - unique.min()) if unique.size > 1 else 0
    seconds = span_ns / 1e9
    valid = np.asarray([bool(flag) for flag in value.columns["motion_valid"]])
    # 서브프레임들이 같은 클라우드를 공유하므로 점 개수는 파일 단위로 센다.
    by_file = {
        name: points
        for name, points in zip(
            value.columns.get("pointcloud_file", []), value.columns.get("pointcloud_points", [])
        )
        if name is not None
    }

    return {
        "clip": label,
        "bytes": path.stat().st_size,
        "kind": kind.label(),
        "bone_count": value.layout.bone_count,
        "packets": len(payloads),
        "rows": value.row_count,
        "distinct_frames": int(unique.size),
        "seconds": round(seconds, 3),
        # 센서는 20 Hz로 들어오고 백엔드가 서브프레임으로 60 Hz 모션을 만든다.
        # 두 비율을 따로 봐야 업샘플러(Stage 2)의 입출력 규격이 보인다.
        "sensor_hz": round(unique.size / seconds, 2) if seconds > 0 else None,
        "motion_hz": round(value.row_count / len(set(value.columns["obj_id"])) / seconds, 2)
        if seconds > 0
        else None,
        "subjects": sorted({int(obj) for obj in value.columns["obj_id"]}),
        "valid_motion_rows": int(valid.sum()),
        "pointcloud_rows": sum(
            name is not None for name in value.columns.get("pointcloud_file", [])
        ),
        "pointcloud_files": len(by_file),
        "pointcloud_points": int(sum(by_file.values())),
    }


def sources(targets: list[Path]) -> list[tuple[Path, str]]:
    """`(경로, 표시 이름)`. 하위 디렉터리마다 `ObjsFrame.packets.bin`이 또 있으므로
    basename으로는 구분이 안 된다 — 스캔한 루트 기준 상대 경로로 부른다."""
    found: list[tuple[Path, str]] = []
    for target in targets:
        if target.is_dir():
            found.extend((path, str(path.relative_to(target))) for path in sorted(target.rglob("*.bin")))
        else:
            found.append((target, target.name))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="+", type=Path, help="클립 `.bin` 또는 그것을 담은 디렉터리")
    parser.add_argument("--json", type=Path, help="요약을 JSON으로도 쓴다")
    arguments = parser.parse_args()

    rows = [survey(path, label) for path, label in sources(arguments.targets)]

    width = max([len(row["clip"]) for row in rows] + [4])
    header = (
        f"{'clip':{width}s} {'kind':18s} {'bones':>5s} {'rows':>6s} {'frames':>6s} {'sec':>7s} "
        f"{'sensorHz':>8s} {'motionHz':>8s} {'valid':>6s} {'clouds':>6s} {'points':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        rate = lambda name: "-" if row[name] is None else f"{row[name]:.2f}"
        print(
            f"{row['clip']:{width}s} {row['kind']:18s} {row['bone_count']:5d} {row['rows']:6d} "
            f"{row['distinct_frames']:6d} {row['seconds']:7.2f} {rate('sensor_hz'):>8s} "
            f"{rate('motion_hz'):>8s} {row['valid_motion_rows']:6d} {row['pointcloud_files']:6d} "
            f"{row['pointcloud_points']:10d}"
        )

    total_seconds = sum(row["seconds"] for row in rows)
    total_valid = sum(row["valid_motion_rows"] for row in rows)
    print("-" * len(header))
    print(f"{len(rows)} clips, {total_seconds / 60:.1f} min, {total_valid} rows with valid motion")

    if arguments.json:
        arguments.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
