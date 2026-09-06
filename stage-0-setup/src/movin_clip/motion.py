"""모션 열을 다루는 helper. 6D 회전은 반드시 여기를 거쳐 읽는다.

세대마다 본당 6 float의 배치가 다르고(`schema.motion_layout.rotation_packing`)
틀린 쪽으로 읽어도 조용히 그럴듯한 값이 나오기 때문이다.
"""

from __future__ import annotations

import numpy as np

from . import schema

rotation_axes = {
    "interleaved": (( 0, 2, 4), (1, 3, 5)),
    "consecutive": (( 0, 1, 2), (3, 4, 5)),
}


class motion_error(Exception):
    pass


def frames(column, bone_count: int, per_bone: int) -> np.ndarray:
    """Parquet의 평평한 열 → `(frame, bone, per_bone)`."""
    values = np.stack([np.asarray(row, dtype=np.float32) for row in column])
    return values.reshape(len(values), bone_count, per_bone)


def unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def basis(rotation_6d: np.ndarray, layout: schema.motion_layout) -> np.ndarray:
    """`(..., 6)` → 회전행렬 `(..., 3, 3)`. 열이 순서대로 (right, up, forward).

    재직교화는 `bvh_parity.rotation_from_6d`와 같은 순서로 한다. Studio의
    `Quaternion.LookRotation(zB, yB)`도 결국 같은 것을 하지만, 참조 구현을 그대로 따라야
    두 결과를 숫자로 비교할 수 있다.
    """
    axes = rotation_axes.get(layout.rotation_packing)
    if axes is None:
        raise motion_error(f"unknown rotation packing: {layout.rotation_packing}")
    x_axis, y_axis = axes
    x = unit(rotation_6d[..., x_axis])
    y = unit(rotation_6d[..., y_axis])
    forward = unit(np.cross(x, y))
    right = unit(np.cross(y, forward))
    up = unit(np.cross(forward, right))
    return np.stack([right, up, forward], axis=-1)


def basis_residual(rotation_6d: np.ndarray, layout: schema.motion_layout) -> dict[str, float]:
    """패킹이 맞는지 재는 자. 맞으면 셋 다 0에 붙는다."""
    axes = rotation_axes[layout.rotation_packing]
    raw_x = rotation_6d[..., axes[0]]
    raw_y = rotation_6d[..., axes[1]]
    norm_x = np.linalg.norm(raw_x, axis=-1)
    norm_y = np.linalg.norm(raw_y, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = np.einsum(
            "...i,...i->...", raw_x / norm_x[..., None], raw_y / norm_y[..., None]
        )
    return {
        "max_x_norm_error": float(np.abs(norm_x - 1.0).max()),
        "max_y_norm_error": float(np.abs(norm_y - 1.0).max()),
        "max_orthogonality_error": float(np.nanmax(np.abs(dot))),
    }
