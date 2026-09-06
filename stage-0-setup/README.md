# Stage 0 — 발판: 클립 → Parquet

`MOVIN-Studio-App`의 녹화 클립(`*.packets.bin`)을 ML이 바로 먹을 수 있는
**프레임 단위 Parquet + 포인트클라우드 `.npy`** 로 바꾼다. 이후 모든 단계의 입력이다.

값은 **와이어 그대로** 옮긴다. 좌표계 변환도, 회전 표현 변경도 하지 않는다. 규약은
`meta.json`에 적어 둔다 — 변환은 손실이고, 무엇을 잃었는지 나중에 되짚을 수 없기 때문이다.

## 쓰기

```sh
uv sync

# 뭐가 얼마나 있나
uv run movin-clip-inventory <녹화_디렉터리> --json inventory.json

# 변환 (디렉터리 구조를 그대로 편다)
uv run movin-clip-convert <clip.bin> [...] --out <출력_루트>

# 테스트 (Studio 리포가 있어야 픽스처가 있다)
MOVIN_STUDIO_ROOT=<studio-repo> uv run pytest tests -q
```

출력:

```
<출력_루트>/<클립>/
  frames.parquet            프레임(정확히는 패킷 × obj 엔트리) 한 행씩
  pointcloud/<sensor_ts>_<obj_id>.npy   (n, 4) float32, PointXYZI
  meta.json                 출처 sha256, 세대, 규약, 회전 직교성 자체검증 결과
```

## 읽어낸 것 — 규격 (2026-09-05)

전부 Studio 소스에서 확인했고, 커밋된 mock 픽스처로 재현한 것이다. 근거 파일은 괄호 안.

### 1. `ObjsFrame` 와이어 스키마는 한 번 바뀌었고, 틀리게 읽어도 조용하다

`e0f486e8`(2026-06-30)에서 `ObjEntry`에 `pos_x/y/z`가 6~8번으로 들어오며 `motion`이 6 → 9,
그 뒤 필드가 전부 3씩 밀렸다. 그 이전에 녹화된 스토어를 새 스키마로 읽으면 **크래시하지 않고
`has_motion=true`인 채 모션만 빈 상태로 통과한다** — proto3가 모르는 필드를 조용히 보관하기 때문.

그래서 두 세대의 descriptor를 모두 커밋해 두고(`src/movin_clip/objs_frame_motion{6,9}.desc`),
**모션 payload 길이로** 판정한다(`store.detect_kind`). Studio가 또 바꾸면
`schema.objs_frame_variants`에 추가하고 `tools/refresh_objs_frame_descriptor.py`를 돌린다.

커밋된 `mock/*.packets.bin`은 대부분 옛 세대(`motion6`)다. `DH/`만 현재 세대다.

### 2. 두 메시지 계열이 섞여 있다

| 계열 | 본 | 모션 float | 포인트클라우드 | 출처 |
|---|---|---|---|---|
| `movin.stream.ObjsFrame` | 55 | 989 | 있음 | `backend/proto/ObjsFrame.cs` (base64 descriptor 내장) |
| `movin.MocapTopic` (레거시) | 22 | 395 | 없음 | `backend/proto/{mocap,common}.proto` |

레거시 소비자는 `c645d9c0`에서 지워졌다(`git show c645d9c0^:…/backend/subscriber.cs`).
`mocap.proto`는 작아서 descriptor를 코드로 조립한다 — protoc 의존이 없다.

### 3. 모션 989 float의 배치

`scene/mocap+recreiver.cs`의 `create_frame`이 소비하는 순서 그대로다.

```
[0]    위치      55×3     [165]  6D 회전  55×6     [495]  선속도  55×3
[660]  각속도    55×3     [825]  접촉 L,R  2       [827]  스켈레톤 오프셋 54×3   = 989
```

레거시 22본도 **같은 순서**이고 개수만 다르다(`18·b − 1 = 395`).

### 4. 6D 회전의 본당 배치가 두 계열에서 서로 다르다 — 조용한 함정

```
ObjsFrame   interleaved   (xBx, yBx, xBy, yBy, xBz, yBz)   mocap+recreiver.cs
MocapTopic  consecutive   (xBx, yBx, zBx, xBy, yBy, zBy)   옛 backend/subscriber.cs
```

틀린 쪽으로 읽어도 정규화가 다 삼켜서 그럴듯한 숫자가 나온다. 실제로 잘못 읽으면
`|xB|`가 0.31~1.41로 흩어지고 `xB·yB`가 0.99까지 간다 — 맞게 읽으면 `1.0000`과 `1e-7`이다.
**그래서 `movin_clip.motion.basis`를 거쳐서만 읽고**, 변환기는 매번 직교성을 재서
`meta.json.rotation_check`에 남기고 어긋나면 크래시한다.

### 5. 센서는 20 Hz, 모션은 60 Hz — 그 사이가 서브프레임이다

`total_sub_frames = 3`, `sub_frame_index = 0..2`. 같은 `timestamp_sensor_ns`를 가진 세 행이
서로 다른 모션을 싣는다. `frame_id`는 서브프레임까지 세므로 3씩 뛴다.

**세 서브프레임의 포인트클라우드는 바이트 단위로 동일하다**(DH 클립 182 센서 프레임 전부 확인).
즉 클라우드는 20 Hz다. 그래서 `.npy`는 센서 프레임당 하나만 쓰고 세 행이 함께 가리킨다
(중복 제거로 15 MB → 7.5 MB). 다르면 크래시한다 — 이 가정이 깨졌다는 뜻이므로.

이것이 곧 Stage 2 업샘플러의 실제 규격이다: **20 Hz 관측 → 60 Hz 모션**은 이미 백엔드가
하고 있는 일이고, 데이터에 그 입출력이 그대로 들어 있다.

### 6. 스트림의 발 접촉은 이진 라벨이 아니다

연속값이고 `[0, 1]`을 살짝 벗어난다(DH 클립: −0.003 ~ 1.0007). 백엔드 모델의 **회귀 출력**이지
정답이 아니다. 로드맵 원칙 4가 말하는 그대로다 — Stage 1에서 이것을 정답 삼아 학습하면
개선이 아니라 증류다. `contact_l`/`contact_r`은 float으로 그대로 싣는다.

### 7. PLAYING이 아닌 프레임은 0으로 채워져 온다

`mocap_entering.bin`의 53행 전부 `zone_state = ENTERED`이고 회전·속도·오프셋이 정확히 0이다
(레거시 `subscriber.cs`는 그때 `create_frame`을 건너뛰었다). 값은 지우지 않고
**`motion_valid` 열로 표시**한다. 직교성 검증도 유효 행에만 건다.

### 8. 나머지 필드의 폭

`body.keypoints_2d` 51 float = COCO-17 × (x, y, conf) · `body.bbox`/`seg.bbox` 4 float ·
손 `keypoints_2d` 42 float = 21 × (x, y), 미추적이면 0 · 포인트클라우드 4 float/점 (PointXYZI).
`num_points`는 실제 payload 점 개수와 일치했다(DH 클립 전 행).

### 9. 공짜 지연 계측

`timestamp_pub_ns − timestamp_sensor_ns` = DH 클립에서 최소 46.5 ms, 중앙값 64.5 ms, 최대 89.8 ms.
로드맵이 "공짜 메트릭"이라 부른 것의 첫 실측값이다.

## 검증

- **Studio 자체 디코더와 대조** — `tests/test_studio_parity.py`가
  `tools/motion_parity/bvh_parity.py`의 `decode_stream_record`를 같은 패킷에 돌려
  root 위치·회전 54개·오프셋 54개를 `1e-5`/`1e-6` 이내로 비교한다. 두 픽스처 모두 일치.
  우리가 float 989개를 옳게 갈랐다는 **독립적인** 근거는 이것뿐이다.
- 그 밖에 배치 상수, 세대 판정, 6D 패킹, 서브프레임/클라우드 관계, 접촉 연속성, 출력 경로 충돌.
- `MOVIN_STUDIO_ROOT` 없이는 픽스처가 없으므로 대부분 skip된다.

## 이 머신에 있는 데이터의 규모

`inventory/studio-mock-20260905.txt` (전체 표), `.json` (같은 내용).

**리포에 커밋된 mock 전부 합쳐 3.0분.** 그중 현재 세대이면서 포인트클라우드가 있는 것은
`DH/` 9.05초(182 클라우드, 316k 점) 하나뿐이고, `v3/`가 옛 세대로 6.00초(121 클라우드, 144k 점)다.
나머지는 모션만 있거나 레거시 22본이다.

**Stage 1·2를 돌리기엔 턱없이 부족하다.** 실제 녹화(`recordings/<clip>/`)는 Studio 머신에 있고
이 Mac에는 없다. 다음에 Windows 쪽에서 같은 인벤토리를 돌려 실제 규모를 확정해야 한다 —
로드맵대로 **데이터 규모가 모델 선택을 정하기** 때문이다.

## 남은 Stage 0 항목

- [ ] Windows(Studio 머신)에서 실제 `recordings/` 인벤토리 → 클립 수·분량·인원 확정
- [ ] AWS 체크리스트 (크레딧 잔액·만기·적용 범위, IAM, G/VT·P 쿼터, 리전, Budgets, DLAMI 런북)
