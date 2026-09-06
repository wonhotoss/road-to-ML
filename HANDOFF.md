# 인수인계 — road-to-ML

이 문서는 **이 저장소를 처음 여는 사람**(사람이든 에이전트든)을 위한 것이다.
어디까지 왔고, 무엇을 지켜야 하고, 다음에 무엇을 할지를 적는다.
로드맵·판단 근거·지출 계획은 [README.md](README.md), 단계별 상세는 `stage-N-*/README.md` 에 있다.

## 한 문단 요약

MOVIN Studio 개발자(C#/Unity, ML 실무 0)가 남은 AWS 크레딧(≈₩1.4억, **만기 2027-03**)으로
ML을 단계별로 배워 실무 성과까지 가는 프로젝트다. 9~10월은 로컬 6 GB로 배우는 램프(Stage 0~2,
크레딧 거의 안 씀), 11월~2월은 크레딧 없이는 불가능한 규모의 본 과제(Stage 3: **전방향 정합
포인트클라우드 → 휴먼 모션**, LiDAR-mimic 합성 데이터 엔진 + 멀티 GPU 학습)다.
교과서 데이터셋 대신 Studio 녹화 클립을 쓰고, 매 단계의 산출물은 **기존 방법 대비 측정 비교**다.
지금은 **Stage 0 절반 완료, Stage 1 1주차 진행 중**이다.

## 이력

- 2026-08-30 `wonhotoss/scratch` 인큐베이터의 `hello-ML/` 로 착수. D:\projects 자산 조사, 6단계 로드맵.
  같은 날 크레딧 규모·만기와 회사의 본 과제를 반영해 9월~2월 달력에 고정, 지출 계획(~$75k + 버퍼) 추가.
- 2026-09-05 **Stage 0 변환기 + 인벤토리 완료**(Mac에서). 클립 `*.packets.bin` → Parquet + `.npy`,
  Studio 자체 디코더와 값 일치를 35개 테스트로 고정. 규격에서 읽어낸 아홉 가지는
  [stage-0-setup/README.md](stage-0-setup/README.md). 같은 날 Stage 1 1주차 브리프를 쓰고
  **학습 모드 계약**(아래 규칙 1)을 확정.
- 2026-09-06 hello-ML → **road-to-ML** 로 이름을 바꾸며 상설 저장소로 분리 — **이 저장소가 그것이다.**
  그 전 이력은 `scratch` 저장소의 `hello-ML/` 경로에 남아 있다(마지막 커밋 `beb2dff`).
  `stage-1-basics/note.md` 는 scratch 에 커밋된 적 없이 작업 중 상태로 여기로 왔다.

## 지금 어디까지 왔나

| 단계 | 상태 |
|---|---|
| 0 발판 | 변환기·인벤토리·테스트 **완료**. 남은 것 둘 — Windows 에서 실제 `recordings/` 인벤토리, AWS 체크리스트 |
| 1 기초 | **1주차 진행 중.** Karpathy *Zero to Hero* 1편(micrograd)을 보며 쓴 노트가 [stage-1-basics/note.md](stage-1-basics/note.md). 코드 파일은 아직 0개, 과제 1a·1b 미착수. `pyproject.toml`·`uv.lock` 만 있다 |
| 2~5 | 미착수 |

note.md 는 사용자의 시청 노트다. 끝에 "만들어볼 것" 넷이 있다 — 임의 value graph 생성기 + 델타 검증
테스트 수트, `__pow__` 가 상수 지수만 받는 이유, 그래프 축약, 임의 그래프를 임의 신경망으로 맞추는 루프.
이것이 과제 1a 의 자연스러운 확장이고, 사용자가 스스로 세운 목표이므로 **브리프보다 이쪽을 우선한다.**

## 5분 만에 돌려 보기

단계 폴더는 각각 **독립된 uv 프로젝트**다(`pyproject.toml` + `uv.lock`). `.venv` 는 커밋되지 않으니
처음 여는 머신에서 `uv sync` 부터.

```sh
# Stage 0 — 변환기
cd stage-0-setup
uv sync
uv run movin-clip-inventory <녹화_디렉터리> --json inventory.json   # 뭐가 얼마나 있나
uv run movin-clip-convert <clip.bin> [...] --out <출력_루트>         # 변환
MOVIN_STUDIO_ROOT=<studio-repo> uv run pytest tests -q               # 픽스처는 Studio 리포의 mock/*.bin

# Stage 1 — 사용자가 코드를 쓰는 곳
cd stage-1-basics
uv sync                                                              # torch·numpy·matplotlib·sklearn·jupyter
```

> **파이썬 주의.** 이 PC의 Git Bash `python` 은 `C:\Python33`(3.3.0) 스텁으로 잡힌다.
> uv 가 관리하는 Python(3.12+)만 쓴다 — `uv run`, `uv python install 3.12`.

### 머신이 두 대다

| | Windows (`D:\projects`) | Mac (`~/Documents`) |
|---|---|---|
| GPU | RTX 2060 6 GB — **CUDA 학습은 여기** | 없음(MPS) |
| 데이터 | Studio 실제 녹화 `recordings/`, `LiDAR-mimic` — **여기에만** | 커밋된 mock 3.0분만 |
| 역할 | 학습, 인벤토리, 합성 데이터 | 코드·데이터 파이프 작업(Stage 0 이 여기서 됨) |

둘 사이는 **git 원격으로만** 동기화한다. 그래서 산출물은 OS 중립으로 쓰고 머신 의존 경로는
환경변수(`MOVIN_STUDIO_ROOT` 등)로 받는다.

## 이 프로젝트의 규칙

1. **Stage 1 의 코드는 전부 사용자가 쓴다.** 에이전트는 개념·과제·판정 기준·리뷰만 맡고,
   막히면 답이 아니라 질문으로 되돌린다. 남이 짜 준 코드로는 배울 수 없다는 것이 이 단계의 존재 이유다.
   (Stage 0 처럼 발판 성격의 코드는 예외였다.)
2. **산출물은 "기존 방법 대비 측정 가능한 비교"다.** 베이스라인(임계값, Catmull-Rom, 이너셜라이저,
   단일 시점 백엔드)을 먼저 정하고 같은 지표로 이겼다/졌다를 판정한다.
3. **라벨 출처를 항상 의심한다.** 스트림의 발 접촉과 55본 모션은 백엔드 모델의 *추정*이다
   (접촉값은 `[0,1]` 을 벗어나는 연속값 — Stage 0 이 확인했다). 그것을 정답 삼아 학습하면 증류지 개선이 아니다.
   학습 라벨과 평가 라벨을 분리한다.
4. **판정은 참조 구현과 숫자 대조로 한다.** micrograd 는 PyTorch `.grad` 와 `1e-6`, 디코더는 Studio
   `bvh_parity.py` 와 `1e-5`/`1e-6`. 눈으로 봐서 그럴듯한 것과 맞는 것은 다르다.
5. **변환은 와이어 그대로.** 좌표계·회전 표현을 바꾸지 않고 규약은 `meta.json` 에 적는다.
   6D 회전은 `movin_clip.motion.basis` 를 거쳐서만 읽는다(두 계열의 본당 배치가 다르다).
6. **크래시 얼리 — 단, ML 의 함정은 조용히 틀린다.** 브로드캐스팅, 기울기 누적 누락, 재사용 노드의
   `=` 덮어쓰기는 에러를 내지 않는다. 텐서를 만들거나 받을 때마다 shape 을 단언한다. 방어코드가 아니라 계약 명시다.
7. **인프라는 EC2 + DLAMI + SSH, 단일 노드 8-GPU(p4d)가 상한.** SageMaker Studio·멀티노드 금지.
   크레딧은 11~2월 본 과제에 집중하고 3월(만기 달)은 계획에 넣지 않는다.
8. **실제 녹화 데이터는 커밋하지 않는다.** 사용자 이미지·포인트클라우드가 들어 있다. 서울 리전 S3,
   비공개·암호화. 변환 출력(`out/`, `*.parquet`, `*.npy`)은 `.gitignore` 에 있다.
9. **완료한 것은 README 의 진행 기록에 날짜와 결론을 남긴다.** 단계 코드는 `stage-N-*/`.
   실무 과제가 들어오면 해당 단계의 "적용 문제"를 그것으로 교체한다.

## 지금 상태에서 알아야 할 것

- **원격이 없다.** 분리 직후라 아직 GitHub 저장소를 만들지 않았다. Mac 의 `scratch` 사본을 pull 하면
  `hello-ML/` 이 지워지므로, **그 전에** 이 저장소를 푸시하고 Mac 에서 클론한다. Mac 의 `hello-ML/` 에
  커밋 안 된 변경이 있다면 그것부터 여기로 옮긴다.
- **데이터 규모가 미확정이다.** Mac 에는 커밋된 mock 3.0분(현 세대 + 클라우드는 `DH/` 9초 하나)뿐이다.
  Stage 1 2주차와 Stage 2 는 실제 `recordings/` 가 필요하고, 그 규모가 모델 선택을 정한다.
- **`ObjsFrame` 와이어 스키마는 두 세대가 공존한다**(`e0f486e8` 전후). 틀린 세대로 읽으면 크래시 없이
  모션만 사라진다. 변환기는 payload 길이로 판정한다. Studio 가 또 바꾸면 `schema.objs_frame_variants` 에
  추가하고 `tools/refresh_objs_frame_descriptor.py` 를 돌린다.
- **센서 20 Hz, 모션 60 Hz(서브프레임 3).** 포인트클라우드는 서브프레임 셋에 바이트 동일. 이것이
  Stage 2 업샘플러의 입출력 규격이다.
- **Stage 1 2~3주차는 설계 변경이 예고돼 있다.** 로드맵의 "발 접촉 분류"는 라벨이 연속값이라 그대로
  갈 수 없다. 1주차 결과(특히 막혔던 지점 목록)를 보고 확정한다 — [stage-1-basics/README.md](stage-1-basics/README.md).
- AWS 계정 `156041436196`(회사). 현재 CLI 자격증명 `lightsail-admin` 은 EC2 권한이 없다. 사용자가
  관리자이므로 IAM 유저·G/VT 쿼터는 계획 성숙 시 직접 준비한다. **P 쿼터(L-417A185B)는 11월 초에 미리.**

## 다음에 할 일 (우선순위)

1. **원격 저장소를 만들어 붙이고 푸시, Mac 에서 클론.** 위 "원격이 없다" 항목의 순서를 지킨다.
2. **Stage 1 과제 1a → 1b** (사용자 코드). 1a = `Value` autograd 엔진 + PyTorch 대조 `1e-6`,
   1b = `Neuron → Layer → MLP` + 손으로 도는 학습 루프. note.md 의 "만들어볼 것" 목록을 1a 에 합친다.
   1주차 끝에 가져올 것 셋: micrograd 구현과 대조 결과, MLP loss 곡선, **막혔던 지점 목록**.
3. **Windows 에서 `recordings/` 인벤토리** — 클립 수·분량·인원 확정. 2주차 전에 끝나야 한다.
   ```sh
   cd stage-0-setup && uv run movin-clip-inventory <recordings_경로> --json inventory/recordings-<날짜>.json
   ```
4. **AWS 체크리스트**(README Stage 0 절) — 크레딧 잔액·만기·적용 범위 확인이 첫 항목.
5. **2~3주차 설계 확정** — 학습 라벨/평가 라벨 분리, 소수 클립 수작업·물리규칙 정답.

손대지 않은 것: `stage-0-setup/aws/` 런북(로드맵엔 있으나 폴더가 없다), Stage 2 이후 폴더.

## 저장소

원격은 **아직 없다.** 만들고 푸시하는 것은 저장소 주인이 한다.
`.gitignore` 는 파이썬 부산물, 변환 출력(개인정보 포함 가능), Claude Code 로컬 설정을 뺀다.
`uv.lock` 과 `inventory/*.json|txt` 는 커밋한다.
