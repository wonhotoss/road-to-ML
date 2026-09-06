# road-to-ML

> 2026-09-06까지 `wonhotoss/scratch` 인큐베이터의 `hello-ML/` 였다. 그 전 이력은 그 저장소의 같은 경로에 있다(마지막 커밋 `beb2dff`). 처음 여는 사람은 [HANDOFF.md](HANDOFF.md)부터.

머신러닝을 **단계별로 학습하고, 개념을 몸에 익히고, 실무에서 쓸 수 있는 성과까지** 가는 프로젝트.
남은 AWS 크레딧(약 ₩1.4억, **만기 2027-03**)을 GPU 임대와 학습 비용으로 쓴다. 2026-08-30 착수.

이 문서는 로드맵과 판단 근거다. 단계별 실습 코드·노트·결과는 `stage-N-*/` 하위 폴더에 쌓고,
맨 아래 [진행 기록](#진행-기록)에 날짜와 결론만 남긴다.

---

## 0. 출발점 — 2026-08-30 조사 결과

### 나(개발자)

- 주력 C#/Unity 6. `MOVIN-Studio-App` 1087 커밋의 1위 기여자. 약 40k 라인의 `Assets/Scripts`.
- Python은 언어로서 충분히 쓴다. 단 리포의 Python 자산은 내 것이 아니다 — `tools/motion_parity/`(4.2k 라인)는 다른 사람이 썼고 목적·내용을 모른다, `text_to_mock/`은 내가 구상하고 Claude Code가 구현했다. **NumPy·PyTorch·pandas·노트북·학습 루프 경험은 0.** `requirements.txt`엔 `protobuf`, `pyzmq`만 있다.
- ML은 Google Crash Course 수준의 개념만 있음. 실무 경험 없음.
- 일하는 방식: 1차 자료(논문·규격·프로토콜 덤프)를 먼저 읽고 문서를 진실의 원본으로 삼아 코드를 맞춘다(`frame-based-space-mapping`, `foreign-force`). 크래시 얼리·방어코드 금지·선언적 스타일의 확고한 코딩 원칙.
- AWS 계정 관리자다. IAM·쿼터는 계획이 성숙하면 직접 준비한다.

### 회사의 관심사 — 본 과제

**전방향(360°) 정합된 인체 포인트클라우드로부터 휴먼 모션 추론.** 현재 제품(Tracin)은 LiDAR 1대 + RGB 카메라의 단일 시점이다. 여러 센서를 정합해 사람을 전방향으로 덮는 포인트클라우드가 입력이 되면 가림은 줄지만, 입력 분포·밀도·정합 오차가 전혀 다른 새 문제가 된다. 이 프로젝트의 **3단계(11월~2월)를 이 문제에 배정**한다. 크레딧 규모(≈$100k)는 이 규모의 연구를 실제로 돌릴 수 있는 크기이고, 만기가 3월이므로 그 전에 써야 한다.

### 실무 자산 — 이미 가진 것

MOVIN Studio는 **블랙박스 인식 스택의 소비자**다. ML(YOLO 세그, ViTPose 2D, LiDAR→3D 모션 회귀, 20→60 Hz 업샘플러, 손가락 후처리)은 전부 Jetson 위 `MOVIN-Backend`(연구팀: Dongseok Yang, DK Jang — 논문 *MOVIN*(PG 2023), *ELMO*(SIGGRAPH Asia 2024)의 저자들)에 있고 Studio는 와이어 바이트만 본다. 그 대신 Studio에는 **고전적·휴리스틱 후처리 전부**가 있고, 그것이 ML로 바꿀 수 있는 표면이다.

| 자산 | 위치 (`studio-app-develop/` 기준) | ML 관점 |
|---|---|---|
| **녹화 클립 원본** — 프레임마다 55본 모션(위치·6D 회전·선속도·각속도·발접촉 L/R·오프셋) 60 Hz + 원시 LiDAR 포인트클라우드 + YOLO bbox·마스크 + ViTPose COCO-17 2D 키포인트(신뢰도) + 2D 손 키포인트 + 센서/발행 타임스탬프 | `recordings/<clip>/ObjsFrame.packets.bin`, 컨테이너 `Utility.cs:985-1090`, 디코더 `tools/motion_parity/bvh_parity.py:449-503`, 커밋된 샘플 `MOVIN_Studio_Unity/mock/*.bin` | **시간 정렬된 멀티모달 데이터셋이 이미 디스크에 있다.** 재직렬화 없이 와이어 그대로라 손실 없음 |
| 글러브 참조 데이터(Rokoko/Manus/Quest)가 같은 클립에 동기 | `scene/AppData.cs:57-70` `ClipData` | LiDAR 단독 손 추정에 대한 **쌍(pair) 감독 신호** |
| 사용자 체형 보정 슬라이더 로그(`leg_stretch`, `leg_tilt`, `torso_tilt`, `head_tilt`, `user_body_size`) | `logv2_spec.md` §3.14, `Animation/ActorOffset/` | 사용자가 매 세션 손으로 고치는 값 = "장치 추정을 어떻게 고칠지"의 라벨 |
| **발 접촉 락/언락 휴리스틱 — 자기 주석으로 "미해결"** | `Animation/Retargeting/ChainRetargetIK.cs:1242-1245, 1271-1276`, 임계값 `GroundContactDetector.cs:63-64` | 지금 존재하는 발 슬라이딩·지터 버그. 첫 ML 타깃 |
| 스무딩은 Holden 이너셜라이저 하나 | `Retargeting/Spring.cs` | One-Euro·칼만·학습 디노이저 전무 |
| 프레임 보간 Catmull-Rom | `pipeline.cs:16-42` | 백엔드의 학습 업샘플러 옆에 있는 순진한 리샘플링 |
| 본맵 자동 탐지(이름 휴리스틱) | `Animation/BoneMapDetector.cs` | 분류 문제 |
| **삭제된 LibTorch SMPL 피팅 플러그인** — Chamfer(`verts2pcd/pcd2verts`)·posePrior·betaReg·footContact(가중 100)·height 손실 + Adam 스케줄, 33관절/595정점/10블렌드셰이프 바디모델 | `git show 1cb607d5^:MOVIN_Studio_Unity/Assets/MOVINCalibration/…`, `…/StreamingAssets/calibration/hyperparameters.json` | **우리 팀의 최적화 기반 체형 피팅.** 학습 회귀기의 고전적 전신. 본 과제에서 밀집 포인트클라우드에 대한 피팅 베이스라인·라벨 생성기로 부활 |
| **LiDAR-mimic** — 임의 3D 씬을 센서 시점에서 래스터화해 LiDAR 스캔 패턴대로 포인트클라우드를 GPU에서 생성. 좌표 readback 지원 | `D:\projects\LiDAR-mimic` (Unity, Linux 빌드 가능) | **본 과제의 합성 데이터 엔진.** SMPL-X 바디에 AMASS 모션을 입혀 N개 시점에서 스캔하면 전방향 정합 포인트클라우드 + 정답 관절이 무한히 나온다 |
| 지연 계측 필드 미사용 | `backend/proto/ObjsFrame.cs:145-165` `timestamp_sensor_ns`/`timestamp_pub_ns` | 공짜 메트릭 |
| 웹: Athena 텔레메트리 레이크(로그인·라이선스·구독·기능별 툴팁 클릭·free clip 초) + 구독 만료·전환 라벨, Bedrock Claude로 text-to-SQL 운영 중 | `MOVIN-Web/api/movin/log_query_*.py`, `log-design-20260427.csv` | 표형 ML(이탈·전환 예측)의 데이터와 라벨이 이미 있음 |

Studio에 ONNX/Sentis/Barracuda 등 추론 런타임은 없다(`Packages/manifest.json` 확인).

### 관심영역 (사이드 프로젝트가 말해주는 것)

- `foreign-force`: 모캡 → 비인간형 소형 로봇(Microduck 2026-12 배송) 의도 전이. 논문 14편 정리. **RL 정책 조건화·MuJoCo 시뮬**이 다음 단계.
- `my-talking-claw`, `super-clova`: 자체 STT·LLM·TTS 음성 스택, 임베디드.
- `frame-based-space-mapping`, `LiDAR-mimic`: 지오메트리·렌더링.

### 인프라

- 로컬은 **두 대**다.
  - **Windows**(`D:\projects`) — RTX 2060 **6 GB**, RAM 16 GB, Ryzen 3 3300X(4코어). uv 0.9, Docker 28, AWS CLI 2.36. `python --version`이 3.3.0으로 나오는 스텁이 잡히므로 uv가 관리하는 Python을 쓸 것. **CUDA 학습은 여기서.** Studio 실제 녹화(`recordings/`)와 `LiDAR-mimic`도 여기에만 있다.
  - **Mac**(`~/Documents`, Apple Silicon) — uv 0.11, Docker 28, AWS CLI 2.34. CUDA 없음(MPS). `MOVIN-Studio-App`·`MOVIN-Web` 사본이 있어 **코드·데이터 파이프 작업은 여기서** 한다.
  - 그래서 산출물은 OS 중립으로 쓰고, 머신 의존 경로는 환경변수(`MOVIN_STUDIO_ROOT` 등)로 받는다.
- AWS: 계정 `156041436196`(회사 계정 — MOVIN-Web Lightsail이 여기). **크레딧 잔액 약 ₩1.4억(≈$100k, 환율 1,400원 가정), 만기 2027-03.** 현재 CLI 자격증명 `lightsail-admin`은 EC2 권한이 없으나, 사용자가 관리자이므로 계획 성숙 시 IAM 유저와 **"Running On-Demand G and VT instances"** 쿼터를 준비하기로 함(→ Stage 0 체크리스트).

---

## 1. 설계 원칙 — 의견

1. **교과서 데이터셋 단계는 최소화하고, 2주차부터 자기 데이터를 쓴다.** 기초 개념(텐서·autograd·학습 루프·과적합)은 작은 장난감으로 1~2주면 충분하다. 진짜 격차는 수학이 아니라 **데이터 엔지니어링·평가 설계·배포**이고, 그건 자기 데이터로만 배울 수 있다.
2. **매 단계의 산출물은 "기존 방법 대비 측정 가능한 비교"다.** 정확도 숫자 하나가 아니라, 지금 코드(임계값, Catmull-Rom, 이너셜라이저, 단일 시점 백엔드)를 베이스라인으로 두고 같은 지표로 이긴다/진다를 판정한다.
3. **실무 성과는 두 층이다.** 빠른 층 = Studio 안에서 돌아가는 ONNX 모델 하나(10월). 큰 층 = 본 과제(전방향 포인트클라우드→모션)의 **데이터 엔진 + 벤치마크 + 학습된 모델 + 보고서**(2월). 큰 층은 연구팀에 넘길 수 있는 형태로 만든다.
4. **라벨 출처를 항상 의심한다.** 스트림의 발 접촉(825-826)과 55본 모션은 백엔드 모델의 *추정*이다. 그것을 정답 삼아 학습하면 지식 증류지 개선이 아니다. 개선하려면 수작업 라벨·물리 규칙·합성 데이터처럼 **독립된 정답**이 필요하다. 본 과제에서 합성 데이터가 중요한 이유가 이것이다.
5. **크레딧은 이번엔 계획의 변수다 — 만기가 있기 때문이다.** ≈$100k를 6.5개월 안에 써야 하고, 안 쓰면 소멸한다. 그래서 9~10월은 **로컬 6 GB로 빠르게 배우는 램프**(크레딧 거의 안 씀), 11월~2월은 **크레딧 없이는 불가능한 규모의 연구**(합성 데이터 수백만 프레임, 멀티 GPU 학습, 어블레이션)에 쓴다. 3월은 만기 달이므로 사용량을 계획에 넣지 않는다(만기일과 "만기 달 사용분 적용 여부"는 Billing → Credits에서 확인).
6. **크레딧이 크다고 복잡한 인프라를 사면 안 된다.** SageMaker Studio·멀티노드 클러스터는 배움을 가리고 시간을 먹는다. **EC2 + Deep Learning AMI + SSH, 단일 노드 8-GPU(p4d)까지가 상한.** 그 안에서 DDP·혼합정밀도·체크포인트·spot 재개를 익히는 것이 실무 스킬이다. 마켓플레이스 AMI는 크레딧 적용이 안 되는 경우가 많으니 AWS 자체 DLAMI를 쓴다.
7. **연구팀을 초기에 끌어들인다.** 본 과제는 그들의 논문(MOVIN, ELMO) 위에 서야 한다. 3단계 첫 주에 (a) 다중 센서 정합 데이터 보유 여부, (b) 학습 코드·바디모델(G4_ViconXsmplx) 공유 가능 여부, (c) 그들이 원하는 평가 지표를 묻는다. 내 역할은 경쟁이 아니라 **데이터 엔진과 컴퓨트를 가진 협업자**다.

---

## 2. 로드맵 — 2026-09 ~ 2027-02

기간은 주 8~10시간 기준(본 과제 기간은 그 이상 필요할 수 있음). 실무 과제가 들어오면(사용자가 중간중간 준다고 했다) 해당 단계의 "적용 문제"를 그것으로 교체한다.

| 단계 | 기간 | 주제 | 산출물 | 크레딧 |
|---|---|---|---|---|
| 0 | 9월 1주 | 환경·데이터 파이프·AWS 발판 | 클립→Parquet 변환기, GPU 박스 런북, 예산 가드 | ~$0 |
| 1 | 9월 2~3주 | 기초 — 학습 루프를 손으로, 자기 데이터로 | 발 접촉 분류기 vs 임계값 휴리스틱 | ~$0 |
| 2 | 9월 4주~10월 | 시계열·모션 모델 | 20→60 Hz 학습 업샘플러 → **ONNX → Unity Sentis, Studio 안에서 실행** | ~$1k |
| **3** | **11월~2월** | **본 과제: 전방향 정합 포인트클라우드→모션** | 합성 데이터 엔진, 벤치마크, 베이스라인→스케일업 모델, 보고서 | **~$70k** |
| 4 | 12월~2월 병행 | 강화학습·로봇 (foreign-force) | MuJoCo 모캡 의도 조건화 정책, Microduck 실기 | ~$5k |
| 5 | 여유 시 | 음성·LLM 미세조정, 표형 이탈 예측 | Whisper 한국어 / LoRA / LightGBM 이탈 모델 | ~$2k |

### Stage 0 — 발판 (9월 1주)

**목표:** 이후 단계에서 환경 문제로 멈추지 않게 한다.

- Python: `uv init`, `uv python install 3.12`, PyTorch CUDA(로컬 2060은 CUDA 12.x OK), numpy·pandas·polars·pyarrow·matplotlib·jupyter. 프로젝트별 `uv` 가상환경.
- **데이터 파이프:** `ObjsFrame.packets.bin` → 프레임 단위 Parquet(모션 989 float 분해, 접촉, 타임스탬프, 2D 키포인트) + 포인트클라우드는 프레임별 `.npy`. 디코더는 `bvh_parity.py:449-503`, 컨테이너는 `Utility.cs:985-1090`에서 옮긴다. `mock/*.bin`으로 먼저 검증. 클립이 몇 개·몇 분·몇 명인지 인벤토리를 만든다 — **데이터 규모가 이후 모델 선택을 결정한다.**
- **AWS 체크리스트(관리자 = 나):**
  - [ ] Billing → Credits: 정확한 잔액·만기일, 적용 서비스 범위(EC2·S3 외 SageMaker·Bedrock 포함 여부), 만기 달 사용분 적용 여부 기록
  - [ ] IAM: ML 작업용 유저/역할(EC2·EBS·S3·CloudWatch·Budgets·Service Quotas). 인스턴스에는 S3 접근용 인스턴스 프로파일
  - [ ] Service Quotas — **G/VT On-Demand**(L-DB2E81BA) 96 vCPU 이상, **P On-Demand**(L-417A185B) 96 vCPU 이상(p4d.24xlarge 1대), **G/VT Spot**(L-3819A6DF). 승인 1~3일, P는 더 걸릴 수 있음 → 3단계 전 11월 초에 미리
  - [ ] 리전 결정: 서울(`ap-northeast-2`)에 g5·g6·g6e·p4d 있음. 단가·p4d/p5 가용성은 `us-east-1`/`us-west-2`가 유리. **합성 데이터는 어디든 무방, 실제 사용자 클립(이미지 포함)은 서울 유지**를 기본으로
  - [ ] AWS Budgets 월 한도 + 50/80/100% 알림, Cost Anomaly Detection
  - [ ] 런북: DLAMI(PyTorch) 기반 기동/정지 스크립트, 데이터 EBS 볼륨 분리 유지, `aws s3 sync`, 유휴 자동 정지(CloudWatch `CPUUtilization < 5% for 30 min` → stop)
- 산출물: [`stage-0-setup/`](stage-0-setup/) (변환기, 인벤토리 노트, `aws/` 런북·체크리스트).
  **변환기·인벤토리는 완료(2026-09-05).** 규격에서 읽어낸 것과 남은 항목은 [`stage-0-setup/README.md`](stage-0-setup/README.md)에.

### Stage 1 — 기초: 학습 루프를 손으로, 자기 데이터로 (9월 2~3주)

**배울 개념:** 텐서·브로드캐스팅, autograd, 손실·옵티마이저·학습률, 배치·에폭, train/val/test 분할과 **누출(leakage)**, 과적합/정규화, 분류 지표(정밀도·재현율·PR 곡선·캘리브레이션).

- 1주: Karpathy *Zero to Hero* 1~3편(micrograd → makemore)을 따라 **자동미분과 MLP를 손으로 구현**. 이 한 번의 "밑바닥"이 이후 모든 프레임워크 코드를 읽히게 한다.
- 2주: **적용 — 발 접촉 분류.** 입력 = 발·발끝·힙 위치/속도 창(window), 출력 = L/R 접촉. 라벨은 원칙 4에 따라 두 벌: (i) 스트림 825-826(백엔드 추정), (ii) 소수 클립 수작업/물리규칙 정답. 베이스라인 = `GroundContactDetector.cs` 임계값(0.25 m / 0.1 m/s). 같은 정답셋에서 PR 곡선·**전환 지연·플리커 횟수**로 비교. 로지스틱 회귀 → MLP → 작은 1D-CNN 순으로 올리며 언제부터 이득이 없는지 본다.
- 산출물: `stage-1-basics/` — micrograd 노트, 접촉 분류기 비교 리포트. **판정:** 임계값을 이겼는가, 어떤 지표에서, 어떤 라벨셋에서.

### Stage 2 — 시계열·모션 모델 → Studio 안으로 (9월 4주~10월)

**배울 개념:** 시퀀스 모델(1D-CNN, GRU, 소형 Transformer), 자기지도 학습, 데이터 증강, 회귀 손실 설계(위치 vs 회전 6D vs 속도), ONNX 내보내기와 런타임 수치 일치 검증, 지연·처리량 측정, 첫 AWS 스윕.

- **적용 A — 학습 업샘플러(자기지도).** 60 Hz 클립을 20 Hz로 다운샘플해 입력, 원본 60 Hz를 정답으로. 라벨 비용 0. 베이스라인 = `pipeline.cs` Catmull-Rom. 지표 = 관절 위치 오차, 각속도 오차, 접촉 프레임의 발 슬라이딩. 백엔드 ELMO가 하는 일의 미니판이라, 연구팀 논문을 읽을 어휘가 생긴다.
- **적용 B — 디노이저/후처리.** 깨끗한 클립에 합성 노이즈·드롭아웃을 넣고 복원. 또는 A에 접촉 헤드를 붙여 Stage 1과 합친다.
- **배포:** ONNX → Python `onnxruntime`으로 PyTorch와 수치 일치 확인 → **Unity Sentis**로 Studio 리타게팅 앞단에 삽입, 프레임당 추론 시간 측정. 6D 회전·좌표계 플립(`Utility.cs:307-321`)·서브프레임 규약을 그대로 맞춰야 하는데, 이게 실무의 배포 문제다.
- **AWS 첫 사용:** 하이퍼파라미터 스윕(창 길이·모델 크기·손실 가중)을 `g6.xlarge` spot으로. 런북이 실전에서 동작하는지 여기서 검증한다 — 3단계 진입 전 리허설.
- 산출물: `stage-2-motion/` + Studio 브랜치. **판정:** Studio 안에서 ONNX 모델이 실시간으로 돌고, Catmull-Rom 대비 지표가 좋아졌는가.

### Stage 3 — 본 과제: 전방향 정합 포인트클라우드 → 휴먼 모션 (11월~2월)

**배울 개념:** 포인트클라우드 백본(PointNet++, Point Transformer v3), 파라메트릭 바디모델(SMPL-X: β·θ·LBS), 최적화 피팅 vs 학습 회귀, Chamfer·사전(prior) 손실, 시간적 모델(프레임열→모션), **합성 데이터와 도메인 랜더마이제이션·sim-to-real 격차**, 멀티 GPU DDP·혼합정밀도·체크포인트·spot 재개, 어블레이션 설계, TensorRT/Jetson 배포 가능성 평가.

**문제 정의(초안, 3a에서 확정):** 입력 = 사람 한 명을 덮는 정합된 포인트클라우드 시퀀스(N 시점 융합, 정합 오차 포함, 20 Hz). 출력 = 55본(또는 SMPL-X θ·β) 모션 60 Hz + 발 접촉. 지표 = MPJPE·PA-MPJPE·PVE, 지터(가속도), 발 슬라이딩, 접촉 정확도, 지연. 베이스라인 = (i) 단일 시점 백엔드 출력, (ii) 복원한 SMPL 피팅(최적화, 느리지만 정확 — 상한선 겸 라벨 생성기).

- **3a. 정의·문헌·협업 (11월 1~2주).** *MOVIN*(PG 2023), *ELMO*(SA 2024)를 먼저 읽고 연구팀에 원칙 7의 세 질문. 외부: *LiDARCap*(CVPR 2022, LiDARHuman26M), *LiDAR-HMR*, *LiveHPS/LiveHPS++*(CVPR/ECCV 2024), *SLOPER4D*, *HumanM3*(다시점 다중센서). 대부분 단일 LiDAR 야외라 **전방향 정합 입력의 공개 데이터는 사실상 없다** → 합성이 필수라는 근거. 평가 프로토콜·벤치마크 셋(보류 실제 클립 + 합성 테스트셋) 확정.
- **3b. 데이터 엔진 (11월).** LiDAR-mimic 기반 합성기: AMASS 모션 → SMPL-X 메시 → Tracin 스캔 패턴을 흉내낸 N개 시점 래스터화 → 융합 포인트클라우드 + 정답(관절·θ·β·접촉). **랜더마이제이션이 핵심**: 체형, 의복 두께(메시 오프셋), 센서 노이즈·드롭아웃, 시점 수·배치, **정합 오차(회전·이동 미세 오차, 시점 간 시간 오프셋)**, 배경 잔여점. Unity Linux 헤드리스 빌드를 `g5.xlarge`/`g6.xlarge` 여러 대에서 병렬 실행, S3에 샤드로 저장(수백만 프레임, 수 TB). 실제 단일 시점 클립은 도메인 적응 타깃·실측 평가용.
- **3c. 베이스라인 (12월).** PointNet++ 프레임별 회귀(관절) → SMPL-X 파라미터 회귀 → 시간 모델(프레임열 → 모션, GRU/Transformer, ELMO식 업샘플 포함). `g6e.12xlarge`(L40S×4)에서 학습. 복원 SMPL 피팅과 비교해 회귀기의 격차를 수치화. 여기서 **합성→실제 격차**가 처음 보인다.
- **3d. 스케일업·어블레이션 (1월~2월 중순).** Point Transformer v3급 백본, 합성 데이터 전량, `p4d.24xlarge`(A100×8) DDP. 어블레이션: 시점 수(1·2·4·8), 정합 오차 크기, 점 밀도, 시간 창 길이, 합성/실제 혼합 비율. 최종 모델 ONNX/TensorRT 변환 후 Jetson급 지연 추정. Studio에 오프라인 재생 데모(정합 클라우드 → 모션).
- **3e. 보고서·인계 (2월 말).** 데이터 엔진(재현 가능), 벤치마크, 모델·가중치, 어블레이션 표, 실패 사례 분석을 연구팀에 인계.
- 산출물: `stage-3-omni-pc2motion/`. **판정:** 합성 벤치마크에서 단일 시점 베이스라인을 이기는가, 실제 클립에서 격차가 얼마인가, 연구팀이 데이터 엔진을 이어 쓰는가.

### Stage 4 — 강화학습·로봇: foreign-force와 병행 (12월~2월)

- Spinning Up + CleanRL PPO로 CartPole → MuJoCo Humanoid 한 번. `microduck_rl` MJCF로 시뮬 환경, foreign-force의 "의도 특징"을 조건으로 받는 정책. MJX GPU 병렬은 `g6.xlarge`~`g6e.xlarge`.
- 3단계와 컴퓨트를 경쟁하지 않도록 **별도 인스턴스·예산 $5k 상한.** Microduck 실기(12월 배송) 연결은 foreign-force 쪽에서.

### Stage 5 — 여유 시

- Whisper 한국어 미세조정·웨이크워드(super-clova/talking-claw), 소형 LLM LoRA. Athena 이탈/전환 예측(LightGBM, GPU 불필요 — 실무 표형 ML 절차 학습용).

---

## 3. AWS 사용 계획 — 크레딧 ≈$100k, 2027-02까지

### 지출 계획

| 항목 | 시기 | 인스턴스 | 산정 | 금액 |
|---|---|---|---|---|
| 개발 박스 | 9월~2월 | `g6.xlarge` 온디맨드, 하루 4~8h | ~$0.8/h × ~800h | ~$1k |
| Stage 2 스윕 | 10월 | `g6.xlarge` spot | | ~$0.5k |
| 3b 합성 데이터 생성 | 11월 | `g5.xlarge`/`g6.xlarge` × 8~16대 병렬, 수일 | ~$1/h × ~3,000h | ~$3k |
| 3b 저장·전송 | 11월~2월 | S3 수 TB + EBS | | ~$2k |
| 3c 베이스라인 | 12월 | `g6e.12xlarge`(L40S×4, ~$10.5/h) | ~600h | ~$6k |
| 3d 스케일업·어블레이션 | 1월~2월 중순 | `p4d.24xlarge`(A100×8, ~$33/h) | ~1,200h ≈ 50일 상당 | ~$40k |
| 3d 예비(가용성·재실행) | 2월 | p4d 또는 p5 | | ~$15k |
| Stage 4 RL | 12월~2월 | `g6.xlarge`~`g6e.xlarge` | 상한 | ~$5k |
| Stage 5·기타 | | | | ~$2k |
| **계획 합계** | | | | **~$75k** |
| 버퍼(미계획, 2월 마지막 대규모 실행용) | | | | ~$25k |

만기 달(3월) 사용은 계획하지 않는다. 2월 중순까지 큰 실행을 끝내고 남은 것은 2월 말 최종 실행에 쓴다.

### 인스턴스 참고 (2026-08 기준, 온디맨드 미국 리전 근사치 — 콘솔에서 재확인)

| 인스턴스 | GPU | VRAM | 시간당 | 용도 |
|---|---|---|---|---|
| `g6.xlarge` | L4 | 24 GB | ~$0.8 | 개발·소형 학습·스윕·합성 데이터 렌더 |
| `g5.xlarge` | A10G | 24 GB | ~$1.0 | 위 대체·렌더 병렬 |
| `g6e.xlarge` | L40S | 48 GB | ~$1.9 | 큰 배치 단일 GPU |
| `g6e.12xlarge` | L40S×4 | 192 GB | ~$10.5 | 3c 베이스라인, 첫 DDP |
| `p4d.24xlarge` | A100×8 40 GB | 320 GB | ~$33 | 3d 스케일업. 단일 노드 상한 |
| `p5.48xlarge` | H100×8 | 640 GB | ~$55~98 | 가용성 있으면 3d 대체. **Capacity Blocks for ML**(1~14일 예약)로 확보하는 게 보통 — 크레딧 적용 여부 확인 |

- Spot: 60~90% 할인이지만 중단. 합성 렌더·스윕·RL처럼 체크포인트로 재개 가능한 것에만. 대화형 개발과 p4d 장기 학습은 온디맨드.
- p4d/p5는 온디맨드 가용성이 리전·시간대에 따라 없을 수 있다. 3d 시작 2주 전에 리전별로 기동 테스트를 해두고, 안 되면 `g6e.48xlarge`(L40S×8) 또는 Capacity Blocks로 대체.

### 비용 가드

- AWS Budgets 월 한도(계획표 기준 + 20%)와 50/80/100% 알림, Cost Anomaly Detection.
- 유휴 자동 정지, 학습 스크립트 끝 `sudo shutdown -h`, 태그(`project=road-to-ML`, `stage=3c`)로 비용 배분.
- 데이터는 S3, 코드는 git, 환경은 DLAMI + `uv sync` — 인스턴스는 언제 죽여도 되는 상태로.

### 데이터 취급

실제 녹화 클립엔 사용자 이미지·포인트클라우드가 있다. 서울 리전 S3, 비공개·암호화·수명주기. 합성 데이터는 개인정보가 없어 어느 리전이든 무방하며, 그래서 3d를 미국 리전에서 돌릴 수 있다.

---

## 4. 참고 자료

- 기초: Karpathy *Neural Networks: Zero to Hero*; fast.ai *Practical Deep Learning*; *Dive into Deep Learning*(d2l.ai); PyTorch 공식 튜토리얼.
- 모션: Daniel Holden 블로그(이너셜라이제이션 — `Spring.cs`의 원저), *Phase-Functioned Neural Networks*(2017), *Learned Motion Matching*(2020), *Robust Motion In-betweening*(2020).
- 회사 논문: *MOVIN: Real-time Motion Capture using a Single LiDAR*(Jang, Yang 외, PG 2023), *ELMO: Enhanced Real-time LiDAR Motion Capture through Upsampling*(SIGGRAPH Asia 2024).
- LiDAR 휴먼 모션: *LiDARCap*(CVPR 2022), *LiDAR-HMR*(2023), *LiveHPS*(CVPR 2024)·*LiveHPS++*(ECCV 2024), *SLOPER4D*(CVPR 2023), *HumanM3*(2023).
- 3D 일반: *PointNet*/*PointNet++*, *Point Transformer V3*(CVPR 2024), *SMPL*(2015)·*SMPL-X*(2019), *SMPLify*(2016), *AMASS*(2019), *HMR*/*VIBE*.
- RL: OpenAI *Spinning Up*, CleanRL, MuJoCo Playground/MJX, *DeepMimic*(2018), *AMP*(2021). foreign-force `papers/`.
- AWS 가격: [EC2 On-Demand](https://aws.amazon.com/ec2/pricing/on-demand/), [Spot](https://aws.amazon.com/ec2/spot/pricing/), [G6e 서울 출시](https://aws.amazon.com/about-aws/whats-new/2025/03/amazon-ec2-g6e-instances-seoul-region), 정리 글 [Thunder Compute](https://www.thundercompute.com/blog/ec2-gpu-instances)·[CloudPrice g6e.xlarge](https://cloudprice.net/aws/ec2/instances/g6e.xlarge).

---

## 진행 기록

- 2026-08-30 — 착수. D:\projects 조사(Studio·Web·사이드 프로젝트), 로컬/AWS 인프라 점검, 로드맵 작성.
- 2026-09-06 — **scratch에서 분리 → `D:\projects\road-to-ML`(이 저장소). 이름을 hello-ML → road-to-ML로.** Stage 1 1주차가 진행 중인 상태 그대로 옮겼다 — Karpathy 1편(micrograd) 시청 노트 [`stage-1-basics/note.md`](stage-1-basics/note.md)가 있고 코드는 아직 없다. 원격은 붙이지 않았다(사용자가 붙인다). 다음 = **원격·Mac 클론 → 과제 1a**.
- 2026-09-05 — **Stage 0 절반: 클립 → Parquet 변환기 + 인벤토리 완료** ([`stage-0-setup/`](stage-0-setup/)). Studio 자체 디코더(`bvh_parity.py`)와 값이 일치함을 테스트로 고정(35 tests). 규격에서 나온 것 넷: (1) `ObjsFrame` 와이어 스키마가 `e0f486e8`에서 바뀌어 **두 세대가 공존**하고, 틀린 세대로 읽으면 크래시 없이 모션만 조용히 사라진다 → 길이로 판정. (2) 6D 회전의 본당 배치가 현 세대(interleaved)와 레거시 `MocapTopic`(consecutive)에서 **다르다** → `motion.basis` 경유 강제 + 직교성 자체검증. (3) **센서 20 Hz, 모션 60 Hz(서브프레임 3)** 이고 포인트클라우드는 서브프레임 셋에 바이트 동일 = 20 Hz → Stage 2 업샘플러의 입출력 규격이 데이터에 이미 있다. (4) 스트림의 발 접촉은 이진이 아니라 `[0,1]`을 벗어나는 연속값 = 백엔드의 회귀 출력 → 원칙 4를 데이터가 확인해 줬다. 지연 실측 중앙값 64.5 ms. **이 Mac의 데이터는 mock 3.0분뿐**(현 세대 + 클라우드는 `DH/` 9초 하나) → 실제 규모 확정은 Windows에서. 다음 = **실제 `recordings/` 인벤토리 + AWS 체크리스트**.
- 2026-08-30 — 사용자 정보 반영: 크레딧 ≈₩1.4억·만기 2027-03, 사용자가 AWS 관리자(IAM·G 쿼터는 계획 성숙 시 준비), 회사 관심사 = **전방향 정합 인체 포인트클라우드→모션 추론**. 로드맵을 9월~2월 달력에 고정하고 3단계를 본 과제로 확장, 지출 계획(~$75k + 버퍼) 추가, LiDAR-mimic을 합성 데이터 엔진으로 편입. 다음 = Stage 0.
