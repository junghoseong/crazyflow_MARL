# LIA_MADDPG 기술 문서

동적 다중 로봇 태스크 할당(Dynamic Task Allocation) 문제에 대한 **Local Information Aggregation 기반 MADDPG(LIA_MADDPG)** 재현 구현을 설명합니다. 논문: *“A Local Information Aggregation based Multi-Agent Reinforcement Learning for Robot Swarm Dynamic Task Allocation”* (Lv, Lei, Yi).

이 저장소의 **Phase 1** 구현은 논문의 2D 단순 운동 모델과 보상 설계에 맞춘 **순수 JAX 환경**과 **CTDE(중앙 집중 학습 / 분산 실행)** 학습 파이프라인입니다. Crazyflow 드론 물리(Phase 2)와는 분리되어 있으며, 같은 `marl/` 패키지를 확장하는 방식으로 연결할 수 있습니다.

---

## 1. 문제 설정 (Problem setup)

### 1.1 배경

- **에이전트**: 동질(homogeneous) 로봇 N대. 각 로봇은 시각마다 **목표 태스크 M개 중 하나**를 선택하고, 선택한 태스크를 향해 일정 속도로 이동합니다.
- **태스크**: 2D 평면에서 이동하는 M개의 동적 태스크. 속도·방향이 시간에 따라 변합니다.
- **바인딩(binding)**: 로봇과 **현재 목표 태스크** 사이 거리가 `d_bind` 이하이면 해당 로봇은 태스크에 “바인딩”되고, 이후 목표 전환 없이 태스크와 동기화된 움직임을 따릅니다(구현 상 status=0).
- **용량 제약**: 태스크 j당 최대 수용 가능 로봇 수는 `ceil(N/M)`에 해당하는 상한(코드: `EnvConfig.task_max_demand`)으로 둡니다. 초과 시 바인딩 보상이 제한됩니다(논문 Eq. 9와 정합).

### 1.2 최적화 관점 (요약)

각 로봇의 유틸리티는 **태스크 보상**과 **이동 비용**으로 구성됩니다. 에피소드 동안 로봇들이 협력적으로 태스크를 나누어 가져야 하며, 부분 관측과 비정상성(non-stationarity)을 **Dec-POMDP**로 정식화합니다.

### 1.3 Dec-POMDP와의 대응 (코드 매핑)

| 논문 요소 | 코드 |
|-----------|------|
| 전역 상태 (로봇+태스크) | `EnvState` (`marl/envs/task_allocation.py`) |
| 로봇 i의 관측 | `_get_obs()` → 벡터 `(N, obs_dim)` |
| 이산 행동 (태스크 선택) | `(N, M)` 로짓/원-핫; 학습 시 Gumbel-Softmax |
| 전이 | `TaskAllocationEnv.step()` |
| 보상 | `_compute_rewards()` (아래 §3 참고) |

---

## 2. 환경 (Environment)

### 2.1 모듈 위치

| 파일 | 역할 |
|------|------|
| [marl/envs/task_allocation.py](marl/envs/task_allocation.py) | `EnvConfig`, `EnvState`, `TaskAllocationEnv` |
| [marl/lia.py](marl/lia.py) | 이웃 집합, 거리 가중 LIA 집계 (크리틱 입력용) |

### 2.2 설정 (`EnvConfig`)

주요 하이퍼파라미터(논문 Section IV-A와 맞춘 기본값):

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `n_robots` | 60 | N |
| `n_tasks` | 10 | M |
| `area_size` | 1000 | 2D 영역 한 변 길이 (m) |
| `robot_speed_min`, `robot_speed_max` | 2, 5 | 로봇 속도 범위 (m/s) |
| `task_speed_min`, `task_speed_max` | 0.5, 1 | 태스크 속도 범위 (m/s) |
| `d_bind` | 30 | 바인딩 거리 (m) |
| `max_neighbors` | α (논문 10) | 관측·LIA에서 쓰는 최대 이웃 수 |
| `max_steps` | 150 | 에피소드 최대 스텝 |
| `tau` | 1.0 | 의사결정 시간 간격 |
| `phi_1`, `phi_2`, `phi_3` | 10, 0.001, 1 | 보상 가중 (§3) |

### 2.3 상태 (`EnvState`)

- **로봇**: `robot_pos (N,2)`, `robot_speed (N,)`, `robot_dir (N,)`, `robot_status (N,)` (1=미바인딩, 0=바인딩), `robot_target (N,)`.
- **태스크**: `task_pos (M,2)`, `task_speed (M,)`, `task_angle (M,)`, `task_bound_count (M,)`, `task_rewards (M,)` (태스크별 \(r_{i,j}\) 스칼라 샘플).
- **기타**: `step_count`, `rng_key`.

### 2.4 동역학 (`step`)

1. 행동으로부터 목표 태스크 인덱스 결정(`argmax` of soft/hard one-hot). 미바인딩 로봇만 목표 갱신.
2. 태스크 위치 갱신: 속도·각도에 따른 이동 후 영역 `[0, area_size]`로 클립.
3. 로봇 위치 갱신: 목표 태스크를 향해 `robot_speed * tau`만큼 이동. 이미 바인딩된 로봇은 해당 태스크 위치와 동기화.
4. 거리 `≤ d_bind`인 미바인딩 로봇을 바인딩 처리, `task_bound_count` 갱신.
5. 보상 계산 후 `step_count` 증가. `max_steps` 도달 또는 전 로봇 바인딩 시 에피소드 종료 플래그.

### 2.5 관측 공간 (`obs_dim`)

로봇 i당 벡터는 다음 세 블록의 연결입니다.

1. **자기 상태** `o_self` (5차원): 정규화된 위치(x,y), 정규화 속도, 방향/π, status.
2. **태스크 상대** `o_task` (6M차원): 각 태스크 j에 대해 상대 위치/속도/각도 차이, 정규화된 바인딩 수 \(h_j\), 정규화된 태스크 보상 가중 \(\kappa\).
3. **이웃 상대** `o_neighbor` (5α차원): 거리 기준 상위 α개 이웃에 대한 상대 상태 및 이전 목표 태스크 인덱스(정규화).

→ `obs_dim = 5 + 6*M + 5*max_neighbors`.

### 2.6 행동 공간

- **형태**: 이산 M-way 선택. 네트워크는 M차원 로짓을 출력하고, 학습 시 **Gumbel-Softmax**로 미분 가능한 샘플, 실행 시 **softmax/argmax**를 사용합니다.

---

## 3. 보상 설계 (Rewards)

논문 Eq. (7)–(9)에 대응하는 **밀집(shaping) 보상**을 사용합니다.

- **거리/시간 패널티** (`r_dis`): 미바인딩 로봇에 대해 매 스텝 `-phi_2` (코드에서 `phi_2`는 양수로 저장).
- **용량 정합 항** (`r_step`): `phi_3 * (task_max_demand - h_j)` 형태로, 현재 목표 태스크 j의 여유 슬롯을 반영(과밀 페널티 유도).
- **바인딩 최종 보상** (`r_final`): 새로 바인딩된 경우, \(h_j \le \bar{h}_j\)이면 `phi_1 * task_rewards[j]`.

에피소드 종료 시 모든 로봇에 동일한 `done` 플래그를 두어 멀티에이전트 TD에서 일관되게 사용합니다.

---

## 4. 에이전트 학습 프레임워크 (LIA_MADDPG)

### 4.1 CTDE 개요

- **학습**: 모든 로봇의 전이가 리플레이에 저장되며, **공유(sharing) 액터·크리틱**으로 업데이트합니다.
- **실행**: 각 로봇은 자신의 국소 관측만으로 정책을 구동합니다(Algorithm 2에서 선택적 정책 개선 포함).

### 4.2 LIA (Local Information Aggregation)

**목적**: 전 에이전트 joint 관측/행동을 크리틱에 넣지 않고, 로봇 i와 **국소적으로 연관된** 에이전트 집합 \( \mathcal{G}_i \)의 정보만 거리 가중 합으로 고정 차원 벡터화합니다.

구현([marl/lia.py](marl/lia.py)):

- **공간 이웃** \( \mathcal{N}_i \): 유클리드 거리 상위 α개.
- **동일 행동 집합** \( \mathcal{L}_i \): 같은 목표 태스크 인덱스를 선택한 로봇.
- **\( \mathcal{G}_i = (\mathcal{N}_i \cup \mathcal{L}_i) \setminus \{i\} \)** 마스크 후 가중치  
  \( w_{i,k} \propto d_{i,k}^{\beta} \) (기본 `beta=-1` → 가까울수록 큰 가중).
- **집계**: \( \phi_i(o) = \sum_k w_{i,k} o_k \), \( \phi_i(a) = \sum_k w_{i,k} a_k \).

리플레이에는 샘플 시점의 `phi_obs`, `phi_actions`가 함께 저장됩니다.

### 4.3 네트워크 ([marl/networks.py](marl/networks.py))

- **SharedActor**: 관측 → M차원 로짓. 은닉 128×2 + 잔차 블록 + BatchNorm.
- **SharedCritic**: 입력 `[o_i, a_i, phi_i(o), phi_i(a)]` → 스칼라 Q. 동일 은닉 구조.
- **이산 행동 + DDPG 계열**: 연속 행동 가정을 피하기 위해 **Gumbel-Softmax**로 soft one-hot을 씁니다(`gumbel_softmax`, temperature 스케줄은 `MADDPGConfig`).

**Flax BatchNorm 주의**: 학습 시 `mutable=["batch_stats"]`로 업데이트, 타깃·롤아웃 추론은 `use_running_average=True` 모드의 별도 `apply` 경로(`actor_inference`, `critic_inference` in [marl/maddpg.py](marl/maddpg.py))를 사용합니다.

### 4.4 리플레이 및 우선순위 ([marl/replay_buffer.py](marl/replay_buffer.py))

- 용량 기본 5000, 배치 64.
- 전이 단위: `(obs, actions, phi_obs, phi_actions, rewards, next_obs, dones, robot_pos)` 등.
- TD 오차 기반 우선 샘플링 및 importance weight(β 애닐링).

### 4.5 학습 업데이트 ([marl/maddpg.py](marl/maddpg.py))

- **크리틱**: Huber 대신 MSE TD 오차 최소화, 타깃 액터·타깃 크리틱으로 \(y = r + \gamma (1-d) Q'\). 다음 스텝의 LIA 특징은 구현 상 현재 transition의 phi를 근사로 재사용합니다(속도·정확도 트레이드오프).
- **액터**: 현재 크리틱 Q에 대해 정책 경사(softmax 온도 반영).
- **소프트 타깃 업데이트**: `tau_soft` (논문 η=0.01).
- **에피소드 루프**: [examples/train_lia_maddpg.py](examples/train_lia_maddpg.py)에서 수집 후 `train_step`; 공유 네트워크 비용 때문에 매 배치마다 **일부 로봇 인덱스만** 순차 업데이트하는 최적화가 들어가 있습니다(기본 최대 10명).

### 4.6 분산 실행 시 정책 개선 (Algorithm 2)

`policy_improvement()`:

- 정책 출력으로 목표 태스크를 고른 뒤, **편차 확률** \(\delta\) (남은 슬롯·이웃과의 행동 충돌 정도)에 따라 확률적으로 **휴리스틱** `argmax_j (phi_1 * r_j - d_{i,j})`로 재선택합니다.

### 4.7 평가 지표 ([marl/utils.py](marl/utils.py), [examples/eval_lia_maddpg.py](examples/eval_lia_maddpg.py))

- **NATU / NATC / DR**: 여러 시나리오에 대한 정규화 유틸리티, 정규화 완료 시간, 대 baseline 우세 비율.
- `greedy_policy`: 비교용 베이스라인.

---

## 5. 실행 방법

저장소 루트에서 **`marl` 패키지를 PYTHONPATH에 포함**해야 합니다.

```bash
# 학습 (논문 규모 예시)
PYTHONPATH=. python examples/train_lia_maddpg.py \
  --n_robots 60 --n_tasks 10 --num_episodes 5000

# 소규모 스모크 테스트
PYTHONPATH=. python examples/train_lia_maddpg.py \
  --n_robots 10 --n_tasks 3 --num_episodes 200 --batch_size 32

# 평가 (여러 시나리오)
PYTHONPATH=. python examples/eval_lia_maddpg.py \
  --n_robots 60 --n_tasks 10 --num_scenarios 100
```

체크포인트: `--save_path DIR` 시 액터/크리틱 파라미터 및 BatchNorm 통계가 `npz`로 저장됩니다. 평가 스크립트는 현재 **학습된 가중치 로드** 경로가 없으면 무작위 초기화 네트워크로 동작하므로, 의미 있는 비교를 위해서는 저장된 파라미터를 불러오는 로더를 추가하는 것이 좋습니다(향후 작업).

---

## 6. Crazyflow(3D)와의 관계 (Phase 2)

- 현재 `TaskAllocationEnv`는 **2D 기하 + 단순 운동 모델**로 논문 시뮬레이션 설정에 맞춥니다.
- [crazyflow/envs/drone_env.py](crazyflow/envs/drone_env.py)의 `n_drones>1` 확장, 이동 목표를 3D 트래젝토리로 두는 방식으로 동일한 `marl/` 학습 루프를 재사용할 수 있습니다.

---

## 7. 참고

- 논문 원문: 저장소 내 `LIA_MADDPG.tex`
- 단일 드론 PPO 예시: [examples/train_ppo.py](examples/train_ppo.py)
