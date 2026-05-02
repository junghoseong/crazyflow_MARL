# crazyflow_MARL

[Crazyflow](https://github.com/utiasDSL/crazyflow) 기반 드론 강화학습 실험 레포지토리.

Crazyflow의 JAX/MuJoCo 시뮬레이션과 Gymnasium 환경을 그대로 포함하며, 그 위에 RL 학습 알고리즘을 구현합니다.

## 프로젝트 구조

```
crazyflow_MARL/
├── crazyflow/           # 시뮬레이션 & 환경 (JAX + MuJoCo MJX)
│   ├── sim/             #   물리 시뮬레이션 코어
│   ├── envs/            #   Gymnasium VectorEnv 구현
│   ├── control/         #   Attitude/Force-Torque 컨트롤러
│   └── randomize/       #   도메인 랜덤화
├── marl/                # 다중 에이전트 태스크 할당 (LIA_MADDPG, JAX)
│   ├── envs/            #   2D Task Allocation 환경
│   ├── lia.py           #   Local Information Aggregation
│   ├── maddpg.py        #   LIA_MADDPG 학습·실행 로직
│   └── ...
├── submodules/          # 드론 모델 & 컨트롤러 파라미터
│   ├── drone-models/
│   └── drone-controllers/
├── examples/            # 환경 데모 & 학습 스크립트
│   ├── train_ppo.py     #   PPO 학습 (JAX/Flax)
│   ├── train_lia_maddpg.py  # LIA_MADDPG 학습 (2D 태스크 할당)
│   ├── eval_lia_maddpg.py   # LIA_MADDPG 평가 (NATU/NATC/DR)
│   ├── gymnasium_env.py #   Gymnasium 환경 사용 예시
│   ├── figure8.py       #   Figure-8 궤적 추종 예시
│   └── ...
├── LIA_MADDPG.md        # LIA_MADDPG 문제·환경·학습 프레임워크 기술 문서
├── tests/               # 유닛/통합 테스트
├── docs/                # 문서
└── pyproject.toml       # 의존성 및 프로젝트 설정
```

## 설치

```bash
# 클론
git clone https://github.com/junghoseong/crazyflow_MARL.git
cd crazyflow_MARL

# pip 설치 (editable)
pip install -e .

# 또는 pixi 사용
pixi install
```

## 학습 실행

### LIA_MADDPG (다중 로봇 동적 태스크 할당, 2D)

문제 설정, 관측·보상, LIA·MADDPG 구현 세부는 [LIA_MADDPG.md](LIA_MADDPG.md)를 참고하세요. `marl` 패키지는 저장소 루트를 `PYTHONPATH`에 두어야 import됩니다.

```bash
# 논문 규모에 가깝게 (N=60, M=10)
PYTHONPATH=. python examples/train_lia_maddpg.py \
  --n_robots 60 --n_tasks 10 --num_episodes 5000

# 평가 (다수 시나리오, greedy 대비 지표)
PYTHONPATH=. python examples/eval_lia_maddpg.py \
  --n_robots 60 --n_tasks 10 --num_scenarios 100
```

### PPO - 목표 위치 도달 (DroneReachPos)
```bash
python examples/train_ppo.py \
    --env_id DroneReachPos-v0 \
    --num_envs 64 \
    --total_timesteps 2000000 \
    --freq 50
```

### PPO - Figure-8 궤적 추종
```bash
python examples/train_ppo.py \
    --env_id DroneFigureEightTrajectory-v0 \
    --num_envs 64 \
    --total_timesteps 2000000 \
    --max_episode_time 10.0
```

### 주요 하이퍼파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--env_id` | `DroneReachPos-v0` | 환경 ID |
| `--num_envs` | `64` | 병렬 환경 수 |
| `--total_timesteps` | `2000000` | 총 학습 스텝 수 |
| `--learning_rate` | `3e-4` | 학습률 |
| `--num_steps` | `128` | 업데이트 당 rollout 길이 |
| `--gamma` | `0.99` | 할인 계수 |
| `--clip_eps` | `0.2` | PPO 클리핑 엡실론 |
| `--save_path` | `""` | 모델 저장 경로 |

## 사용 가능한 환경

| 환경 ID | 설명 | Obs 차원 | Action 차원 |
|---------|------|----------|-------------|
| `DroneReachPos-v0` | 랜덤 목표 위치 도달 | 16 (pos, quat, vel, ang_vel, diff_to_goal) | 4 |
| `DroneReachVel-v0` | 목표 속도 도달 | 16 (pos, quat, vel, ang_vel, diff_to_vel) | 4 |
| `DroneLanding-v0` | 착륙 | 16 | 4 |
| `DroneFigureEightTrajectory-v0` | Figure-8 궤적 추종 | 13 + 3*n_samples | 4 |

## 기술 스택

- **시뮬레이션**: MuJoCo MJX (GPU-accelerated physics)
- **프레임워크**: JAX (JIT, vmap, grad)
- **환경 API**: Gymnasium VectorEnv
- **네트워크**: Flax (linen)
- **최적화**: Optax
- **제어**: Attitude (roll, pitch, yaw, thrust) / Force-Torque

## TODO

- [x] LIA_MADDPG (2D Task Allocation, JAX/Flax) — [LIA_MADDPG.md](LIA_MADDPG.md)
- [ ] MAPPO (Multi-Agent PPO) 구현
- [ ] WandB 로깅 연동
- [ ] 학습된 정책 시각화 스크립트
- [ ] 하이퍼파라미터 sweep 설정
- [ ] Multi-drone 환경 확장 (Crazyflow + `marl` 연동)

## 참고 자료

- [LIA_MADDPG 기술 문서 (이 레포)](LIA_MADDPG.md)
- [Crazyflow (원본)](https://github.com/utiasDSL/crazyflow)
- [Schulman et al., "Proximal Policy Optimization Algorithms", 2017](https://arxiv.org/abs/1707.06347)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [PureJaxRL](https://github.com/luchris429/purejaxrl)
