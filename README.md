# RLNF-RRT-v2

딥러닝 모델 제작 프로젝트 기본 구조입니다.

## Project Structure

```text
.
├── configs/
│   ├── data/
│   ├── model/
│   └── train/
├── data/
│   ├── train/
│   │   ├── map/
│   │   ├── start_goal/
│   │   ├── gt_path/
│   │   └── meta.csv
│   ├── val/
│   └── test/
├── notebooks/
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   └── figures/
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── data/
│       └── generate_2d.py
├── src/
│   └── rlnf_rrt_v2/
│       ├── data/
│       │   ├── dataset.py
│       │   └── generator_2d.py
│       ├── models/
│       │   └── model.py
│       ├── engine/
│       │   ├── train.py
│       │   └── evaluate.py
│       └── utils/
│           └── seed.py
├── tests/
│   └── test_smoke.py
├── main.py
└── pyproject.toml
```

## Quick Start

```bash
./.venv/bin/python main.py
./.venv/bin/python scripts/train.py
./.venv/bin/python scripts/eval.py
```

## Generate Dataset

```bash
./.venv/bin/python scripts/data/generate_2d.py --split train --num-maps 100 --num-start-goal 10
./.venv/bin/python scripts/data/generate_2d.py --split val --num-maps 20 --num-start-goal 10
./.venv/bin/python scripts/data/generate_2d.py --split test --num-maps 20 --num-start-goal 10
```

주의: `gt_path` 생성에는 `AStar` 구현이 필요합니다.
`src/rlnf_rrt_v2/utils/astar.py`에 `AStar` 클래스를 추가한 뒤 실행하세요.
