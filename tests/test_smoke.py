from rlnf_rrt_v2.engine.evaluate import evaluate
from rlnf_rrt_v2.engine.train import train


def test_entrypoints_smoke() -> None:
    train()
    evaluate()
