from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from rlnf_rrt_v2.engine.train import train


if __name__ == "__main__":
    train()
