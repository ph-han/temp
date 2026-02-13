from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.flow import Flow
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.utils import get_device


def _build_model_from_ckpt(ckpt: dict, device: torch.device) -> Flow:
    cfg = ckpt.get("config")
    if cfg is None or "model" not in cfg:
        raise RuntimeError("Checkpoint missing model config. Re-train with current train.py.")

    m = cfg["model"]
    model = Flow(
        num_blocks=int(m["num_blocks"]),
        latent_dim=int(m["latent_dim"]),
        hidden_dim=int(m["hidden_dim"]),
        s_max=float(m["s_max"]),
        channels=tuple(int(x) for x in m["channels"]),
        norm=str(m["norm"]),
        map_scale=float(m.get("map_scale", 1.5)),
        sg_scale=float(m.get("sg_scale", 2.0)),
        cond_norm=str(m.get("cond_norm", "none")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _minmax01(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot map encoder feature heatmaps")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--stage", type=str, default="backbone", choices=["backbone", "pool"])
    parser.add_argument("--num-channels", type=int, default=6)
    parser.add_argument("--save", type=str, default="outputs/figures/map_feature_heatmap.png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    return parser


@torch.no_grad()
def main() -> None:
    args = _make_parser().parse_args()

    cfg = load_toml(args.eval_config)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    ckpt_path = resolve_project_path(args.checkpoint or eval_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    split = args.split or str(data_cfg.get("split", "test"))
    ds = RLNFDataset(
        split=split,
        data_root=resolve_project_path(data_cfg.get("data_root", "data")),
        noise_std=0.0,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty with current config filters.")

    idx = max(0, min(args.index, len(ds) - 1))
    sample = ds[idx]
    map_np = sample["map"].squeeze(0).numpy()
    start_np = sample["start"].numpy()
    goal_np = sample["goal"].numpy()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = _build_model_from_ckpt(ckpt, device)

    map_img = sample["map"].unsqueeze(0).to(device)
    feat = model.cond_encoder.map_encoder.backbone(map_img)
    if args.stage == "pool":
        feat = model.cond_encoder.map_encoder.pool(feat)

    feat_np = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (C,H,W)
    c = feat_np.shape[0]
    if c == 0:
        raise RuntimeError("No feature channels found.")

    mean_map = _minmax01(feat_np.mean(axis=0))
    channel_scores = feat_np.mean(axis=(1, 2))

    n_ch = max(0, min(args.num_channels, c))
    top_channels = np.argsort(channel_scores)[::-1][:n_ch]

    total_panels = 2 + n_ch
    ncols = min(4, total_panels)
    nrows = math.ceil(total_panels / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    map_cmap = ListedColormap(["#3a3337", "#dcdcdc"])

    ax0 = axes[0]
    ax0.imshow(
        map_np,
        cmap=map_cmap,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        extent=(0, 1, 0, 1),
        interpolation="nearest",
    )
    ax0.scatter([start_np[0]], [start_np[1]], s=120, c="red", label="Start", zorder=3)
    ax0.scatter([goal_np[0]], [goal_np[1]], s=120, c="lime", label="Goal", zorder=3)
    ax0.set_title("Input Map", fontsize=12, fontweight="bold")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_aspect("equal", adjustable="box")
    ax0.legend(loc="upper right", framealpha=0.9, fontsize=9)

    ax1 = axes[1]
    ax1.imshow(
        map_np,
        cmap=map_cmap,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        extent=(0, 1, 0, 1),
        interpolation="nearest",
    )
    ax1.imshow(
        mean_map,
        cmap="inferno",
        origin="lower",
        extent=(0, 1, 0, 1),
        interpolation="bilinear",
        alpha=0.72,
    )
    ax1.set_title(f"{args.stage}: Mean Activation", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal", adjustable="box")

    for i, ch in enumerate(top_channels, start=2):
        ax = axes[i]
        ch_map = _minmax01(feat_np[ch])
        ax.imshow(
            map_np,
            cmap=map_cmap,
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=(0, 1, 0, 1),
            interpolation="nearest",
        )
        ax.imshow(
            ch_map,
            cmap="viridis",
            origin="lower",
            extent=(0, 1, 0, 1),
            interpolation="bilinear",
            alpha=0.74,
        )
        ax.set_title(
            f"Ch {int(ch)} (score={channel_scores[ch]:.3f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[total_panels:]:
        ax.axis("off")

    for ax in axes[:total_panels]:
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        f"Map Encoder Feature Heatmaps ({args.stage}) | split={split}, index={idx}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    save_path = resolve_project_path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {save_path}")
    print(f"checkpoint: {ckpt_path}")
    print(f"split/index: {split}/{idx}")
    print(f"feature shape: {tuple(feat.shape)}")
    print(f"top channels: {top_channels.tolist()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
