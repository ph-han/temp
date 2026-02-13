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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_predictions_from_dir(pred_dir: Path, ds_idx: int) -> list[np.ndarray]:
    files = sorted(pred_dir.glob(f"pred_{ds_idx:07d}_s*.npy"))
    return [np.load(f).astype(np.float32) for f in files]


@torch.no_grad()
def _predict_from_checkpoint(
    model: Flow,
    sample: dict,
    num_samples_per_condition: int,
    device: torch.device,
) -> list[np.ndarray]:
    map_img = sample["map"].unsqueeze(0).to(device)
    start = sample["start"].unsqueeze(0).to(device)
    goal = sample["goal"].unsqueeze(0).to(device)
    gt_path = sample["gt_path"].unsqueeze(0).to(device)

    t = gt_path.size(1)
    z = torch.randn(
        (num_samples_per_condition, t, 2),
        device=device,
        dtype=gt_path.dtype,
    )

    map_rep = map_img.expand(num_samples_per_condition, -1, -1, -1)
    start_rep = start.expand(num_samples_per_condition, -1)
    goal_rep = goal.expand(num_samples_per_condition, -1)

    pred_path, _ = model.inverse(map_rep, start_rep, goal_rep, z)
    pred_np = pred_path.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    return [pred_np[i] for i in range(pred_np.shape[0])]


def _parse_indices(indices_arg: list[int] | None, ds_len: int) -> list[int]:
    if not indices_arg:
        return list(range(ds_len))

    picked: list[int] = []
    for i in indices_arg:
        if 0 <= i < ds_len:
            picked.append(i)
    if not picked:
        raise RuntimeError("No valid indices selected.")
    return sorted(set(picked))


def _render_axes(
    ax,
    sample: dict,
    preds: list[np.ndarray],
    ds_idx: int,
    max_samples: int | None,
    cmap: ListedColormap,
) -> None:
    map_np = sample["map"].squeeze(0).numpy()
    start = sample["start"].numpy()
    goal = sample["goal"].numpy()
    gt = sample["gt_path"].numpy()

    if max_samples is not None:
        preds = preds[: max(0, max_samples)]

    ax.imshow(map_np, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")

    if len(gt) > 0:
        ax.scatter(gt[:, 0], gt[:, 1], s=10, c="#18a71e", alpha=0.65, label="GT Path", edgecolors="none")

    if preds:
        pred_all = np.concatenate(preds, axis=0)
        ax.scatter(
            pred_all[:, 0],
            pred_all[:, 1],
            s=12,
            c="#1a2cff",
            alpha=0.45,
            label=f"Samples (n={len(preds)})",
            edgecolors="none",
        )

    ax.scatter([start[0]], [start[1]], s=220, c="red", label="Start", edgecolors="none", zorder=3)
    ax.scatter([goal[0]], [goal[1]], s=220, c="lime", label="Goal", edgecolors="none", zorder=3)

    ax.set_title(f"Example {ds_idx}", fontsize=18, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prediction plotter (default: infer directly from checkpoint and save pages)"
    )
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--pred-dir", type=str, default=None, help="If set, use saved pred_*.npy instead of checkpoint inference")
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--num-samples-per-condition", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--examples-per-figure", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="outputs/figures/prediction_pages")
    parser.add_argument("--prefix", type=str, default="prediction")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.examples_per_figure <= 0:
        raise ValueError("--examples-per-figure must be > 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_toml(args.eval_config)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

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

    picked = _parse_indices(args.indices, len(ds))
    if args.max_examples is not None:
        picked = picked[: max(0, args.max_examples)]
    if not picked:
        raise RuntimeError("No examples to render.")

    use_pred_dir = args.pred_dir is not None
    model: Flow | None = None
    device: torch.device | None = None
    pred_dir: Path | None = None

    if args.device == "auto":
        selected_device = get_device()
    else:
        selected_device = torch.device(args.device)

    if use_pred_dir:
        pred_dir = resolve_project_path(args.pred_dir)
        if not pred_dir.exists():
            raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    else:
        ckpt_path = resolve_project_path(args.checkpoint or eval_cfg["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        device = selected_device
        ckpt = torch.load(ckpt_path, map_location=device)
        model = _build_model_from_ckpt(ckpt, device)

    num_samples = int(args.num_samples_per_condition or eval_cfg.get("num_samples_per_condition", 5))
    if num_samples <= 0:
        raise ValueError("--num-samples-per-condition must be > 0")

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(["#3b3336", "#dfdfdf"])
    ncols = 2

    page_paths: list[Path] = []
    total_pages = math.ceil(len(picked) / args.examples_per_figure)

    for page_idx in range(total_pages):
        start_i = page_idx * args.examples_per_figure
        end_i = min(start_i + args.examples_per_figure, len(picked))
        page_indices = picked[start_i:end_i]

        n = len(page_indices)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 8 * nrows))
        axes_arr = np.atleast_1d(axes).reshape(-1)

        for ax_i, ds_idx in enumerate(page_indices):
            sample = ds[ds_idx]
            if use_pred_dir:
                assert pred_dir is not None
                preds = _load_predictions_from_dir(pred_dir, ds_idx)
            else:
                assert model is not None and device is not None
                preds = _predict_from_checkpoint(model, sample, num_samples, device)

            _render_axes(
                ax=axes_arr[ax_i],
                sample=sample,
                preds=preds,
                ds_idx=ds_idx,
                max_samples=args.max_samples,
                cmap=cmap,
            )

            if ax_i in (0, n - 1):
                axes_arr[ax_i].legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=13)

        for ax in axes_arr[n:]:
            ax.axis("off")

        fig.tight_layout()

        page_path = out_dir / f"{args.prefix}_page_{page_idx + 1:04d}.png"
        fig.savefig(page_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        page_paths.append(page_path)
        print(f"Saved page {page_idx + 1}/{total_pages}: {page_path}")

    print(f"Saved {len(page_paths)} figure(s) to: {out_dir}")
    print(f"Examples: {len(picked)}  per_figure: {args.examples_per_figure}")
    if use_pred_dir:
        assert pred_dir is not None
        print(f"Source: pred_dir={pred_dir}")
    else:
        print(f"Source: checkpoint={resolve_project_path(args.checkpoint or eval_cfg['checkpoint'])}")

    if args.show and page_paths:
        img = plt.imread(page_paths[0])
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Preview: {page_paths[0].name}")
        plt.show()


if __name__ == "__main__":
    main()
