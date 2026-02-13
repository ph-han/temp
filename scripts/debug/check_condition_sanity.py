from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


def _path_mean_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: (B,T,2)
    return float(torch.norm(a - b, dim=-1).mean().item())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks for flow conditioning and invertibility")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--index-a", type=int, default=0, help="Primary sample index")
    parser.add_argument("--index-b", type=int, default=1, help="Comparison sample index")
    parser.add_argument("--num-z", type=int, default=32, help="Number of latent draws for sensitivity checks")
    parser.add_argument("--z-mode", type=str, default="rand", choices=["rand", "zero"], help="Latent sampling mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_z <= 0:
        raise ValueError("--num-z must be > 0")

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
    if len(ds) < 2:
        raise RuntimeError("Need at least 2 samples in dataset for comparison checks.")

    idx_a = max(0, min(args.index_a, len(ds) - 1))
    idx_b = max(0, min(args.index_b, len(ds) - 1))
    if idx_a == idx_b:
        idx_b = (idx_a + 1) % len(ds)

    # Ensure map check uses a different map file than index_a.
    map_a_file = str(ds.meta_data.iloc[idx_a]["map_file"])
    if str(ds.meta_data.iloc[idx_b]["map_file"]) == map_a_file:
        found = None
        for j in range(len(ds)):
            if str(ds.meta_data.iloc[j]["map_file"]) != map_a_file:
                found = j
                break
        if found is None:
            raise RuntimeError("Could not find a sample with a different map for map-sensitivity check.")
        idx_b = found

    sample_a = ds[idx_a]
    sample_b = ds[idx_b]

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    if device.type == "mps":
        # MPS backend has adaptive pooling shape restrictions for some input sizes.
        device = torch.device("cpu")
        print("info: switched device from mps to cpu for compatibility.")

    ckpt_path = resolve_project_path(args.checkpoint or eval_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = _build_model_from_ckpt(ckpt, device)

    map_a = sample_a["map"].unsqueeze(0).to(device)
    map_b = sample_b["map"].unsqueeze(0).to(device)
    start_a = sample_a["start"].unsqueeze(0).to(device)
    goal_a = sample_a["goal"].unsqueeze(0).to(device)
    start_b = sample_b["start"].unsqueeze(0).to(device)
    goal_b = sample_b["goal"].unsqueeze(0).to(device)
    gt_a = sample_a["gt_path"].unsqueeze(0).to(device)

    t = gt_a.size(1)
    if args.z_mode == "zero":
        z = torch.zeros((args.num_z, t, 2), device=device, dtype=gt_a.dtype)
    else:
        z = torch.randn((args.num_z, t, 2), device=device, dtype=gt_a.dtype)

    map_a_rep = map_a.expand(args.num_z, -1, -1, -1)
    map_b_rep = map_b.expand(args.num_z, -1, -1, -1)
    start_a_rep = start_a.expand(args.num_z, -1)
    goal_a_rep = goal_a.expand(args.num_z, -1)
    # Use swapped start/goal for SG sensitivity on the same map.
    start_swap_rep = goal_a.expand(args.num_z, -1)
    goal_swap_rep = start_a.expand(args.num_z, -1)

    # Check 1: start/goal conditioning sensitivity (same map, same z)
    pred_base, _ = model.inverse(map_a_rep, start_a_rep, goal_a_rep, z)
    pred_sg_changed, _ = model.inverse(map_a_rep, start_swap_rep, goal_swap_rep, z)
    sg_diff = _path_mean_l2(pred_base, pred_sg_changed)

    # Check 2: map conditioning sensitivity (same start/goal, same z)
    pred_map_changed, _ = model.inverse(map_b_rep, start_a_rep, goal_a_rep, z)
    map_diff = _path_mean_l2(pred_base, pred_map_changed)

    # Check 3: cycle consistency x -> z -> x_recon
    z_f, log_det_f = model(map_a, start_a, goal_a, gt_a)
    x_recon, log_det_inv = model.inverse(map_a, start_a, goal_a, z_f)
    recon_mse = float(torch.mean((x_recon - gt_a) ** 2).item())
    recon_linf = float(torch.max(torch.abs(x_recon - gt_a)).item())
    logdet_closure = float(torch.mean(torch.abs(log_det_f + log_det_inv)).item())

    print("=== Condition/Flow Sanity Check ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"split: {split}  index_a: {idx_a}  index_b: {idx_b}")
    print(f"num_z: {args.num_z}  z_mode: {args.z_mode}  T: {t}")
    print("")
    print("[1] Start/Goal sensitivity (same map, same z)")
    print(f"mean_path_L2_diff: {sg_diff:.6f}")
    print("guide: near 0이면 start/goal 조건이 약하게 반영될 가능성")
    print("")
    print("[2] Map sensitivity (same start/goal, same z)")
    print(f"mean_path_L2_diff: {map_diff:.6f}")
    print("guide: near 0이면 map 조건이 약하게 반영될 가능성")
    print("")
    print("[3] Cycle consistency (x -> z -> x_recon)")
    print(f"recon_mse: {recon_mse:.8f}")
    print(f"recon_linf: {recon_linf:.8f}")
    print(f"logdet_closure_abs_mean: {logdet_closure:.8f}")
    print("guide: recon/logdet closure가 크면 invertibility 경로 점검 필요")


if __name__ == "__main__":
    main()
