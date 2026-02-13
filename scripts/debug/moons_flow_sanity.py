from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.models.affine_coupling_block import AffineCouplingBlock
from rlnf_rrt.utils.config import resolve_project_path


def make_moons(n_samples: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    t0 = rng.uniform(0.0, np.pi, size=n0)
    x0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)

    t1 = rng.uniform(0.0, np.pi, size=n1)
    x1 = np.stack([1.0 - np.cos(t1), -np.sin(t1) - 0.5], axis=1)

    x = np.concatenate([x0, x1], axis=0).astype(np.float32)
    x += rng.normal(0.0, noise, size=x.shape).astype(np.float32)
    return x


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return (train_x - mu) / std, (test_x - mu) / std, mu, std


class UncondFlow(nn.Module):
    def __init__(self, num_blocks: int, hidden_dim: int, s_max: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AffineCouplingBlock(cond_dim=0, hidden_dim=hidden_dim, s_max=s_max) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B,2)
        z = x[:, None, :]
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        cond = x.new_zeros((x.shape[0], 0))
        for block in self.blocks:
            z, d = block(z, cond)
            log_det += d
        return z[:, 0, :], log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, None, :]
        cond = z.new_zeros((z.shape[0], 0))
        for block in reversed(self.blocks):
            x, _ = block.inverse(x, cond)
        return x[:, 0, :]


def nll_loss(z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
    # z: (B,2), log_det: (B,)
    log_2pi = np.log(2.0 * np.pi)
    log_pz = -0.5 * (z.pow(2).sum(dim=-1) + 2.0 * log_2pi)
    return -(log_pz + log_det).mean()


@torch.no_grad()
def eval_nll(model: UncondFlow, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for (x,) in loader:
        x = x.to(device)
        z, log_det = model(x)
        loss = nll_loss(z, log_det)
        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check affine flow on a 2D moons dataset")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--s-max", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/figures/moons_flow_sanity.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_x_np = make_moons(args.n_train, args.noise, args.seed)
    test_x_np = make_moons(args.n_test, args.noise, args.seed + 1)
    train_x_np, test_x_np, mu, std = standardize(train_x_np, test_x_np)

    train_ds = TensorDataset(torch.from_numpy(train_x_np).float())
    test_ds = TensorDataset(torch.from_numpy(test_x_np).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cpu")
    model = UncondFlow(num_blocks=args.num_blocks, hidden_dim=args.hidden_dim, s_max=args.s_max).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_test = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for (x,) in train_loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            z, log_det = model(x)
            loss = nll_loss(z, log_det)
            loss.backward()
            opt.step()
            bs = x.size(0)
            total += float(loss.item()) * bs
            n += bs
        train_nll = total / max(1, n)
        test_nll = eval_nll(model, test_loader, device)
        if test_nll < best_test:
            best_test = test_nll
            best_epoch = epoch
        print(f"[epoch {epoch:03d}/{args.epochs:03d}] train_nll={train_nll:.6f} test_nll={test_nll:.6f}")

    z = torch.randn((args.n_test, 2), device=device)
    gen_x = model.inverse(z).cpu().numpy()
    gen_x = gen_x * std + mu
    test_x_vis = test_x_np * std + mu

    out_path = resolve_project_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(test_x_vis[:, 0], test_x_vis[:, 1], s=3, alpha=0.45, c="#1f77b4")
    axes[0].set_title("Target Moons")
    axes[1].scatter(gen_x[:, 0], gen_x[:, 1], s=3, alpha=0.45, c="#d62728")
    axes[1].set_title("Flow Samples")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"Moons Flow Sanity | blocks={args.num_blocks}, s_max={args.s_max}, best_test_nll={best_test:.3f} (ep {best_epoch})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"saved_figure={out_path}")
    print(f"best_test_nll={best_test:.6f} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
