import os
import glob
import math
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# -------------------------
# 1) 거리맵(SDF-like) 생성
# -------------------------

def sdf_distance_transform_scipy(obs01: np.ndarray) -> np.ndarray:
    """
    obs01: (H,W) obstacle=1, free=0
    returns: (H,W) distance to obstacle (float32), 0 on obstacle
    """
    from scipy.ndimage import distance_transform_edt
    free = (obs01 == 0).astype(np.uint8)
    dist = distance_transform_edt(free).astype(np.float32)
    return dist

def sdf_distance_transform_bfs(obs01: np.ndarray) -> np.ndarray:
    """
    SciPy 없을 때 fallback: multi-source BFS로 Manhattan distance 계산
    obs01: (H,W) obstacle=1, free=0
    returns: (H,W) dist (float32), 0 on obstacle
    """
    H, W = obs01.shape
    INF = 10**9
    dist = np.full((H, W), INF, dtype=np.int32)

    # obstacle pixels are sources (dist=0)
    qy, qx = np.where(obs01 == 1)
    from collections import deque
    dq = deque()
    for y, x in zip(qy, qx):
        dist[y, x] = 0
        dq.append((y, x))

    # if no obstacle at all, return large constant distance map
    if len(dq) == 0:
        return np.full((H, W), float(max(H, W)), dtype=np.float32)

    # 4-neighbor BFS
    neigh = [(1,0), (-1,0), (0,1), (0,-1)]
    while dq:
        y, x = dq.popleft()
        d = dist[y, x] + 1
        for dy, dx in neigh:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and d < dist[ny, nx]:
                dist[ny, nx] = d
                dq.append((ny, nx))

    return dist.astype(np.float32)

def compute_sdf(obs01: np.ndarray) -> np.ndarray:
    """
    obs01: (H,W) obstacle=1, free=0
    """
    try:
        return sdf_distance_transform_scipy(obs01)
    except Exception:
        return sdf_distance_transform_bfs(obs01)

def normalize_sdf(dist: np.ndarray, clip_max: float = 30.0) -> np.ndarray:
    """
    dist: (H,W) float32
    clip + scale to [0,1]
    """
    dist = np.clip(dist, 0.0, clip_max)
    dist = dist / clip_max
    return dist.astype(np.float32)


# -------------------------
# 2) Dataset: map.png -> sdf_target
# -------------------------

class SDFPretrainDataset(Dataset):
    def __init__(self, data_root: str, map_glob: str = "**/map.png", clip_max: float = 30.0):
        super().__init__()
        self.paths = sorted(glob.glob(os.path.join(data_root, map_glob), recursive=True))
        if len(self.paths) == 0:
            raise RuntimeError(f"No map images found under {data_root} with glob {map_glob}")
        self.clip_max = clip_max

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("L")
        arr = np.array(img).astype(np.float32)

        # map assumed 0/255 (free/obstacle) OR 0/1; make obs01 (1=obstacle)
        if arr.max() > 1.5:
            arr01 = (arr / 255.0)
        else:
            arr01 = arr
        obs01 = (arr01 > 0.5).astype(np.uint8)  # obstacle=1

        dist = compute_sdf(obs01)                # distance to obstacle
        sdf = normalize_sdf(dist, self.clip_max) # [0,1]

        # input map as free/obstacle in [0,1] (keep original semantics)
        map_t = torch.from_numpy(arr01).unsqueeze(0)       # (1,H,W)
        sdf_t = torch.from_numpy(sdf).unsqueeze(0)         # (1,H,W)

        return map_t, sdf_t


# -------------------------
# 3) Model: reuse your backbone, add a light head
# -------------------------

class MapEncoderBackbone(nn.Module):
    def __init__(self, channels=(32, 48, 64, 96, 128), norm="gn"):
        super().__init__()

        def Norm(c: int):
            if norm == "bn":
                return nn.BatchNorm2d(c)
            if norm == "gn":
                g = min(8, c)
                while c % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, c)
            if norm == "none":
                return nn.Identity()
            raise ValueError("norm must be 'bn', 'gn', or 'none'.")

        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                Norm(out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                Norm(out_ch),
                nn.SiLU(),
            ]
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)  # (B,C,h,w)

class SDFHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_ch, 1, 1, 1, 0),
        )

    def forward(self, feat, out_hw):
        logits_small = self.conv(feat)  # (B,1,h,w)
        logits = F.interpolate(logits_small, size=out_hw, mode="bilinear", align_corners=False)
        return logits

class SDFPretrainNet(nn.Module):
    def __init__(self, channels=(32,48,64,96,128), norm="gn"):
        super().__init__()
        self.enc = MapEncoderBackbone(channels=channels, norm=norm)
        self.head = SDFHead(in_ch=channels[-1])

    def forward(self, map01):
        B, _, H, W = map01.shape
        feat = self.enc(map01)
        logits = self.head(feat, (H, W))  # logits for sdf in [0,1]
        return logits


# -------------------------
# 4) Train loop
# -------------------------

@dataclass
class TrainCfg:
    data_root: str = "data/train"
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    clip_max: float = 30.0
    channels: tuple = (32, 48, 64, 96, 128)
    norm: str = "gn"
    grad_clip: float = 1.0
    out_dir: str = "outputs/pretrain"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    cfg = TrainCfg()

    os.makedirs(cfg.out_dir, exist_ok=True)

    ds = SDFPretrainDataset(cfg.data_root, clip_max=cfg.clip_max)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    model = SDFPretrainNet(channels=cfg.channels, norm=cfg.norm).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # regression on [0,1] sdf: SmoothL1 is stable
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    best = float("inf")
    for ep in range(1, cfg.epochs + 1):
        ep_start = time.time()
        model.train()
        run = 0.0
        n = 0
        pbar = tqdm(dl, desc=f"ep {ep:03d}/{cfg.epochs:03d}", leave=False)
        for map_t, sdf_t in pbar:
            map_t = map_t.to(cfg.device, non_blocking=True).float()
            sdf_t = sdf_t.to(cfg.device, non_blocking=True).float()

            logits = model(map_t)
            pred = torch.sigmoid(logits)  # ensure [0,1]

            loss = loss_fn(pred, sdf_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            run += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(run / n):.4f}")

        avg = run / max(1, n)
        ep_sec = time.time() - ep_start
        is_best = avg < best
        tag = " *best*" if is_best else ""
        print(
            f"[ep {ep:03d}/{cfg.epochs:03d}] "
            f"loss={avg:.6f} best={min(best, avg):.6f} "
            f"steps={n:04d} time={ep_sec:.1f}s{tag}"
        )

        # save best
        if is_best:
            best = avg
            torch.save(
                {
                    "enc_backbone": model.enc.state_dict(),
                    "channels": cfg.channels,
                    "norm": cfg.norm,
                    "clip_max": cfg.clip_max,
                },
                os.path.join(cfg.out_dir, "mapencoder_backbone_best.pt")
            )

    # save last too
    torch.save(
        {"enc_backbone": model.enc.state_dict(), "channels": cfg.channels, "norm": cfg.norm, "clip_max": cfg.clip_max},
        os.path.join(cfg.out_dir, "mapencoder_backbone_last.pt")
    )

if __name__ == "__main__":
    main()
