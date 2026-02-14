import torch
from torch import nn

class MapEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, channels=(32, 48, 64, 96, 128), norm: str = "bn"):
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
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(channels[-1] * 4 * 4, latent_dim)

    def load_pretrained_backbone(self, ckpt_path: str, strict: bool = True):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        enc_state = ckpt["enc_backbone"]

        state = {}
        for k, v in enc_state.items():
            if k.startswith("backbone."):
                state[k[len("backbone."):]] = v

        missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
        print("[MapEncoder] loaded:", ckpt_path)
        print("[MapEncoder] missing:", missing)
        print("[MapEncoder] unexpected:", unexpected)

    def forward(self, binary_map: torch.Tensor) -> torch.Tensor:
        x = self.backbone(binary_map)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class CondEncoder(nn.Module):
    def __init__(
        self,
        sg_dim: int = 2,
        latent_dim: int = 128,
        channels=(32, 48, 64, 96, 128),
        norm: str = "bn",
        pretrained_backbone: str | None = None,
    ):
        super().__init__()

        self.map_encoder = MapEncoder(latent_dim, channels, norm)

        if pretrained_backbone is not None:
            self.map_encoder.load_pretrained_backbone(pretrained_backbone, strict=True)

        self.sg_dim = sg_dim

    def forward(self, map: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        w = self.map_encoder(map)
        return torch.cat([w, start, goal], dim=-1)
