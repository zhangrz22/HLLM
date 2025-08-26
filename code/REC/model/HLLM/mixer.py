import torch
import numpy as np
from torch import nn
# from einops.layers.torch import Rearrange as einops_rearrange
# import einops.layers.torch as einops
import einops

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, seq_lens, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        
        self.token_norm = nn.LayerNorm(dim)
        self.token_forward = FeedForward(seq_lens, token_dim, dropout)

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        residual = x
        x = einops.rearrange(self.token_norm(x), 'b n d -> b d n')
        x = residual + einops.rearrange(self.token_forward(x), 'b d n -> b n d')
        
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):

    def __init__(self, dim, seq_lens, num_classes, depth, token_dim, channel_dim):
        super().__init__()
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, seq_lens, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)


if __name__ == "__main__":
    img = torch.ones([2, 2, 128])
    print(img.size())
    model = MLPMixer(num_classes=1, seq_lens=2, dim=128, depth=2, token_dim=16, channel_dim=128*3)
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    out_img = model(img)
    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
