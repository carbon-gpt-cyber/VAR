import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Encoder1D', 'Decoder1D',
]

# utilities

def Normalize(c, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=c, eps=1e-6, affine=True)

class Upsample1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

class Downsample1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1), mode='constant', value=0))

class ResnetBlock1D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_c = in_channels
        self.out_c = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h

class AttnBlock1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.C = c
        self.norm = Normalize(c)
        self.qkv = nn.Conv1d(c, 3 * c, kernel_size=1, stride=1, padding=0)
        self.w_ratio = c ** -0.5
        self.proj_out = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, L = qkv.shape
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, L).unbind(1)
        q = q.permute(0, 2, 1)  # B,L,C
        w = torch.bmm(q, k) * self.w_ratio
        w = F.softmax(w, dim=2)
        h = torch.bmm(v, w.permute(0, 2, 1))
        return x + self.proj_out(h)

def make_attn1d(c, using_sa=True):
    return AttnBlock1D(c) if using_sa else nn.Identity()

class Encoder1D(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=1,
        z_channels=32, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        self.conv_in = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn1d(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample1D(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn1d(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h

class Decoder1D(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=1,
        z_channels=32, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn1d(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn1d(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample1D(block_in)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h
