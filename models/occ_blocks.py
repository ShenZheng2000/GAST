import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from einops import rearrange

def nonlinearity(x):    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 
    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 
    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 
    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb

def get_normalized_pixel_coords(grid_size):
    h_idx = np.arange(grid_size, dtype=np.float32)
    w_idx = np.arange(grid_size, dtype=np.float32)
    w_grid, h_grid = np.meshgrid(w_idx, h_idx)
    w_norm = w_grid / (grid_size - 1)
    h_norm = h_grid / (grid_size - 1)
    coords = np.stack([w_norm, h_norm], axis=-1)
    return coords

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
    
    def forward(self, x, shape):
        do_rearrange = False
        if x.dim() == 5:    # (B,C,F,H,W)
            do_rearrange = True
            b, _, f, _, _= x.size()
            x = rearrange(x, "b c f h w -> (b f) c h w")
        assert x.dim() == 4
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        diffY = shape[0] - x.size()[2]
        diffX = shape[1] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        if self.with_conv:
            x = self.conv(x)
        if do_rearrange:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
    
    def forward(self, x):
        do_rearrange = False
        if x.dim() == 5:  # (B,C,F,H,W)
            do_rearrange=True
            b, _, f, _, _ = x.size()
            x = rearrange(x, "b c f h w -> (b f) c h w")
        assert x.dim() == 4
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool3d(x, kernel_size=2, stride=2)
        if do_rearrange:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.1, temb_channels=0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)    # b, c, hw
        q = q.permute(0, 2, 1)      # b, hw, c
        k = k.reshape(b, c, h*w)    # b, c, hw
        w_ = torch.bmm(q, k)        # b, hw, hw
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h*w)    # b, c, hw
        w_ = w_.permute(0, 2, 1)    # b, hw, hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)       # b, c, hw (hw of q)
        h_ = h_.reshape(b, c, h, w) # b, c, h, w

        h_ = self.proj_out(h_)
        return x + h_


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q = q.reshape(b * t, c, h * w)  # bt, c, hw
        q = q.permute(0, 2, 1)          # bt, hw, c
        k = k.reshape(b * t, c, h * w)  # bt, c, hw
        w_ = torch.bmm(q, k)            # bt, hw, hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b * t, c, h * w)  # bt, c, hw
        w_ = w_.permute(0, 2, 1)        # bt, hw, hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)           # bt, c, hw (hw of q) 
        h_ = h_.reshape(b, c, t, h, w)  # b, c, t, h, w

        h_ = self.proj_out(h_)
        return x + h_
        

class BEVEncoder2D(nn.Module):
    def __init__(self, *,
                 in_channels, 
                 out_channels, 
                 base_channels, 
                 channels_mult, 
                 num_res_blocks,
                 input_resolution,
                 attn_resolutions, 
                 dropout=0.0, 
                 resamp_with_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.input_resolution = input_resolution
        self.attn_resolutions = attn_resolutions
        self.temb_ch = 0
        
        self.conv_in = nn.Conv2d(
            self.in_channels, self.base_channels, kernel_size=3, stride=1, padding=1)

        cur_resolution = input_resolution
        input_channels_mult = (1,) + tuple(channels_mult)   # (1,1,2,4,8)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_channels * input_channels_mult[i_level]
            block_out = base_channels * channels_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if cur_resolution in attn_resolutions:
                    # print('[*] Enc has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                cur_resolution = cur_resolution // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, self.out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):   # x: (B, C, H, W)
        temb = None
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class BEVDecoder2D(nn.Module):
    def __init__(self, *,
                 in_channels, 
                 out_channels, 
                 base_channels, 
                 num_stages,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_stages = num_stages

        self.conv_in = nn.Conv2d(
            self.in_channels, self.base_channels, kernel_size=3, stride=1, padding=1)

        self.block = nn.ModuleList()
        for idx in range(self.num_stages):
            self.block.append(ResnetBlock(in_channels=base_channels, 
                                          out_channels=base_channels, 
                                          dropout=dropout))

        self.norm_out = Normalize(self.base_channels)
        self.conv_out = nn.Conv2d(
            self.base_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):   # x: (B, C, H, W)
        h = self.conv_in(x)

        for idx in range(self.num_stages):
            h = F.interpolate(h, scale_factor=2)
            h = self.block[idx](h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class OccAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config['model_params']['num_classes']
        self.expansion = config['model_params']['expansion']
        self.class_embeds = nn.Embedding(self.num_classes, self.expansion)
        self.encoder = BEVEncoder2D(**config['model_params']['encoder'])
        self.decoder = BEVDecoder2D(**config['model_params']['decoder'])

    def forward(self, data_dict):
        occ = data_dict['occ'] # (b, f, h, w, d)
        b, f, h, w, d = occ.shape
        # encode
        occ_feats = self.class_embeds(occ) # (b, f, h, w, d, c)
        occ_feats = rearrange(occ_feats, 'b f h w d c -> (b f) (d c) h w') # (b*f, d*c, h, w)
        latent_occ, _ = self.encoder(occ_feats) # (b*f, c1, h1, w1)
        # decode
        logits = self.decoder(latent_occ) # (b*f, d*c, h, w)
        logits = logits.permute(0,2,3,1).reshape(-1, d, self.expansion) # (b*f*h*w, d, c)
        cls_embed = self.class_embeds.weight.T.unsqueeze(0) # (1, expansion, num_classes)
        logits = torch.matmul(logits, cls_embed) # (b*f*h*w, d, num_classes)
        logits = logits.reshape(b, f, h, w, d, self.num_classes).permute(0,1,5,2,3,4) # (b, f, num_classes, h, w, d)
        return logits
    

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim, 
            2 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.conv_candidate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
    
    def forward(self, x, h_prev):
        if h_prev is None:
            B, _, H, W = x.shape
            h_prev = torch.zeros(B, self.hidden_dim, H, W, device=x.device)

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.sigmoid(gates).chunk(2, dim=1)
        combined_reset = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))
        h_next = (1 - update_gate) * h_prev + update_gate * candidate
        return h_next
    

class TemporalBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, return_all=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_all = return_all
        
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 1)
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.gru_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gru_layers.append(
                ConvGRUCell(input_dim=hidden_dim, hidden_dim=hidden_dim)
            )
        
    def forward(self, x):
        B, F, C, H, W = x.shape
        x_seq = x.permute(1, 0, 2, 3, 4)

        if self.return_all:
            res = []

        hidden_states = [None] * self.num_layers
        for t in range(F):
            x_t = x_seq[t]
            h_t = self.input_proj(x_t)
            for i, layer in enumerate(self.gru_layers):  
                hidden_states[i] = layer(h_t, hidden_states[i])
                h_t = hidden_states[i]
            if self.return_all:
                h_res = self.output_proj(h_t)
                res.append(h_res)
        
        if self.return_all:
            res = torch.stack(res, dim=1)
            return res
            
        last_output = hidden_states[-1]
        output = self.output_proj(last_output)
        return output
    

class SpatialBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
            )
        self.layers = nn.ModuleList()
        for _ in range(num_layers-1):
            self.layers.append(
                ResnetBlock(in_channels=hidden_dim*2, out_channels=hidden_dim)
            )

    def forward(self, x_list):
        for idx, conv in enumerate(self.convs):
            x_list[idx] = conv(x_list[idx])
        
        x = x_list[0]
        for idx, layer in enumerate(self.layers):
            x = F.interpolate(x, scale_factor=2)
            x = torch.cat([x, x_list[idx+1]], dim=1)
            x = layer(x)
            
        return x