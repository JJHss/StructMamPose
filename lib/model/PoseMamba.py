## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt
import os
import sys
# 获取当前工作目录
current_directory = os.path.dirname(__file__) + '/../' + '../'
sys.path.append(current_directory)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

from lib.model.mambablocks_my import BiSTSSMBlock
class  PoseMamba(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=6, mlp_ratio=2., drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth
        self.STEblocks = nn.ModuleList([
           BiSTSSMBlock(
                hidden_dim = embed_dim_ratio, 
                mlp_ratio = mlp_ratio, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                forward_type='v2_plus_poselimbs'
                )
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
           BiSTSSMBlock(
                hidden_dim = embed_dim, 
                mlp_ratio = mlp_ratio, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                forward_type='v2_plus_poselimbs'
                )
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head_1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

        self.head_2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )
        self.proj_side = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.cross_attn = CrossAttention(embed_dim=embed_dim, num_heads=1, dropout=0)
        
    def get_side_view_90(self, p_3d):
        root = p_3d[:, :, 0:1, :]  # [B, T, 1, 3]
        p_rel = p_3d - root  # [B, T, N, 3]
        y_rel = p_rel[..., 1]  # 原来的垂直坐标 (在新视角下依然是垂直坐标)
        z_rel = p_rel[..., 2]  # 原来的深度坐标 (在新视角下变成了水平坐标)
        x_side = z_rel
        y_side = y_rel
        x_side_2d = torch.stack([x_side, y_side], dim=-1)  # [B, T, N, 2]

        return x_side_2d
    
    
        
    def STE_forward(self, x):
        b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, '(b f) n c  -> b f n c', f=f)
        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        return x

    def TTE_foward(self, x):
        # assert len(x.shape) == 3, "shape is equal to 3"
        b, f, n, c  = x.shape
        x = rearrange(x, 'b f n cw -> (b n) f cw', f=f)
        x += self.Temporal_pos_embed[:,:f,:]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        blk = self.TTEblocks[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def ST1_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, 9):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = tteblock(x)
            x = self.Temporal_norm(x)
        return x

    def ST2_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(9, self.block_depth):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = tteblock(x)
            x = self.Temporal_norm(x)
        return x

    def forward(self, x):
        b, f, n, c = x.shape
        x = self.STE_forward(x)
        x = self.TTE_foward(x)
        x = self.ST1_foward(x)
        o1 = self.head_1(x)
        x_side = self.proj_side(self.get_side_view_90(o1))
        x = x + self.cross_attn(x, x_side) 
        x = self.ST2_foward(x)
        
        #x_side = x_side + self.cross_attn(x_side, x)
        #x = self.ST2_foward(x+x_side)
        o2 = (o1 + self.head_2(x)).view(b, f, n, -1)
        
        return o2
    
'''  
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_joints=17, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 1. 空间压缩层：将 17 个节点压缩为 1 个
        # 这里的权重代表了“哪些关节对判断整体动作更重要”
        self.spatial_reducer = nn.Linear(num_joints, 1)

        # 2. 注意力层：现在的输入维度仅仅是 embed_dim (C)，而不是 17*C
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, y, attn_mask=None):
        """
        x: [B, T, N, C] (正视图)
        y: [B, T, N, C] (侧视图)
        """
        B, T, N, C = x.shape

        # =======================================================
        # 步骤 1: 空间压缩 (N -> 1)
        # 我们需要在 N 维度上进行线性变换
        # =======================================================
        # 变换维度以便 Linear 作用于 N: [B, T, C, N]
        x_perm = x.permute(0, 1, 3, 2) 
        y_perm = y.permute(0, 1, 3, 2)

        # 压缩: [B, T, C, N] -> [B, T, C, 1]
        x_global = self.spatial_reducer(x_perm).squeeze(-1) # [B, T, C]
        y_global = self.spatial_reducer(y_perm).squeeze(-1) # [B, T, C]

        # =======================================================
        # 步骤 2: 标准 Cross Attention (在帧级别计算)
        # 输入形状现在是 [B, T, C]，非常轻量
        # =======================================================
        q = self.q_proj(x_global).view(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(y_global).view(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(y_global).view(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Attention Map: [B, Heads, T, T]
        # 计算正视图第 i 帧与侧视图第 j 帧的整体相似度
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / ((C // self.num_heads) ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 聚合: [B, Heads, T, Head_Dim]
        attn_out = torch.matmul(attn_weights, v)
        
        # 还原: [B, T, C]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, C)
        attn_out = self.out_proj(attn_out) # [B, T, C]

        # =======================================================
        # 步骤 3: 广播并残差连接 (1 -> N)
        # 将学习到的“全局帧信息”加回到每一个节点上
        # =======================================================
        
        # 扩展维度: [B, T, C] -> [B, T, 1, C]
        attn_out_expanded = attn_out.unsqueeze(2) 
        
        # 这里的残差连接非常关键！
        # 我们把提取出的全局侧视图信息，加到原始的正视图节点特征上
        # 利用广播机制，attn_out_expanded 会自动复制 N 次以匹配 x
        output = self.norm(x + attn_out_expanded)

        return output  
'''

'''
大参数
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 线性变换层输入维度为 17 * C，这符合将一帧内17个节点拉平为整体特征的设计
        self.q_proj = nn.Linear(17 * embed_dim, 17 * embed_dim)
        self.k_proj = nn.Linear(17 * embed_dim, 17 * embed_dim)
        self.v_proj = nn.Linear(17 * embed_dim, 17 * embed_dim)
        self.out_proj = nn.Linear(17 * embed_dim, 17 * embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, y, attn_mask=None):
        """
        x: [B, T, N, C] (正视图 - Query)
        y: [B, T, N, C] (侧视图 - Key/Value)
        目标：以帧为Token，计算 [B, Heads, T, T] 的注意力图
        """
        batch_size, t_len, n_len, embed_dim = x.shape

        # =======================================================
        # 修改点 1: 形状变换
        # 将 N 和 C 维度合并，作为特征维度。保持 T 维度独立作为序列长度。
        # 形状变为: [B, T, N*C]
        # =======================================================
        x_flat = x.flatten(2)  # [B, T, N*C]
        y_flat = y.flatten(2)  # [B, T, N*C]

        # 线性变换 [B, T, N*C]
        q = self.q_proj(x_flat)
        k = self.k_proj(y_flat)
        v = self.v_proj(y_flat)

        # =======================================================
        # 修改点 2: 多头拆分
        # 注意：这里的特征总维度是 N*C。
        # 拆分时，每个头的维度应该是 (N*C) / num_heads = N * head_dim
        # 形状变为: [B, T, num_heads, N*head_dim]
        # =======================================================
        real_head_dim = (n_len * embed_dim) // self.num_heads
        
        q = q.view(batch_size, t_len, self.num_heads, real_head_dim)
        k = k.view(batch_size, t_len, self.num_heads, real_head_dim)
        v = v.view(batch_size, t_len, self.num_heads, real_head_dim)

        # 调整维度顺序 -> [B, num_heads, T, real_head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # =======================================================
        # 修改点 3: 计算注意力
        # 结果形状: [B, num_heads, T, T]
        # 含义: 正视图的第 i 帧与侧视图的第 j 帧之间的相似度
        # =======================================================
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (real_head_dim ** 0.5)

        if attn_mask is not None:
            # 如果使用 mask，mask 形状应为 [B, 1, T, T] 或 [T, T]
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 聚合 Value -> [B, num_heads, T, real_head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # =======================================================
        # 修改点 4: 还原维度
        # [B, num_heads, T, real_head_dim] -> [B, T, num_heads, real_head_dim]
        # -> [B, T, N*C]
        # =======================================================
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, t_len, n_len * embed_dim)
        
        # 最终线性变换
        attn_output = self.out_proj(attn_output)

        # 还原回 [B, T, N, C] 以便后续层处理
        attn_output = attn_output.view(batch_size, t_len, n_len, embed_dim)

        return self.norm(attn_output)
'''

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 注意：这里的 head_dim 计算方式需要改变思路
        # 我们希望最终的 attention 还是在 [B, T, N*C] 这个级别上做
        # 所以每个头的维度应该是 (N * C) / num_heads
        self.total_dim = 17 * embed_dim
        self.head_dim = self.total_dim // num_heads
        
        assert self.total_dim % num_heads == 0, "17 * embed_dim must be divisible by num_heads"

        # =======================================================
        # 优化点：修改线性层定义
        # 原来: nn.Linear(17*C, 17*C) -> 参数量巨大
        # 现在: nn.Linear(C, C) -> 参数量极小，所有节点共享权重
        # =======================================================
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim) # 输出层也改小，最后再做

        # 这里的 out_proj 需要特殊处理，因为 Attention 输出是 [B, T, 17*C]
        # 如果为了极致省参数，我们先 reshape 回 [B, T, N, C] 再过线性层
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, y, attn_mask=None):
        """
        x: [B, T, N, C]
        y: [B, T, N, C]
        """
        batch_size, t_len, n_len, embed_dim = x.shape

        # =======================================================
        # 步骤 1: 先进行投影 (在 N, C 维度上进行，而不是 flatten 后)
        # x: [B, T, N, C] -> q: [B, T, N, C]
        # 这样避免了 (N*C) x (N*C) 的巨大矩阵
        # =======================================================
        q = self.q_proj(x) 
        k = self.k_proj(y)
        v = self.v_proj(y)

        # =======================================================
        # 步骤 2: 拉平并分头
        # [B, T, N, C] -> [B, T, N*C] -> [B, T, num_heads, head_dim]
        # =======================================================
        q = q.flatten(2).view(batch_size, t_len, self.num_heads, self.head_dim)
        k = k.flatten(2).view(batch_size, t_len, self.num_heads, self.head_dim)
        v = v.flatten(2).view(batch_size, t_len, self.num_heads, self.head_dim)

        # [B, num_heads, T, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # =======================================================
        # 步骤 3: 计算 Attention (保持不变)
        # 结果仍是 [B, num_heads, T, T]，全帧对全帧
        # =======================================================
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [B, num_heads, T, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # =======================================================
        # 步骤 4: 还原维度与输出投影
        # =======================================================
        # 还原为 [B, T, N*C]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, t_len, n_len, embed_dim) # 变回 [B, T, N, C]
        
        # 输出层投影：同样使用共享权重的线性层
        # 输入 [B, T, N, C] -> 输出 [B, T, N, C]
        attn_output = self.out_proj(attn_output)

        # 残差连接 (推荐加上，防止原始信息丢失)
        return self.norm(attn_output)    
        
        
        
if __name__ == "__main__":
    torch.cuda.set_device(3)
    model = PoseMamba(num_frame=243, embed_dim_ratio=128,mlp_ratio = 2, depth = 10).cuda()
    from thop import profile, clever_format
    input_shape = (1, 243, 17, 2)
    x = torch.randn(input_shape).cuda()
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))