from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import math


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv1(x))
        return x * attention


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for processing image observations.
    """
    def __init__(self, in_channels=2, base_channels=32, latent_dim=256):
        super().__init__()
        
        # 增加基础通道数
        base_channels = 64
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            SpatialAttention(base_channels)
        )
        
        # 残差块1
        self.res1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            SpatialAttention(base_channels * 2)
        )
        
        # 残差块2
        self.res2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2)
        )
        
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SpatialAttention(base_channels * 4)
        )
        
        # 残差块3
        self.res3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4)
        )
        
        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            SpatialAttention(base_channels * 8)
        )
        
        # 残差块4
        self.res4 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8)
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 投影到潜在空间
        self.fc = nn.Linear(base_channels * 8, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # 第一个卷积块
        x1 = self.conv1(x)
        x1 = x1 + self.res1(x1)  # 残差连接
        
        # 第二个卷积块
        x2 = self.conv2(x1)
        x2 = x2 + self.res2(x2)  # 残差连接
        
        # 第三个卷积块
        x3 = self.conv3(x2)
        x3 = x3 + self.res3(x3)  # 残差连接
        
        # 第四个卷积块
        x4 = self.conv4(x3)
        x4 = x4 + self.res4(x4)  # 残差连接
        
        # 全局平均池化
        x = self.global_pool(x4)
        x = x.view(x.size(0), -1)
        
        # 投影到潜在空间
        x = self.fc(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class ActionEncoder(nn.Module):
    """
    MLP encoder for processing action sequences.
    """
    def __init__(self, action_dim=2, hidden_dim=128, output_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, action):
        # action shape: [B, action_dim]
        return self.net(action)


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SpatialTransformerLayer(nn.Module):
    """空间Transformer层"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        
        # 前馈网络
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor that models the dynamics.
    """
    def __init__(self, embed_dim=256, nhead=8, dim_feedforward=1024, num_layers=3):
        super().__init__()
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # 空间Transformer层
        self.spatial_layers = nn.ModuleList([
            SpatialTransformerLayer(embed_dim, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        # 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        """使用Xavier初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, state_embeds, action_embeds, mask=None):
        # state_embeds: [B, T, D]
        # action_embeds: [B, T-1, D]
        
        # 创建序列：[state_0, action_0, state_1, action_1, ...]
        B, T, D = action_embeds.shape
        
        # 准备transformer输入
        transformer_in = torch.zeros(B, T*2+1, D, device=state_embeds.device)
        transformer_in[:, 0::2, :] = state_embeds
        transformer_in[:, 1::2, :] = action_embeds
        
        # 添加位置编码
        transformer_in = self.pos_encoder(transformer_in)
        
        # 通过空间Transformer层
        for layer in self.spatial_layers:
            transformer_in = layer(transformer_in, src_key_padding_mask=mask)
        
        # 提取状态嵌入（每隔一个位置）
        state_preds = transformer_in[:, 0::2, :]
        
        # 应用预测头
        predicted_embeds = self.pred_head(state_preds)
        
        return predicted_embeds


class JEPAModel(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA) model.
    """
    def __init__(
        self,
        latent_dim=256,
        action_dim=2,
        action_embed_dim=128,
        base_channels=32,
        transformer_layers=3,
        nhead=8,
        dim_feedforward=1024,
        device="cuda"
    ):
        super().__init__()
        
        self.device = device
        self.repr_dim = latent_dim
        
        # 观察编码器
        self.obs_encoder = ConvEncoder(
            in_channels=2,
            base_channels=base_channels,
            latent_dim=latent_dim
        )
        
        # 动作编码器
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=action_embed_dim,
            output_dim=latent_dim
        )
        
        # 动态预测器
        self.predictor = TransformerPredictor(
            embed_dim=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=transformer_layers
        )
        
        # 投影头（用于对比损失）
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # 辅助任务：预测墙和门的位置
        self.wall_door_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 4)  # 预测4个值：墙的起点和终点坐标
        )
        
        # 空间一致性预测器
        self.spatial_consistency = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
    
    def encode_observations(self, states):
        """
        Encode batch of observations.
        
        Args:
            states: [B, T, C, H, W]
        
        Returns:
            state_embeds: [B, T, D]
        """
        B, T, C, H, W = states.shape
        
        # Reshape for encoding
        flat_states = states.reshape(B * T, C, H, W)
        flat_embeds = self.obs_encoder(flat_states)
        
        # Reshape back
        state_embeds = flat_embeds.reshape(B, T, -1)
        
        return state_embeds
    
    def encode_actions(self, actions):
        """
        Encode batch of actions.
        
        Args:
            actions: [B, T, 2]
        
        Returns:
            action_embeds: [B, T, D]
        """
        B, T, A = actions.shape
        
        # Reshape for encoding
        flat_actions = actions.reshape(B * T, A)
        flat_embeds = self.action_encoder(flat_actions)
        
        # Reshape back
        action_embeds = flat_embeds.reshape(B, T, -1)
        
        return action_embeds
    
    def forward(self, states, actions):
        """
        Forward pass for the JEPA model.
        
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]
        
        Returns:
            predictions: [B, T, D]
        """
        # Encode observations and actions
        state_embeds = self.encode_observations(states)
        action_embeds = self.encode_actions(actions)
        
        # Predict future states
        predicted_embeds = self.predictor(state_embeds, action_embeds)
        if not self.training and states.shape[1] == 1 and actions.shape[1] > 1:
            # 评估模式：只返回预测嵌入，无需辅助任务
            return predicted_embeds, None, None
        # 预测墙和门的位置
        wall_door_pred = self.wall_door_predictor(predicted_embeds)
        
        # 计算空间一致性
        spatial_consistency = self.spatial_consistency(
            torch.cat([state_embeds, predicted_embeds], dim=-1)
        )
        
        return predicted_embeds, wall_door_pred, spatial_consistency


# 以下是三种损失函数的独立实现

def compute_jepa_contrastive_loss(predicted_embeds, target_embeds, temperature=0.1, contrastive_weight=0.1, device="cuda"):
    """
    InfoNCE风格的对比损失实现，用于JEPA训练。
    
    Args:
        predicted_embeds: 模型预测的嵌入 [B, T, D] 或 [B*T, D]
        target_embeds: 目标嵌入 [B, T, D] 或 [B*T, D]
        temperature: 温度参数，控制分布的平滑度
        contrastive_weight: 对比损失的权重
        device: 计算设备
    
    Returns:
        total_loss: 总损失标量
        metrics: 包含各损失组件的字典
    """
    # 确保输入形状统一
    if len(predicted_embeds.shape) == 3:
        B, T, D = predicted_embeds.shape
        # 重塑为[B*T, D]
        predicted_embeds_flat = predicted_embeds.reshape(B*T, D)
        target_embeds_flat = target_embeds.reshape(B*T, D)
    else:
        predicted_embeds_flat = predicted_embeds
        target_embeds_flat = target_embeds
    
    # 预测损失组件（嵌入空间中的L2距离）
    pred_loss = F.mse_loss(predicted_embeds_flat, target_embeds_flat.detach())
    
    # 对比损失组件
    # 归一化嵌入
    z_i = F.normalize(predicted_embeds_flat, dim=1)
    z_j = F.normalize(target_embeds_flat, dim=1)
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(z_i, z_j.T) / temperature
    
    # InfoNCE损失
    labels = torch.arange(z_i.shape[0], device=device)
    contrastive_loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
    contrastive_loss = contrastive_loss / 2
    
    # 总损失
    total_loss = pred_loss + contrastive_weight * contrastive_loss
    
    metrics = {
        "pred_loss": pred_loss.item(),
        "contrastive_loss": contrastive_loss.item(),
        "total_loss": total_loss.item()
    }
    
    return total_loss, metrics


def off_diagonal(x):
    """
    返回方阵的非对角线元素。
    用于VicReg和Barlow Twins损失计算。
    
    Args:
        x: 方阵 [D, D]
    
    Returns:
        非对角线元素 [D*(D-1)]
    """
    n = x.shape[0]
    return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()


def compute_vicreg_loss(predicted_embeds, target_embeds, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """
    VicReg损失实现，通过三个正则化项防止表示坍塌。
    
    Args:
        predicted_embeds: 模型预测的嵌入 [B, T, D] 或 [B*T, D]
        target_embeds: 目标嵌入 [B, T, D] 或 [B*T, D]
        sim_weight: 不变性损失权重
        var_weight: 方差损失权重
        cov_weight: 协方差损失权重
    
    Returns:
        total_loss: 总损失标量
        metrics: 包含各损失组件的字典
    """
    # 确保输入形状统一
    if len(predicted_embeds.shape) == 3:
        B, T, D = predicted_embeds.shape
        # 重塑为[B*T, D]
        predicted_embeds = predicted_embeds.reshape(B*T, D)
        target_embeds = target_embeds.reshape(B*T, D)
    
    # 不变性损失（相当于预测损失）
    sim_loss = F.mse_loss(predicted_embeds, target_embeds)
    
    # 方差损失 - 鼓励嵌入多样性
    std_pred = torch.sqrt(predicted_embeds.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(1 - std_pred))
    
    # 协方差损失 - 解相关维度
    pred_centered = predicted_embeds - predicted_embeds.mean(dim=0)
    cov_pred = (pred_centered.T @ pred_centered) / (predicted_embeds.shape[0] - 1)
    cov_loss = off_diagonal(cov_pred).pow_(2).sum() / predicted_embeds.shape[1]
    
    # 总损失
    total_loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    
    metrics = {
        "sim_loss": sim_loss.item(),
        "var_loss": var_loss.item(),
        "cov_loss": cov_loss.item(),
        "total_loss": total_loss.item()
    }
    
    return total_loss, metrics


def compute_barlow_twins_loss(predicted_embeds, target_embeds, lambda_param=0.005):
    """
    Barlow Twins损失实现，通过交叉相关矩阵解相关防止表示坍塌。
    
    Args:
        predicted_embeds: 模型预测的嵌入 [B, T, D] 或 [B*T, D]
        target_embeds: 目标嵌入 [B, T, D] 或 [B*T, D]
        lambda_param: 非对角线项的惩罚系数
    
    Returns:
        total_loss: 总损失标量
        metrics: 包含各损失组件的字典
    """
    # 确保输入形状统一
    if len(predicted_embeds.shape) == 3:
        B, T, D = predicted_embeds.shape
        # 重塑为[B*T, D]
        predicted_embeds = predicted_embeds.reshape(B*T, D)
        target_embeds = target_embeds.reshape(B*T, D)
    
    # 归一化表示沿批次维度
    pred_norm = (predicted_embeds - predicted_embeds.mean(0)) / predicted_embeds.std(0)
    target_norm = (target_embeds - target_embeds.mean(0)) / target_embeds.std(0)
    
    # 交叉相关矩阵
    N = predicted_embeds.shape[0]  # 批次大小
    c = torch.matmul(pred_norm.T, target_norm) / N
    
    # Barlow Twins损失
    # 对角线项应接近1（完全相关）
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    
    # 非对角线项应接近0（完全不相关）
    off_diag = off_diagonal(c).pow_(2).sum()
    
    # 对角线损失（嵌入空间中的不变性）
    invariance_loss = on_diag
    
    # 非对角线损失（表示维度解相关）
    redundancy_reduction_loss = lambda_param * off_diag
    
    # 总损失
    loss = invariance_loss + redundancy_reduction_loss
    
    # MSE损失用于度量预测质量
    mse_loss = F.mse_loss(predicted_embeds, target_embeds)
    
    metrics = {
        "invariance_loss": invariance_loss.item(),
        "redundancy_reduction_loss": redundancy_reduction_loss.item(),
        "barlow_loss": loss.item(),
        "mse_loss": mse_loss.item(),
        "total_loss": (loss + mse_loss).item()
    }
    
    # 这里我们结合Barlow Twins损失和MSE损失
    # 这样既防止坍塌又保证预测质量
    total_loss = loss + mse_loss
    
    return total_loss, metrics


def compute_auxiliary_loss(predicted_embeds, target_embeds, wall_door_pred, spatial_consistency):
    """
    计算辅助损失，包括墙和门的位置预测损失以及空间一致性损失。
    """
    # 计算预测损失
    pred_loss = F.mse_loss(predicted_embeds, target_embeds.detach())
    
    # 计算空间一致性损失
    consistency_loss = F.binary_cross_entropy(
        spatial_consistency,
        torch.ones_like(spatial_consistency)
    )
    
    # 计算总损失
    total_loss = pred_loss + 0.1 * consistency_loss
    
    metrics = {
        "pred_loss": pred_loss.item(),
        "consistency_loss": consistency_loss.item(),
        "total_loss": total_loss.item()
    }
    
    return total_loss, metrics
class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)