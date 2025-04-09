import os
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import datetime
import json
import torch.multiprocessing as mp

# 在脚本开始时设置多进程启动方法为'spawn'
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# 导入自定义模块
from models import JEPAModel, compute_jepa_contrastive_loss, compute_vicreg_loss, compute_barlow_twins_loss, compute_auxiliary_loss
from normalizer import Normalizer
from schedulers import Scheduler, LRSchedule
from dataset import create_wall_dataloader  # Import the dataset module

#########################
# 模型和训练参数配置
#########################

# 模型架构参数
LATENT_DIM = 256            # 潜在空间维度
ACTION_DIM = 2              # 动作维度（固定为2，x和y方向）
ACTION_EMBED_DIM = 128      # 动作嵌入维度
BASE_CHANNELS = 32          # 卷积基础通道数
TRANSFORMER_LAYERS = 3      # Transformer层数
NHEAD = 8                   # 注意力头数
DIM_FEEDFORWARD = 1024      # 前馈网络维度

# 训练参数
CONTEXT_FRAMES = 5          # 上下文帧数
TARGET_FRAMES = 5           # 目标预测帧数
BATCH_SIZE = 64             # 批次大小
EPOCHS = 10                 # 训练轮数
LR = 1e-4                   # 学习率
WEIGHT_DECAY = 1e-4         # 权重衰减
GRAD_CLIP = 1.0             # 梯度裁剪

# 损失函数配置
LOSS_TYPE = "contrastive"   # 损失函数类型：'contrastive', 'vicreg', 'barlow'

# 对比损失(InfoNCE)参数
CONTRASTIVE_WEIGHT = 0.1    # 对比损失权重
TEMPERATURE = 0.1           # 对比损失温度参数

# VicReg损失参数
SIM_WEIGHT = 25.0           # 相似性损失权重
VAR_WEIGHT = 25.0           # 方差损失权重
COV_WEIGHT = 1.0            # 协方差损失权重

# Barlow Twins损失参数
LAMBDA_PARAM = 0.005        # 冗余降低系数

# 数据设置
DATA_PATH = "/scratch/DL25SP/train"  # 训练数据路径
OUTPUT_DIR = "./checkpoints"         # 输出目录
SAVE_EVERY = 10                      # 每多少个epoch保存一次模型
PROBING = False                      # 是否使用定位信息

# 数据增强设置
AUGMENTATION = False                 # 是否使用数据增强
AUG_PROBABILITY = 0.0               # 应用每种增强的概率

# 学习率调度
SCHEDULE = "Cosine"                  # 学习率调度方式："Constant" 或 "Cosine"

# 其他设置
SEED = 42                            # 随机种子
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_jepa():
    """
    Train the JEPA model using the parameters defined at the top of this script.
    """
    # Set seed for reproducibility
    set_seed(SEED)
    
    print(f"Using device: {DEVICE}")
    print(f"Using loss type: {LOSS_TYPE}")
    
    # Create timestamp for experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_DIR, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save parameter configuration for reference
    config = {
        "latent_dim": LATENT_DIM,
        "action_dim": ACTION_DIM,
        "action_embed_dim": ACTION_EMBED_DIM,
        "base_channels": BASE_CHANNELS,
        "transformer_layers": TRANSFORMER_LAYERS,
        "nhead": NHEAD,
        "dim_feedforward": DIM_FEEDFORWARD,
        "context_frames": CONTEXT_FRAMES,
        "target_frames": TARGET_FRAMES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "loss_type": LOSS_TYPE,
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "temperature": TEMPERATURE,
        "sim_weight": SIM_WEIGHT,
        "var_weight": VAR_WEIGHT, 
        "cov_weight": COV_WEIGHT,
        "lambda_param": LAMBDA_PARAM,
        "data_path": DATA_PATH,
        "schedule": SCHEDULE,
        "probing": PROBING,
        "augmentation": AUGMENTATION,
        "aug_probability": AUG_PROBABILITY,
        "seed": SEED,
        "device": DEVICE
    }
    
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {config_path}")
    
    # Create dataloader using the imported function
    dataloader = create_wall_dataloader(
        data_path=DATA_PATH,
        probing=PROBING,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        train=True,
        augmentation=AUGMENTATION,
        aug_probability=AUG_PROBABILITY
    )
    
    # Initialize model
    model = JEPAModel(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        action_embed_dim=ACTION_EMBED_DIM,
        base_channels=BASE_CHANNELS,
        transformer_layers=TRANSFORMER_LAYERS,
        nhead=NHEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        device=DEVICE
    )
    model = model.to(DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    # Set base_lr for each parameter group (needed for scheduler)
    for param_group in optimizer.param_groups:
        param_group['base_lr'] = LR
    
    # 初始化学习率调度器
    scheduler = Scheduler(
        schedule=getattr(LRSchedule, SCHEDULE),
        base_lr=LR,
        data_loader=dataloader,
        epochs=EPOCHS,
        optimizer=optimizer,
        batch_steps=len(dataloader),
        batch_size=BATCH_SIZE
    )
    
    # Training loop
    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'learning_rate': []
    }
    
    # 根据损失函数类型初始化额外的指标跟踪字段
    if LOSS_TYPE == "contrastive":
        metrics_history.update({
            'pred_loss': [],
            'contrastive_loss': [],
            'consistency_loss': []
        })
    elif LOSS_TYPE == "vicreg":
        metrics_history.update({
            'sim_loss': [],
            'var_loss': [],
            'cov_loss': [],
            'consistency_loss': []
        })
    elif LOSS_TYPE == "barlow":
        metrics_history.update({
            'invariance_loss': [],
            'redundancy_reduction_loss': [],
            'mse_loss': [],
            'consistency_loss': []
        })
    
    # Normalizer for metrics
    normalizer = Normalizer()
    
    step = 0
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        epoch_metric_dict = {}  # 用于存储各种损失类型的指标
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for batch in pbar:
                # Process data from the WallSample namedtuple
                # 创建可写的副本以避免不可写张量的警告
                states = batch.states.clone()
                actions = batch.actions.clone()
                
                # Update learning rate
                lr = scheduler.adjust_learning_rate(step)
                step += 1
                
                # Forward pass
                optimizer.zero_grad()
                
                # 编码状态和动作
                state_embeds = model.encode_observations(states)
                action_embeds = model.encode_actions(actions)
                
                # 预测嵌入和辅助任务输出
                predicted_embeds, wall_door_pred, spatial_consistency = model(states, actions)
                
                # 获取目标嵌入
                target_embeds = model.encode_observations(states[:, :predicted_embeds.shape[1]])
                
                # 根据损失类型选择不同的损失函数
                if LOSS_TYPE == "contrastive":
                    # 使用InfoNCE对比损失
                    loss, metrics = compute_jepa_contrastive_loss(
                        predicted_embeds, 
                        target_embeds,
                        temperature=TEMPERATURE,
                        contrastive_weight=CONTRASTIVE_WEIGHT,
                        device=DEVICE
                    )
                    
                    # 添加辅助损失
                    aux_loss, aux_metrics = compute_auxiliary_loss(
                        predicted_embeds,
                        target_embeds,
                        wall_door_pred,
                        spatial_consistency
                    )
                    
                    # 合并损失
                    total_loss = loss + 0.1 * aux_loss
                    metrics.update(aux_metrics)
                    metrics['total_loss'] = total_loss.item()
                    
                    pbar_metrics = {
                        'loss': f"{metrics['total_loss']:.4f}",
                        'pred': f"{metrics['pred_loss']:.4f}",
                        'contr': f"{metrics['contrastive_loss']:.4f}",
                        'consist': f"{metrics['consistency_loss']:.4f}"
                    }
                elif LOSS_TYPE == "vicreg":
                    # 使用VicReg损失
                    loss, metrics = compute_vicreg_loss(
                        predicted_embeds, 
                        target_embeds,
                        sim_weight=SIM_WEIGHT,
                        var_weight=VAR_WEIGHT,
                        cov_weight=COV_WEIGHT
                    )
                    
                    # 添加辅助损失
                    aux_loss, aux_metrics = compute_auxiliary_loss(
                        predicted_embeds,
                        target_embeds,
                        wall_door_pred,
                        spatial_consistency
                    )
                    
                    # 合并损失
                    total_loss = loss + 0.1 * aux_loss
                    metrics.update(aux_metrics)
                    metrics['total_loss'] = total_loss.item()
                    
                    pbar_metrics = {
                        'loss': f"{metrics['total_loss']:.4f}",
                        'sim': f"{metrics['sim_loss']:.4f}",
                        'var': f"{metrics['var_loss']:.4f}",
                        'cov': f"{metrics['cov_loss']:.4f}",
                        'consist': f"{metrics['consistency_loss']:.4f}"
                    }
                elif LOSS_TYPE == "barlow":
                    # 使用Barlow Twins损失
                    loss, metrics = compute_barlow_twins_loss(
                        predicted_embeds, 
                        target_embeds,
                        lambda_param=LAMBDA_PARAM
                    )
                    
                    # 添加辅助损失
                    aux_loss, aux_metrics = compute_auxiliary_loss(
                        predicted_embeds,
                        target_embeds,
                        wall_door_pred,
                        spatial_consistency
                    )
                    
                    # 合并损失
                    total_loss = loss + 0.1 * aux_loss
                    metrics.update(aux_metrics)
                    metrics['total_loss'] = total_loss.item()
                    
                    pbar_metrics = {
                        'loss': f"{metrics['total_loss']:.4f}",
                        'inv': f"{metrics['invariance_loss']:.4f}",
                        'red': f"{metrics['redundancy_reduction_loss']:.4f}",
                        'mse': f"{metrics['mse_loss']:.4f}",
                        'consist': f"{metrics['consistency_loss']:.4f}"
                    }
                else:
                    raise ValueError(f"Unknown loss type: {LOSS_TYPE}")
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                
                # Update parameters
                optimizer.step()
                
                # Log metrics
                epoch_losses.append(metrics['total_loss'])
                
                # 为每种损失类型记录指标
                for k, v in metrics.items():
                    if k not in epoch_metric_dict:
                        epoch_metric_dict[k] = []
                    epoch_metric_dict[k].append(v)
                
                # 添加学习率到进度条显示
                pbar_metrics['lr'] = f"{lr:.6f}"
                
                # Update progress bar
                pbar.set_postfix(**pbar_metrics)
        
        # Compute epoch metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # 计算所有指标的平均值
        avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metric_dict.items()}
        
        # 保存当前epoch的指标
        metrics_history['epoch'].append(epoch + 1)
        metrics_history['train_loss'].append(avg_loss)
        metrics_history['learning_rate'].append(lr)
        
        # 保存特定损失函数的指标
        for k, v in avg_metrics.items():
            if k != 'total_loss' and k in metrics_history:
                metrics_history[k].append(v)
        
        # 构建日志消息
        log_parts = [f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}"]
        for k, v in avg_metrics.items():
            if k != 'total_loss':
                log_parts.append(f"{k}: {v:.4f}")
        log_parts.append(f"LR: {lr:.6f}")
        
        print(", ".join(log_parts))
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(exp_dir, "jepa_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_type': LOSS_TYPE,
                'model_config': {
                    'latent_dim': LATENT_DIM,
                    'action_dim': ACTION_DIM,
                    'action_embed_dim': ACTION_EMBED_DIM,
                    'base_channels': BASE_CHANNELS,
                    'transformer_layers': TRANSFORMER_LAYERS,
                    'nhead': NHEAD,
                    'dim_feedforward': DIM_FEEDFORWARD,
                }
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            checkpoint_path = os.path.join(exp_dir, f"jepa_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_type': LOSS_TYPE,
                'model_config': {
                    'latent_dim': LATENT_DIM,
                    'action_dim': ACTION_DIM,
                    'action_embed_dim': ACTION_EMBED_DIM,
                    'base_channels': BASE_CHANNELS,
                    'transformer_layers': TRANSFORMER_LAYERS,
                    'nhead': NHEAD,
                    'dim_feedforward': DIM_FEEDFORWARD,
                }
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(exp_dir, "jepa_final.pt")
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': metrics_history['train_loss'][-1],
        'loss_type': LOSS_TYPE,
        'model_config': {
            'latent_dim': LATENT_DIM,
            'action_dim': ACTION_DIM,
            'action_embed_dim': ACTION_EMBED_DIM,
            'base_channels': BASE_CHANNELS,
            'transformer_layers': TRANSFORMER_LAYERS,
            'nhead': NHEAD,
            'dim_feedforward': DIM_FEEDFORWARD,
        }
    }, final_path)
    
    # 同时保存到标准路径，方便 main.py 加载
    std_final_path = os.path.join(OUTPUT_DIR, "jepa_final.pt")
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': metrics_history['train_loss'][-1],
        'loss_type': LOSS_TYPE,
        'model_config': {
            'latent_dim': LATENT_DIM,
            'action_dim': ACTION_DIM,
            'action_embed_dim': ACTION_EMBED_DIM,
            'base_channels': BASE_CHANNELS,
            'transformer_layers': TRANSFORMER_LAYERS,
            'nhead': NHEAD,
            'dim_feedforward': DIM_FEEDFORWARD,
        }
    }, std_final_path)
    
    print(f"Saved final model to {final_path}")
    print(f"Also saved to standard path {std_final_path} for easy evaluation")
    
    # Save training metrics
    metrics_path = os.path.join(exp_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"Training completed. All outputs saved to {exp_dir}")
    
    return model, exp_dir


if __name__ == "__main__":
    # 设置多进程启动方法为'spawn'
    mp.set_start_method('spawn', force=True)
    
    # 训练模型
    model, exp_dir = train_jepa()