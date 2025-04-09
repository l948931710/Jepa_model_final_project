import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import json
import torch.multiprocessing as mp

# 在脚本开始时设置多进程启动方法为'spawn'
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# 导入自定义模块
from models import JEPAModel
from normalizer import Normalizer
from schedulers import Scheduler, LRSchedule

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
EPOCHS = 50                 # 训练轮数
LR = 1e-4                   # 学习率
WEIGHT_DECAY = 1e-4         # 权重衰减
GRAD_CLIP = 1.0             # 梯度裁剪

# 损失权重
CONTRASTIVE_WEIGHT = 0.1    # 对比损失权重
TEMPERATURE = 0.1           # 对比损失温度参数

# 数据设置
DATA_PATH = "/scratch/DL25SP/train"  # 训练数据路径
OUTPUT_DIR = "./checkpoints"         # 输出目录
SAVE_EVERY = 10                      # 每多少个epoch保存一次模型
AUGMENT = True                       # 是否使用数据增强

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


class MemoryEfficientTrajectoryDataset(Dataset):
    """Memory efficient dataset for trajectory data, keeping data on CPU."""
    
    def __init__(self, states, actions, context_frames=5, target_frames=5, augment=False):
        """
        Args:
            states: Array of states [num_trajectories, trajectory_length, channels, height, width]
            actions: Array of actions [num_trajectories, trajectory_length-1, action_dim]
            context_frames: Number of frames to use as context
            target_frames: Number of future frames to predict
            augment: Whether to use data augmentation
        """
        # Keep original arrays in CPU memory
        self.states = states  # numpy array
        self.actions = actions  # numpy array
        self.context_frames = context_frames
        self.target_frames = target_frames
        self.augment = augment
        
        # Initialize normalizer
        self.normalizer = Normalizer()
        
        # Calculate valid starting indices
        self.num_trajectories = states.shape[0]
        self.trajectory_length = states.shape[1]
        self.valid_starts = []
        
        for traj_idx in range(self.num_trajectories):
            max_start = self.trajectory_length - (context_frames + target_frames)
            for start_idx in range(max_start + 1):  # +1 to include the last valid start
                self.valid_starts.append((traj_idx, start_idx))
        
        print(f"Created dataset with {len(self.valid_starts)} valid sequences")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        traj_idx, start_idx = self.valid_starts[idx]
        
        end_idx = start_idx + self.context_frames + self.target_frames
        
        # Get states and actions for this sequence (keep on CPU first)
        states_seq = torch.tensor(self.states[traj_idx, start_idx:end_idx], dtype=torch.float32)
        actions_seq = torch.tensor(self.actions[traj_idx, start_idx:end_idx-1], dtype=torch.float32)
        
        # Normalize actions (which are agent position deltas)
        device_backup = self.normalizer.location_mean.device
        self.normalizer.location_mean = self.normalizer.location_mean.cpu()
        self.normalizer.location_std = self.normalizer.location_std.cpu()
        normalized_actions = self.normalizer.normalize_location(actions_seq)
        self.normalizer.location_mean = self.normalizer.location_mean.to(device_backup)
        self.normalizer.location_std = self.normalizer.location_std.to(device_backup)
        
        # Split into context and target
        context_states = states_seq[:self.context_frames]
        target_states = states_seq[self.context_frames:]
        
        context_actions = normalized_actions[:self.context_frames-1]
        target_actions = normalized_actions[self.context_frames-1:]
        
        # Apply augmentations if enabled
        if self.augment:
            context_states, target_states = self._augment(context_states, target_states)
        
        return {
            'context_states': context_states,
            'context_actions': context_actions,
            'target_states': target_states,
            'target_actions': target_actions,
        }
    
    def _augment(self, context_states, target_states):
        """Apply data augmentation to states."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            context_states = torch.flip(context_states, dims=[-1])
            target_states = torch.flip(target_states, dims=[-1])
        
        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            context_states = torch.flip(context_states, dims=[-2])
            target_states = torch.flip(target_states, dims=[-2])
        
        # Random rotation (90, 180, 270 degrees)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            context_states = torch.rot90(context_states, k, dims=[-2, -1])
            target_states = torch.rot90(target_states, k, dims=[-2, -1])
        
        # Random brightness adjustment
        brightness = 1.0 + 0.2 * (torch.rand(1).item() - 0.5)
        context_states = torch.clamp(context_states * brightness, 0, 1)
        target_states = torch.clamp(target_states * brightness, 0, 1)
        
        return context_states, target_states


def load_data(data_path):
    """Load trajectory data, keeping it on CPU."""
    print(f"Loading data from {data_path}...")
    
    state_path = os.path.join(data_path, "states.npy")
    action_path = os.path.join(data_path, "actions.npy")
    
    print(f"Loading states from {state_path}")
    states = np.load(state_path, mmap_mode='r')  # Memory-mapped mode
    print(f"States loaded: {states.shape}")
    
    print(f"Loading actions from {action_path}")
    actions = np.load(action_path, mmap_mode='r')  # Memory-mapped mode
    print(f"Actions loaded: {actions.shape}")
    
    return states, actions


def collate_fn(batch):
    """将数据移动到GPU的自定义collate函数"""
    # 合并batch中的所有字典
    batch_dict = {
        'context_states': [],
        'context_actions': [],
        'target_states': [],
        'target_actions': []
    }
    
    for sample in batch:
        for key in batch_dict:
            batch_dict[key].append(sample[key])
    
    # 将列表转换为张量
    for key in batch_dict:
        batch_dict[key] = torch.stack(batch_dict[key])
        # 移动到GPU
        batch_dict[key] = batch_dict[key].to(DEVICE)
    
    return batch_dict


def train_jepa():
    """
    Train the JEPA model using the parameters defined at the top of this script.
    """
    # Set seed for reproducibility
    set_seed(SEED)
    
    print(f"Using device: {DEVICE}")
    
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
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "temperature": TEMPERATURE,
        "data_path": DATA_PATH,
        "schedule": SCHEDULE,
        "augment": AUGMENT,
        "seed": SEED,
        "device": DEVICE
    }
    
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {config_path}")
    
    # Load data (keeping it on CPU)
    states, actions = load_data(DATA_PATH)
    
    # Create memory-efficient dataset
    dataset = MemoryEfficientTrajectoryDataset(
        states=states, 
        actions=actions,
        context_frames=CONTEXT_FRAMES,
        target_frames=TARGET_FRAMES,
        augment=AUGMENT
    )
    
    # 减少工作进程数以避免内存问题，并使用自定义的collate_fn在批处理级别上将数据移动到GPU
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # 减少工作线程数
        collate_fn=collate_fn  # 使用自定义collate函数
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
    
    # Set contrastive loss parameters if they exist in the model
    if hasattr(model, 'contrastive_weight'):
        model.contrastive_weight = CONTRASTIVE_WEIGHT
    if hasattr(model, 'temperature'):
        model.temperature = TEMPERATURE
    
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
    
    # 初始化学习率调度器，明确提供batch_steps和batch_size
    scheduler = Scheduler(
        schedule=getattr(LRSchedule, SCHEDULE),
        base_lr=LR,
        data_loader=dataloader,
        epochs=EPOCHS,
        optimizer=optimizer,
        batch_steps=len(dataloader),  # 明确提供batch_steps
        batch_size=BATCH_SIZE  # 明确提供batch_size
    )
    
    # Training loop
    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'pred_loss': [],
        'contrastive_loss': [],
        'learning_rate': []
    }
    
    # Normalizer for metrics
    normalizer = Normalizer()
    
    step = 0
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        epoch_pred_losses = []
        epoch_contrast_losses = []
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for batch in pbar:
                # Prepare data - 数据已经在collate_fn中移动到GPU
                context_states = batch['context_states']
                context_actions = batch['context_actions']
                target_states = batch['target_states']
                target_actions = batch['target_actions']
                
                # Combine context and target for full sequence
                states = torch.cat([context_states, target_states], dim=1)
                actions = torch.cat([context_actions, target_actions], dim=1)
                
                # Update learning rate
                lr = scheduler.adjust_learning_rate(step)
                step += 1
                
                # Forward pass and loss calculation
                optimizer.zero_grad()
                loss, metrics = model.compute_loss(states, actions)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                
                # Update parameters
                optimizer.step()
                
                # Log metrics
                epoch_losses.append(metrics['total_loss'])
                epoch_pred_losses.append(metrics['pred_loss'])
                epoch_contrast_losses.append(metrics['contrastive_loss'])
                
                # Update progress bar
                pbar.set_postfix(
                    loss=f"{metrics['total_loss']:.4f}",
                    pred_loss=f"{metrics['pred_loss']:.4f}",
                    contrast_loss=f"{metrics['contrastive_loss']:.4f}",
                    lr=f"{lr:.6f}"
                )
        
        # Compute epoch metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_pred_loss = sum(epoch_pred_losses) / len(epoch_pred_losses)
        avg_contrast_loss = sum(epoch_contrast_losses) / len(epoch_contrast_losses)
        
        # Save metrics
        metrics_history['epoch'].append(epoch + 1)
        metrics_history['train_loss'].append(avg_loss)
        metrics_history['pred_loss'].append(avg_pred_loss)
        metrics_history['contrastive_loss'].append(avg_contrast_loss)
        metrics_history['learning_rate'].append(lr)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}, Pred Loss: {avg_pred_loss:.4f}, Contrast Loss: {avg_contrast_loss:.4f}, LR: {lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(exp_dir, "jepa_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
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
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Plot loss curves
    plt.subplot(2, 1, 1)
    plt.plot(metrics_history['epoch'], metrics_history['train_loss'], label='Total Loss')
    plt.plot(metrics_history['epoch'], metrics_history['pred_loss'], label='Prediction Loss')
    plt.plot(metrics_history['epoch'], metrics_history['contrastive_loss'], label='Contrastive Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate curve
    plt.subplot(2, 1, 2)
    plt.plot(metrics_history['epoch'], metrics_history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'training_curves.png'))
    
    print(f"Training completed. All outputs saved to {exp_dir}")
    
    return model, exp_dir


if __name__ == "__main__":
    # 设置多进程启动方法为'spawn'
    mp.set_start_method('spawn', force=True)
    
    # 训练模型
    model, exp_dir = train_jepa()