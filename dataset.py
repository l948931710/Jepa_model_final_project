from typing import NamedTuple, Optional
import torch
import numpy as np


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
        train=True,
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

        self.train = train

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        # 确保NumPy数组是可写的
        states = np.asarray(self.states[i], dtype=np.float32)
        actions = np.asarray(self.actions[i], dtype=np.float32)
        
        # 转换为PyTorch张量
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        
        # 处理位置数据
        if self.locations is not None:
            locations = np.asarray(self.locations[i], dtype=np.float32)
            locations = torch.from_numpy(locations).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)
        
        # 如果是训练模式，应用数据增强
        if self.train:
            states = apply_augmentations(states)
        
        return WallSample(states=states, locations=locations, actions=actions)


def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
    augmentation=True,
    aug_probability=0.5
):
    """创建用于墙环境的数据加载器"""
    # 创建数据集
    dataset = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
        train=train
    )
    
    # 设置应用增强的概率
    global p
    p = aug_probability if augmentation and train else 0.0
    
    # 创建数据加载器
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0  # 注意：如果使用多进程，需要确保数据的正确转移到GPU
    )
    
    return loader
# 全局概率参数
p = 0.0

# 图像增强函数
def apply_augmentations(states, p=None):
    """
    对状态序列应用图像增强
    
    Args:
        states: 形状为[T, C, H, W]的状态张量
        p: 应用每种增强的概率，如果为None则使用全局概率
    
    Returns:
        增强后的状态张量
    """
    import random
    # 使用全局概率或传入的概率
    prob = p if p is not None else globals()['p']
    
    # 如果概率为0，则不进行增强
    if prob <= 0:
        return states
    
    # 复制状态以避免修改原始数据
    augmented_states = states.clone()
    
    # 添加高斯噪声
    if random.random() < prob:
        noise = torch.randn_like(augmented_states) * 0.01
        augmented_states = torch.clamp(augmented_states + noise, 0, 1)
    
    # 调整亮度
    if random.random() < prob:
        brightness_factor = random.uniform(-0.1, 0.1)
        augmented_states = torch.clamp(augmented_states + brightness_factor, 0, 1)
    
    # 随机擦除
    if random.random() < prob:
        T, C, H, W = augmented_states.shape
        for t in range(T):
            if random.random() < prob:
                # 确定擦除区域大小
                erase_size = random.uniform(0.02, 0.1)
                area = H * W * erase_size
                aspect_ratio = random.uniform(0.5, 1.5)
                
                erase_h = int(np.sqrt(area / aspect_ratio))
                erase_w = int(np.sqrt(area * aspect_ratio))
                
                if erase_h < H and erase_w < W:
                    i = random.randint(0, H - erase_h)
                    j = random.randint(0, W - erase_w)
                    
                    # 随机值填充擦除区域
                    random_value = random.random()
                    augmented_states[t, :, i:i+erase_h, j:j+erase_w] = random_value
    
    return augmented_states
