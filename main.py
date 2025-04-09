from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
import glob
import os
from models import JEPAModel
def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL25SP"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }

    return probe_train_ds, probe_val_ds


def load_model(model_path, device="cuda"):
    """加载训练好的JEPA模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    # 加载checkpoint
    print(f"从{model_path}加载模型")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从checkpoint获取模型配置
    model_config = checkpoint.get('model_config', {
        'latent_dim': 256,
        'action_dim': 2,
        'action_embed_dim': 128,
        'base_channels': 32,
        'transformer_layers': 3,
        'nhead': 8,
        'dim_feedforward': 1024,
    })
    
    # 实例化模型
    model = JEPAModel(
        latent_dim=model_config.get('latent_dim', 256),
        action_dim=model_config.get('action_dim', 2),
        action_embed_dim=model_config.get('action_embed_dim', 128),
        base_channels=model_config.get('base_channels', 32),
        transformer_layers=model_config.get('transformer_layers', 3),
        nhead=model_config.get('nhead', 8),
        dim_feedforward=model_config.get('dim_feedforward', 1024),
        device=device
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 打印模型信息
    print(f"加载了模型 (Epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"损失类型: {checkpoint.get('loss_type', 'unknown')}")
    print(f"损失值: {checkpoint.get('loss', 'unknown')}")
    
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    model = load_model("./checkpoints/experiment_20250409_041828/jepa_best.pt", device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    probe_train_ds, probe_val_ds = load_data(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
