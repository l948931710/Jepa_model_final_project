from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for processing image observations.
    """
    def __init__(self, in_channels=2, base_channels=32, latent_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Flatten and project to latent space
        self.fc = nn.Linear(base_channels * 8 * 4 * 4, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, x):
        # x shape: [B, C, H, W]
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        latent = self.fc(features)
        latent = self.norm(latent)
        return latent


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


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor that models the dynamics.
    """
    def __init__(self, embed_dim=256, nhead=8, dim_feedforward=1024, num_layers=3):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, state_embeds, action_embeds, mask=None):
        # state_embeds: [B, T, D]
        # action_embeds: [B, T-1, D]
        # Create sequence: [state_0, action_0, state_1, action_1, ...]
        B, T, D = action_embeds.shape
        
        # Prepare for transformer input
        transformer_in = torch.zeros(B, T*2+1, D, device=state_embeds.device)
        # Fill with states and actions in alternating pattern
        transformer_in[:, 0::2, :] = state_embeds
        transformer_in[:, 1::2, :] = action_embeds
        
        # Process with transformer
        transformer_out = self.transformer(transformer_in, src_key_padding_mask=mask)
        
        # Extract only state embeddings (every other position)
        state_preds = transformer_out[:, 0::2, :]
        
        # Apply prediction head to get final predictions
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
        
        # Observation encoder
        self.obs_encoder = ConvEncoder(
            in_channels=2,
            base_channels=base_channels,
            latent_dim=latent_dim
        )
        
        # Action encoder
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=action_embed_dim,
            output_dim=latent_dim
        )
        
        # Dynamics predictor
        self.predictor = TransformerPredictor(
            embed_dim=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=transformer_layers
        )
        
        # Projection for contrastive loss (to prevent collapse)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim)
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
        
        return predicted_embeds
    
    def compute_loss(self, states, actions, target_states=None):
        """
        Compute JEPA loss with contrastive component to prevent collapse.
        
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]
            target_states: [B, T, Ch, H, W] (optional, if None use states)
        
        Returns:
            loss: scalar
            metrics: dict of metrics
        """
        if target_states is None:
            target_states = states
            
        # Encode inputs
        state_embeds = self.encode_observations(states)
        action_embeds = self.encode_actions(actions)
        target_embeds = self.encode_observations(target_states)
        
        # Get predictions
        predicted_embeds = self.predictor(state_embeds, action_embeds)
        
        # Project embeddings for contrastive loss (to prevent collapse)
        proj_predicted = self.projector(predicted_embeds)
        proj_target = self.projector(target_embeds)
        
        # Predictive loss component (L2 distance in embedding space)
        pred_loss = F.mse_loss(predicted_embeds, target_embeds.detach())
        
        # Contrastive loss component to prevent collapse
        # Using InfoNCE/NT-Xent loss
        B, T, D = proj_predicted.shape
        
        # Reshape for contrastive loss calculation
        z_i = proj_predicted.reshape(B*T, D)
        z_j = proj_target.reshape(B*T, D)
        
        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Calculate similarity matrix
        sim_matrix = torch.matmul(z_i, z_j.T) / 0.1  # temperature=0.1
        
        # InfoNCE loss
        labels = torch.arange(B*T, device=self.device)
        contrastive_loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        contrastive_loss = contrastive_loss / 2
        
        # Total loss
        total_loss = pred_loss + 0.1 * contrastive_loss
        
        metrics = {
            "pred_loss": pred_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, metrics

