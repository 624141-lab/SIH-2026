"""Candidate A: Learned Velocity Residual Network (VelocityResidualNet).

Instead of attempting to infer absolute highway speed directly from chassis vibration,
this network conditions on the mechanized INS velocity (integrated from IMU since outage start)
and uses the recent IMU window dynamics to estimate the velocity drift error:
    Δv = v_true - v_INS
along with aleatoric uncertainty log(σ_v^2).

The corrected velocity pseudo-measurement is then:
    v_corrected = v_INS + Δv,  with R_v = exp(log_var_v).
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityResidualNet(nn.Module):
    """Deep 1D CNN-GRU network predicting velocity residual Δv and uncertainty.
    
    Inputs:
        imu_window: (B, 6, W) 6-axis IMU window (acc_xyz, gyro_xyz)
        v_ins: (B, 1) current mechanized INS forward speed
    Outputs:
        (B, 2): [delta_v, log_var_v]
    """

    def __init__(self, in_channels: int = 6, window_size: int = 50, hidden_dim: int = 64):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # Temporal feature extractor
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, dilation=2, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()

        # Recurrent layer for sequential aggregation
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Fusion head: combines IMU temporal feature vector + current v_ins
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # [delta_v, log_var_v]
        )

    def forward(self, imu_window: torch.Tensor, v_ins: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            imu_window: (B, 6, W) or (B, W, 6)
            v_ins: (B, 1) or (B,)
        Returns:
            (B, 2): [delta_v, log_var_v]
        """
        if imu_window.dim() == 3 and imu_window.shape[1] != 6 and imu_window.shape[2] == 6:
            imu_window = imu_window.transpose(1, 2)

        if v_ins.dim() == 1:
            v_ins = v_ins.unsqueeze(-1)

        x = self.act(self.bn1(self.conv1(imu_window)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))

        # GRU expects (B, W, C)
        x_gru, _ = self.gru(x.transpose(1, 2))
        last_hidden = x_gru[:, -1, :]  # (B, hidden_dim)

        fused = torch.cat([last_hidden, v_ins], dim=-1)
        out = self.head(fused)

        delta_v = out[:, :1]
        log_var = torch.clamp(out[:, 1:], min=-4.0, max=4.0)
        return torch.cat([delta_v, log_var], dim=-1)


def residual_nll_loss(pred: torch.Tensor, target_residual: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL loss for residual velocity."""
    delta_v_pred = pred[:, :1]
    log_var = pred[:, 1:]
    precision = torch.exp(-log_var)
    diff_sq = (target_residual - delta_v_pred) ** 2
    return 0.5 * torch.mean(precision * diff_sq + log_var)


# Backward compatibility alias
ResidualDriftNet = VelocityResidualNet

