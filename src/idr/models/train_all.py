"""Unified training and validation pipeline for authentic IO-VNBD models.

Trains:
1. VelocityEstimatorNet (IMU window -> forward velocity)
2. IMUDenoiseNet (noisy IMU -> filtered IMU)

Validates:
- Evaluates on authentic held-out test split (test_data.npz)
- Exports results/velocity_predictions.csv (Rule 23)
- Computes genuine MAE, RMSE, bias, correlation, R^2
- Generates real comparison plot results/figures/velocity_comparison.png

NON-NEGOTIABLE:
- Strict seed=42 (Rule 33)
- Fails loudly if processed data is missing (Rule 2 & 52)
- Zero synthetic data generation
"""

import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import CONFIG, SEED, set_seed, RESULTS_DIR, RESULTS_FIGURES_DIR, MODELS_DIR
from ..io.preprocess import IDRWindowDataset
from .imu_denoise import IMUDenoiseNet
from .velocity_net import VelocityEstimatorNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, target_idx: int) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_vel, batch_imu in loader:
        optimizer.zero_grad()
        target = batch_vel.unsqueeze(1) if target_idx == 0 else batch_imu
        pred = model(batch_x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_x)
    return total_loss / len(loader.dataset)


def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, target_idx: int) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_vel, batch_imu in loader:
            target = batch_vel.unsqueeze(1) if target_idx == 0 else batch_imu
            pred = model(batch_x)
            loss = criterion(pred, target)
            total_loss += loss.item() * len(batch_x)
    return total_loss / len(loader.dataset)


def validate_and_plot_velocity_model(model: nn.Module, test_loader: DataLoader, output_dir: Path) -> dict:
    """Evaluate velocity model on authentic test set, save CSV and plot (Rule 23)."""
    model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch_x, batch_vel, _ in test_loader:
            preds = model(batch_x).squeeze(-1).cpu().numpy()
            targets = batch_vel.cpu().numpy()
            y_pred_all.extend(preds)
            y_true_all.extend(targets)

    y_true = np.array(y_true_all, dtype=np.float64)
    y_pred = np.array(y_pred_all, dtype=np.float64)
    abs_errors = np.abs(y_true - y_pred)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    bias = float(np.mean(y_pred - y_true))

    # Pearson correlation and R2
    if np.std(y_true) > 1e-4 and np.std(y_pred) > 1e-4:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        correlation = 0.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 0.0

    # Save velocity_predictions.csv
    pred_df = pd.DataFrame({
        "sample_index": np.arange(len(y_true)),
        "ground_truth_velocity": y_true,
        "predicted_velocity": y_pred,
        "absolute_error": abs_errors,
    })
    csv_path = RESULTS_DIR / "velocity_predictions.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"Saved velocity predictions to {csv_path}")

    # Generate genuine comparison plot
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RESULTS_FIGURES_DIR / "velocity_comparison.png"

    plt.figure(figsize=(12, 5))
    # Subplot 1: Timeseries snippet (first 300 samples)
    plt.subplot(1, 2, 1)
    N_plot = min(300, len(y_true))
    plt.plot(np.arange(N_plot), y_true[:N_plot], label="Ground Truth (ECU CAN)", color="#1f77b4", linewidth=1.5)
    plt.plot(np.arange(N_plot), y_pred[:N_plot], label="AI Velocity (IMU Window)", color="#ff7f0e", linestyle="--", linewidth=1.5)
    plt.xlabel("Test Window Sample Index")
    plt.ylabel("Forward Velocity (m/s)")
    plt.title(f"Velocity Tracking (MAE={mae:.2f} m/s, RMSE={rmse:.2f} m/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Parity scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true[::5], y_pred[::5], alpha=0.3, s=10, color="#2ca02c")
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)), 10.0)
    plt.plot([0, max_v], [0, max_v], "r--", label="Ideal Parity", alpha=0.8)
    plt.xlabel("Ground Truth Velocity (m/s)")
    plt.ylabel("Predicted Velocity (m/s)")
    plt.title(f"Parity Scatter (r = {correlation:.3f}, R² = {r2:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved velocity comparison figure to {plot_path}")

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "correlation": round(correlation, 4),
        "r2": round(r2, 4),
        "num_test_samples": len(y_true),
    }
    return metrics


def train_pipeline(data_dir: Path, output_dir: Path, epochs: int = 10):
    # Rule 33: Consistent deterministic seed
    set_seed(SEED)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_npz = data_dir / "train_data.npz"
    val_npz = data_dir / "val_data.npz"
    test_npz = data_dir / "test_data.npz"

    # Rule 2 & 52: No synthetic fallback - fail loudly
    if not train_npz.exists() or not val_npz.exists() or not test_npz.exists():
        raise FileNotFoundError(
            f"FATAL: Required authentic dataset processed files missing in {data_dir}.\n"
            f"Expected: {train_npz}, {val_npz}, {test_npz}.\n"
            "Run 'py -3.11 -m idr.io.preprocess' first. No synthetic fallback allowed."
        )

    logger.info(f"Loading processed authentic data from {data_dir}...")
    train_data = np.load(train_npz)
    val_data = np.load(val_npz)
    test_data = np.load(test_npz)

    X_train, y_vel_train, y_imu_train = train_data["windows"], train_data["targets_vel"], train_data["targets_imu"]
    X_val, y_vel_val, y_imu_val = val_data["windows"], val_data["targets_vel"], val_data["targets_imu"]
    X_test, y_vel_test, y_imu_test = test_data["windows"], test_data["targets_vel"], test_data["targets_imu"]

    logger.info(f"Train set: {len(X_train)} windows | Val set: {len(X_val)} windows | Test set: {len(X_test)} windows")

    train_ds = IDRWindowDataset(X_train, y_vel_train, y_imu_train)
    val_ds = IDRWindowDataset(X_val, y_vel_val, y_imu_val)
    test_ds = IDRWindowDataset(X_test, y_vel_test, y_imu_test)

    batch_size = CONFIG["model"].batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 1. Train Velocity Estimator
    logger.info("=== Training Forward Velocity Estimator on Authentic Data ===")
    vel_model = VelocityEstimatorNet()
    optimizer_v = torch.optim.AdamW(vel_model.parameters(), lr=CONFIG["model"].learning_rate, weight_decay=1e-4)
    criterion_v = nn.HuberLoss()

    best_v_loss = float("inf")
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(vel_model, train_loader, optimizer_v, criterion_v, target_idx=0)
        va_loss = eval_epoch(vel_model, val_loader, criterion_v, target_idx=0)
        if va_loss < best_v_loss:
            best_v_loss = va_loss
            torch.save(vel_model.state_dict(), output_dir / "velocity_net.pt")
        logger.info(f"[VelocityNet] Epoch {epoch:02d}/{epochs:02d} - Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

    # Load best checkpoint for validation
    vel_model.load_state_dict(torch.load(output_dir / "velocity_net.pt"))
    vel_metrics = validate_and_plot_velocity_model(vel_model, test_loader, output_dir)
    logger.info(f"VelocityNet Test Metrics: {vel_metrics}")

    # 2. Train IMU Denoise Net (Rule 24)
    logger.info("=== Training IMU Denoise Net on Authentic Data ===")
    denoise_model = IMUDenoiseNet()
    optimizer_d = torch.optim.AdamW(denoise_model.parameters(), lr=CONFIG["model"].learning_rate, weight_decay=1e-4)
    criterion_d = nn.MSELoss()

    best_d_loss = float("inf")
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(denoise_model, train_loader, optimizer_d, criterion_d, target_idx=1)
        va_loss = eval_epoch(denoise_model, val_loader, criterion_d, target_idx=1)
        if va_loss < best_d_loss:
            best_d_loss = va_loss
            torch.save(denoise_model.state_dict(), output_dir / "imu_denoise_net.pt")
        logger.info(f"[DenoiseNet]  Epoch {epoch:02d}/{epochs:02d} - Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

    logger.info(f"Training completed. Checkpoints saved to {output_dir}")
    return vel_metrics


def main():
    parser = argparse.ArgumentParser(description="Train IDR models.")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train_pipeline(Path(args.data_dir), Path(args.output_dir), epochs=args.epochs)


if __name__ == "__main__":
    main()
