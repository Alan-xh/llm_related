"""
Task ID        : SEQ-GEN-CGAN-001
Task Name      : Conditional Time-Series Sequence Generation via CGAN / WGAN-GP
Domain         : Time-Series / Generative Modeling / Sequential Analysis
Architecture   : Conditional Generative Adversarial Network with WGAN-GP Penalty
Reference      : - Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets.
                 - Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN.
                 - Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs.

Core Concept & Mechanism:
    This module implements a Conditional Generative Adversarial Network (CGAN) tailored for
    multi-variate time-series generation. The Generator conditions on temporal features (cond)
    and a global latent noise vector (z) to synthesize realistic temporal feature sequences.
    The Discriminator evaluates sequence authenticity given the exact conditioning input.
    Training stability is enforced via Wasserstein distance with Gradient Penalty (WGAN-GP).

Mathematical Formulations:
    1. Wasserstein GAN Objective with Gradient Penalty (WGAN-GP):
       min_G max_D  E_{x~P_r}[D(x|c)] - E_{\hat{x}~P_g}[D(\hat{x}|c)] - \lambda E_{\tilde{x}~P_{\tilde{x}}}[(||\nabla_{\tilde{x}} D(\tilde{x}|c)||_2 - 1)^2]
       where \tilde{x} = \epsilon x + (1 - \epsilon) \hat{x} for \epsilon ~ U(0, 1).

    2. Generator Initial Hidden State Projection:
       h_0 = MLP([z, c_0]),  c_0 = 0
       LSTM Mapping: h_t, c_t = LSTM(c_t, (h_{t-1}, c_{t-1}))
       Output Projection: x_t = Tanh(MLP(h_t))

Data Input / Output Specification:
    Real Data Tensor (x)     : Shape [B, L, D_in]   - Continuous time-series sequences.
    Condition Tensor (c)     : Shape [B, L, D_cond] - Exogenous temporal driving features.
    Latent Noise Tensor (z)  : Shape [B, D_z]      - Standard normal noise vectors ~ N(0, I).
    Generated Data Output    : Shape [B, L, D_in]   - Synthesized sequences aligned with conditions.
    Discriminator Score Output: Shape [B, 1]         - Scalar score (unbounded logit for WGAN-GP).
"""

import os
import math
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ==============================================================================
# 3. Hyperparameters & Global Configuration
# ==============================================================================

class CGANConfig:
    """
    Global Configuration class for Conditional Time-Series GAN.
    Encapsulates data, model architecture, training loop, and logging parameters.
    """
    def __init__(self):
        # Data Parameters
        self.data_path: str = "./data/train_data.npy"
        self.cond_data_path: str = "./data/conditions.npy"
        self.seq_length: int = 50       # Sequence length L
        self.input_dim: int = 10        # Target feature dimension D_in
        self.cond_dim: int = 5          # Condition feature dimension D_cond

        # Generator Architecture
        self.latent_dim: int = 100      # Latent noise dimension D_z
        self.gen_hidden_dim: int = 256  # Generator LSTM hidden dimension
        self.gen_num_layers: int = 3    # Number of LSTM layers in Generator

        # Discriminator Architecture
        self.dis_hidden_dim: int = 256  # Discriminator LSTM hidden dimension
        self.dis_num_layers: int = 3    # Number of Bidirectional LSTM layers
        self.dis_dropout: float = 0.2   # Dropout rate for Discriminator

        # Training Hyperparameters
        self.batch_size: int = 64
        self.lr_g: float = 2e-4         # Learning rate for Generator
        self.lr_d: float = 2e-4         # Learning rate for Discriminator
        self.beta1: float = 0.5         # Adam beta1 hyperparameter
        self.beta2: float = 0.9         # Adam beta2 hyperparameter
        self.weight_decay: float = 1e-5
        self.num_epochs: int = 200
        self.n_critic: int = 1          # Number of D updates per G update
        self.gp_weight: float = 10.0    # WGAN-GP Gradient Penalty coefficient lambda
        self.label_smoothing: float = 0.1 # Label smoothing factor for real targets

        # Execution Environment
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Logging and Persistence
        self.save_dir: str = "./checkpoints_cgan"
        self.log_dir: str = "./logs_cgan"
        self.save_interval: int = 10


# ==============================================================================
# 4. Data Processing & Dataset Pipeline
# ==============================================================================

class ConditionalDataset(Dataset):
    """
    Sliding-window Dataset for Conditional Sequential Data.

    Inputs:
        data_path (str): Path to target time-series numpy array [N_samples, D_in].
        cond_path (str): Path to condition time-series numpy array [N_samples, D_cond].
        seq_length (int): Sliding window temporal depth (L).
    """
    def __init__(self, data_path: str, cond_path: str, seq_length: int = 50):
        super().__init__()
        self.seq_length = seq_length

        if os.path.exists(data_path) and os.path.exists(cond_path):
            self.data = np.load(data_path)
            self.conditions = np.load(cond_path)
        else:
            # Synthetic Data Generation for standalone execution
            logging.warning(f"Data paths not found. Generating synthetic dummy data at runtime.")
            num_samples = 1000
            self.data = np.sin(np.linspace(0, 100, num_samples)[:, None] + np.arange(10)[None, :]).astype(np.float32)
            self.conditions = np.cos(np.linspace(0, 100, num_samples)[:, None] + np.arange(5)[None, :]).astype(np.float32)

        assert len(self.data) == len(self.conditions), (
            f"Mismatch in temporal length: data ({len(self.data)}) vs conditions ({len(self.conditions)})"
        )
        self.num_windows = len(self.data) - self.seq_length + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts a sequence window of length `seq_length`.

        Outputs:
            sequence (Tensor)  : Real temporal features, shape: [L, D_in]
            condition (Tensor) : Temporal conditioning features, shape: [L, D_cond]
        """
        sequence = self.data[idx : idx + self.seq_length]
        condition = self.conditions[idx : idx + self.seq_length]
        return torch.from_numpy(sequence).float(), torch.from_numpy(condition).float()


# ==============================================================================
# 5. Core Sub-components / Encoder / Decoder
# ==============================================================================

class ConditionalGenerator(nn.Module):
    """
    Conditional Recurrent Generator Module.

    Mathematical Transformation:
        1. Context Vector Initializer:
           h_0 = MLP_in([z, c_0])  where z ~ N(0, I), c_0 = c[:, 0, :]
           Mapped Shape: [B, D_z + D_cond] -> [B, D_gen_hidden]
           Expanded to Multi-layer LSTM state: [Num_Layers, B, D_gen_hidden]

        2. Recurrent Unrolling over Condition Sequence:
           H, (h_n, c_n) = LSTM(C, (h_0, c_0_zeros))
           where C shape: [B, L, D_cond], H shape: [B, L, D_gen_hidden]

        3. Feature Projection per Timestep:
           x_t = Tanh(MLP_out(H_t))
           where x_t shape: [B, D_in]

    Args:
        config (CGANConfig): Global model configuration instance.

    Inputs:
        z (Tensor): Latent noise tensor, shape: [B, D_z]
        cond (Tensor): Temporal conditioning tensor, shape: [B, L, D_cond]

    Outputs:
        output (Tensor): Synthesized time-series sequence, shape: [B, L, D_in]
    """
    def __init__(self, config: CGANConfig):
        super().__init__()
        self.config = config

        # Initial hidden state projector mapping z and c_0 to hidden dimension
        self.fc_input = nn.Sequential(
            nn.Linear(config.latent_dim + config.cond_dim, config.gen_hidden_dim),
            nn.BatchNorm1d(config.gen_hidden_dim),
            nn.SiLU()
        )

        # Autoregressive sequence driving block
        self.lstm = nn.LSTM(
            input_size=config.cond_dim,
            hidden_size=config.gen_hidden_dim,
            num_layers=config.gen_num_layers,
            batch_first=True,
            dropout=0.1 if config.gen_num_layers > 1 else 0.0
        )

        # Non-linear output feature projection head
        self.fc_out = nn.Sequential(
            nn.Linear(config.gen_hidden_dim, config.gen_hidden_dim * 2),
            nn.SiLU(),
            nn.BatchNorm1d(config.gen_hidden_dim * 2),
            nn.Linear(config.gen_hidden_dim * 2, config.input_dim),
            nn.Tanh()  # Bounds outputs to [-1, 1] range
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # z shape: [B, D_z]
        # cond shape: [B, L, D_cond]
        batch_size, seq_len, _ = cond.shape

        # Extract initial condition timestep: [B, D_cond]
        cond_first = cond[:, 0, :]

        # Concatenate latent noise and initial condition: [B, D_z + D_cond]
        init_input = torch.cat([z, cond_first], dim=-1)

        # Compute initial hidden representation: [B, D_gen_hidden]
        init_state = self.fc_input(init_input)

        # Expand hidden state across multi-layer LSTM dimensions: [Num_Layers, B, D_gen_hidden]
        h0 = init_state.unsqueeze(0).repeat(self.config.gen_num_layers, 1, 1)
        c0 = torch.zeros_like(h0)  # Initial cell state set to zero

        # Propagate conditions through recurrent network: lstm_out shape -> [B, L, D_gen_hidden]
        lstm_out, _ = self.lstm(cond, (h0, c0))

        # Flatten sequence and batch dimensions for parallel MLP projection
        # [B, L, D_gen_hidden] -> [B * L, D_gen_hidden]
        flat_lstm_out = lstm_out.reshape(-1, self.config.gen_hidden_dim)

        # Project to input feature dimension: [B * L, D_in]
        flat_output = self.fc_out(flat_lstm_out)

        # Reshape back to target sequence tensor: [B, L, D_in]
        output = flat_output.view(batch_size, seq_len, self.config.input_dim)

        return output


class ConditionalDiscriminator(nn.Module):
    """
    Bidirectional Recurrent Discriminator Module with Temporal Contextualization.

    Mathematical Transformation:
        1. Feature-Condition Concatenation:
           U_t = [X_t || C_t] for t in 1..L
           Combined Tensor Shape: [B, L, D_in + D_cond]

        2. Bidirectional Sequence Encoding:
           H_seq = BiLSTM(U)
           where H_seq Shape: [B, L, 2 * D_dis_hidden]

        3. Global Temporal Representation Aggregation:
           H_final = H_seq[:, -1, :]  # Extracts final temporal slice
           Shape: [B, 2 * D_dis_hidden]

        4. Validity Score Projection:
           Score = LeakyReLU(Linear(Dropout(LeakyReLU(Linear(H_final)))))
           Output Score Shape: [B, 1]

    Args:
        config (CGANConfig): Global configuration parameters.

    Inputs:
        x (Tensor): Target real or synthetic time-series, shape: [B, L, D_in]
        cond (Tensor): Temporal conditioning factors, shape: [B, L, D_cond]

    Outputs:
        score (Tensor): Validity logits/scores, shape: [B, 1]
    """
    def __init__(self, config: CGANConfig):
        super().__init__()
        self.config = config

        input_size = config.input_dim + config.cond_dim

        # Bidirectional LSTM Encoder for long-range temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.dis_hidden_dim,
            num_layers=config.dis_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dis_dropout if config.dis_num_layers > 1 else 0.0
        )

        # High-capacity classifier head
        self.fc_out = nn.Sequential(
            nn.Linear(config.dis_hidden_dim * 2, config.dis_hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dis_dropout),
            nn.Linear(config.dis_hidden_dim * 2, config.dis_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dis_dropout),
            nn.Linear(config.dis_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x shape   : [B, L, D_in]
        # cond shape: [B, L, D_cond]

        # Concatenate along feature dimension: [B, L, D_in + D_cond]
        combined = torch.cat([x, cond], dim=-1)

        # Process full sequence: lstm_out shape -> [B, L, 2 * D_dis_hidden]
        lstm_out, _ = self.lstm(combined)

        # Temporal Pooling: Select final sequence hidden vector: [B, 2 * D_dis_hidden]
        last_out = lstm_out[:, -1, :]

        # Compute validity score: score shape -> [B, 1]
        out = self.fc_out(last_out)

        return out


# ==============================================================================
# 6. Top-Level Model / Pipeline Wrapper
# ==============================================================================

class CGANPipeline:
    """
    High-level Pipeline encapsulating Generator, Discriminator, Optimizers, and
    WGAN-GP Training Step Logic.
    """
    def __init__(self, config: CGANConfig):
        self.config = config
        self.generator = ConditionalGenerator(config).to(config.device)
        self.discriminator = ConditionalDiscriminator(config).to(config.device)

        # AdamW Optimizers with momentum hyperparameter tuning
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=config.lr_g,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=config.lr_d,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Binary Cross Entropy with Logits Loss
        self.bce_criterion = nn.BCEWithLogitsLoss()

    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Computes WGAN-GP Gradient Penalty to enforce 1-Lipschitz continuity constraint.

        Formula:
            GP = E [(|| \nabla_{\tilde{x}} D(\tilde{x}, cond) ||_2 - 1)^2]
            where \tilde{x} = \epsilon * real + (1 - \epsilon) * fake

        Inputs:
            real_data (Tensor): [B, L, D_in]
            fake_data (Tensor): [B, L, D_in]
            cond (Tensor)     : [B, L, D_cond]

        Outputs:
            penalty (Tensor)  : Scalar gradient penalty tensor.
        """
        batch_size = real_data.size(0)

        # Uniform random sample weight: [B, 1, 1]
        epsilon = torch.rand(batch_size, 1, 1, device=self.config.device)
        epsilon = epsilon.expand_as(real_data)

        # Convex combination interpolation: [B, L, D_in]
        interpolated = (epsilon * real_data + (1 - epsilon) * fake_data).requires_grad_(True)

        # Discriminator pass on interpolated inputs: [B, 1]
        disc_interpolated = self.discriminator(interpolated, cond)

        # Calculate exact gradient with respect to interpolated features
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Flatten gradients per sample: [B, L * D_in]
        gradients = gradients.view(batch_size, -1)

        # L2 norm calculation along feature dimensions: [B]
        gradient_norm = gradients.norm(2, dim=1)

        # Compute squared distance penalty from 1
        penalty = ((gradient_norm - 1.0) ** 2).mean()

        return penalty

    def train_discriminator_step(
        self, real_data: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes a single optimization step for the Discriminator using WGAN-GP.
        """
        self.optimizer_d.zero_grad()
        batch_size = real_data.size(0)

        # Sample Latent Noise and Generate Fake Data
        z = torch.randn(batch_size, self.config.latent_dim, device=self.config.device)
        fake_data = self.generator(z, cond).detach()

        # Compute Discriminator Predictions
        real_validity = self.discriminator(real_data, cond)  # [B, 1]
        fake_validity = self.discriminator(fake_data, cond)  # [B, 1]

        # WGAN Critic Loss: Maximize D(real) - D(fake) <=> Minimize D(fake) - D(real)
        d_loss_wasserstein = fake_validity.mean() - real_validity.mean()

        # Calculate Gradient Penalty
        gp = self.gradient_penalty(real_data, fake_data, cond)

        # Total Loss with Lambda Weighting
        total_d_loss = d_loss_wasserstein + self.config.gp_weight * gp

        total_d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.optimizer_d.step()

        return total_d_loss, d_loss_wasserstein, real_validity.mean(), fake_validity.mean()

    def train_generator_step(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a single optimization step for the Generator.
        """
        self.optimizer_g.zero_grad()
        batch_size = cond.size(0)

        # Sample fresh latent noise
        z = torch.randn(batch_size, self.config.latent_dim, device=self.config.device)

        # Forward pass through Generator
        fake_data = self.generator(z, cond)  # [B, L, D_in]

        # Evaluate synthesized data on Discriminator
        fake_validity = self.discriminator(fake_data, cond)  # [B, 1]

        # WGAN Generator Loss: Maximize D(G(z|c)) <=> Minimize -D(G(z|c))
        g_loss = -fake_validity.mean()

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer_g.step()

        return g_loss, fake_data


# ==============================================================================
# 7. Loss & Metrics
# ==============================================================================

def compute_sequence_metrics(real_seq: torch.Tensor, fake_seq: torch.Tensor) -> Dict[str, float]:
    """
    Calculates statistical similarity metrics between real and generated sequence batches.

    Inputs:
        real_seq (Tensor): [B, L, D_in]
        fake_seq (Tensor): [B, L, D_in]

    Outputs:
        metrics (Dict[str, float]): Computed MSE and MAE metrics.
    """
    with torch.no_grad():
        mse = nn.functional.mse_loss(fake_seq, real_seq).item()
        mae = nn.functional.l1_loss(fake_seq, real_seq).item()
    return {"seq_mse": mse, "seq_mae": mae}


# ==============================================================================
# 8. Training/Inference Execution & Entry Point
# ==============================================================================

def main():
    """
    Main Execution Entry Point for Dataset Loading, Model Setup, and Training Loop.
    """
    config = CGANConfig()

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("CGAN_Training")
    writer = SummaryWriter(config.log_dir)

    logger.info("Initializing Dataset and DataLoader...")
    dataset = ConditionalDataset(config.data_path, config.cond_data_path, config.seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for maximum cross-platform compatibility
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    logger.info(f"Dataset successfully created. Total sequence windows: {len(dataset)}")

    logger.info("Initializing CGAN Pipeline and Sub-modules...")
    pipeline = CGANPipeline(config)

    logger.info(f"Generator Params    : {sum(p.numel() for p in pipeline.generator.parameters()):,}")
    logger.info(f"Discriminator Params: {sum(p.numel() for p in pipeline.discriminator.parameters()):,}")

    logger.info("Starting Execution Loop...")
    for epoch in range(config.num_epochs):
        pipeline.generator.train()
        pipeline.discriminator.train()

        running_d_loss = 0.0
        running_g_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{config.num_epochs}]")
        for batch_idx, (real_data, cond) in enumerate(pbar):
            real_data = real_data.to(config.device)
            cond = cond.to(config.device)

            # ------------------------------------------------------------------
            # 1. Update Discriminator
            # ------------------------------------------------------------------
            d_total_loss, d_wasserstein, real_score, fake_score = pipeline.train_discriminator_step(real_data, cond)

            # ------------------------------------------------------------------
            # 2. Update Generator (every n_critic steps)
            # ------------------------------------------------------------------
            if batch_idx % config.n_critic == 0:
                g_loss, fake_data = pipeline.train_generator_step(cond)
            else:
                g_loss = torch.tensor(0.0)

            running_d_loss += d_total_loss.item()
            running_g_loss += g_loss.item()

            pbar.set_postfix({
                "D_Loss": f"{d_total_loss.item():.4f}",
                "G_Loss": f"{g_loss.item():.4f}",
                "D(x)": f"{real_score.item():.2f}",
                "D(G(z))": f"{fake_score.item():.2f}"
            })

            global_step = epoch * len(dataloader) + batch_idx
            if batch_idx % 20 == 0:
                writer.add_scalar("Train/D_Total_Loss", d_total_loss.item(), global_step)
                writer.add_scalar("Train/D_Wasserstein_Distance", -d_wasserstein.item(), global_step)
                writer.add_scalar("Train/G_Loss", g_loss.item(), global_step)
                writer.add_scalar("Train/Real_Score_Mean", real_score.item(), global_step)
                writer.add_scalar("Train/Fake_Score_Mean", fake_score.item(), global_step)

        epoch_d_loss = running_d_loss / len(dataloader)
        epoch_g_loss = running_g_loss / len(dataloader)
        logger.info(f"Epoch [{epoch + 1}/{config.num_epochs}] Finished -> Avg D_Loss: {epoch_d_loss:.6f}, Avg G_Loss: {epoch_g_loss:.6f}")

        # Model Checkpoint Persistence
        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(config.save_dir, f"cgan_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": pipeline.generator.state_dict(),
                "discriminator_state_dict": pipeline.discriminator.state_dict(),
                "optimizer_g_state_dict": pipeline.optimizer_g.state_dict(),
                "optimizer_d_state_dict": pipeline.optimizer_d.state_dict(),
            }, ckpt_path)
            logger.info(f"Checkpoint successfully saved to {ckpt_path}")

    writer.close()
    logger.info("Training pipeline execution completed.")


if __name__ == "__main__":
    main()