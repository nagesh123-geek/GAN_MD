import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ===============================
# Config
# ===============================
class Config:
    save_dir = "wgan_P450_2_dim"
    batch_size = 16
    latent_dim = 1024
    signal_length = None   # autodetect below
    n_features = None      # autodetect below
    n_epochs = 2000
    lr = 1e-4
    n_critic = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
class FrameDataset(Dataset):
    def __init__(self, folder):
        """
        folder: path to folder containing .npy files
                each file should be shape (40000, 8)
        """
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {folder}")

        # Inspect first file to determine shape
        first = np.load(os.path.join(folder, self.files[0]))
        if first.ndim != 2:
            raise RuntimeError(f"Expected shape (T,F), got {first.shape}")

        T, F = first.shape
        Config.signal_length = T
        Config.n_features = F

        print(f"[Dataset] detected signal_length={T}, n_features={F}, n_files={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.files[idx])
        data = np.load(file_path)  # shape (T, F)
        assert data.shape[0] == Config.signal_length
        assert data.shape[1] == Config.n_features
        return torch.tensor(data, dtype=torch.float32)

# -------------------------
# Create dataset + loader
# -------------------------
folder = "../data_p450_reshaped"   # <-- change this path to your folder
dataset = FrameDataset(folder)
loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

# ===============================
# Models
# ===============================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        out_dim = Config.signal_length * Config.n_features
        self.model = nn.Sequential(
            nn.Linear(Config.latent_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, out_dim)
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, Config.signal_length, Config.n_features)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = Config.signal_length * Config.n_features
        self.model = nn.Sequential(
            nn.Linear(in_dim, 8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

G = Generator().to(Config.device)
D = Critic().to(Config.device)

opt_G = optim.Adam(G.parameters(), lr=Config.lr, betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=Config.lr, betas=(0.5, 0.9))

# ===============================
# Gradient Penalty
# ===============================
def gradient_penalty(D, real, fake):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=real.device, dtype=real.dtype)
    epsilon = epsilon.expand_as(real)

    interpolates = epsilon * real + (1 - epsilon) * fake
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

import time 

# ===============================
# Training loop with timing
# ===============================
start_time = time.time()  # Record start time

for epoch in range(1, Config.n_epochs + 1):
    epoch_start = time.time()  # Optional: track per-epoch time
    
    for real in loader:
        real = real.to(Config.device)
        B = real.size(0)

        # Train Critic
        for _ in range(Config.n_critic):
            z = torch.randn(B, Config.latent_dim, device=Config.device)
            fake = G(z).detach()
            d_loss = -(torch.mean(D(real)) - torch.mean(D(fake)))
            gp = gradient_penalty(D, real, fake)
            loss_D = d_loss + 10.0 * gp

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # Train Generator
        z = torch.randn(B, Config.latent_dim, device=Config.device)
        fake = G(z)
        loss_G = -torch.mean(D(fake))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    print(f"[Epoch {epoch}/{Config.n_epochs}] "
          f"D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f} | "
          f"Epoch time: {epoch_time:.1f}s", flush=True)

    # Save + generate samples every 200 epochs
    if epoch % 200 == 0:
        torch.save(G.state_dict(), os.path.join(Config.save_dir, f"G_epoch{epoch}.pt"))

        G.eval()
        with torch.no_grad():
            for i in range(10):
                z = torch.randn(1, Config.latent_dim, device=Config.device)
                sample = G(z).cpu().numpy().squeeze()  # shape (40000, 8)

                time_arr = np.arange(sample.shape[0])

                plt.figure(figsize=(12, 6))
                for fi in range(sample.shape[1]):
                    plt.plot(time_arr, sample[:, fi], label=f"Feature {fi+1}")
                plt.xlabel("Time step")
                plt.ylabel("Value")
                plt.title(f"Generated Time Series (Epoch {epoch}, Sample {i+1})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(Config.save_dir, f"timeplot_epoch{epoch}_{i+1}.png"))
                plt.close()

        G.train()

end_time = time.time()
total_training_time = end_time - start_time
print(f" Total training time: {total_training_time/60:.2f} minutes ({total_training_time:.1f}s)")

