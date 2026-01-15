import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

# ===============================
# Config (same as training)
# ===============================
class Config:
    model_dir = "../wgan_p450"
    save_dir = "generated_samples"
    os.makedirs(save_dir, exist_ok=True)
    latent_dim = 1024
    signal_length = 40000
    n_features = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===============================
# Generator definition (must match training)
# ===============================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Config.latent_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, Config.signal_length * Config.n_features)
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, Config.signal_length, Config.n_features)

# ===============================
# Load generator
# ===============================
G = Generator().to(Config.device)


epoch = 2000  
model_path = os.path.join(Config.model_dir, f"G_epoch{epoch}.pt")
G.load_state_dict(torch.load(model_path, map_location=Config.device))
G.eval()

# ===============================
# Generate and plot 10 samples
# ===============================

colors = plt.cm.tab10.colors

with torch.no_grad():
    for i in range(10):
        z = torch.randn(1, Config.latent_dim, device=Config.device)
        sample = G(z).cpu().numpy().squeeze()  # (40000,)

        # Save raw numpy
        np.save(os.path.join(Config.save_dir, f"gen_sample_epoch{epoch}_{i+1}.npy"), sample)

        # Plot time series
        time = np.arange(sample.shape[0])
        plt.figure(figsize=(12, 4))
        plt.plot(time, sample, color=colors[i % len(colors)], label=f"Sample {i+1}")
        #plt.plot(time, sample, label="Generated Feature")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title(f"Generated Sample {i+1} at Epoch {epoch}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(Config.save_dir, f"gen_timeplot_epoch{epoch}_{i+1}.png"))
        plt.close()
