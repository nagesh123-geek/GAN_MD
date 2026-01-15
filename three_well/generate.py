import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ===============================
# Config
# ===============================
class Config:
    save_dir = "generated_data_threeWELL"
    model_dir = "../wgan_three_well"
    latent_dim = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    signal_length = 40000
    n_features = 2

# ===============================
# Generator definition (same as training)
# ===============================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Config.latent_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096,8192),
            nn.ReLU(),
            #nn.Linear(8192,16384),
            #nn.ReLU(),

            nn.Linear(8192, Config.signal_length * Config.n_features)
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, Config.signal_length, Config.n_features)


# ===============================
# Load trained generator
# ===============================
G = Generator().to(Config.device)
checkpoint = os.path.join(Config.model_dir, "G_epoch2000.pt")  # <-- change if needed
G.load_state_dict(torch.load(checkpoint, map_location=Config.device))
G.eval()

# ===============================
# Generate new samples
# ===============================
output_dir = "generated_samples"
os.makedirs(output_dir, exist_ok=True)

n_samples = 10   # number of samples to generate
colors = cm.tab10.colors  # 10 distinct colors

with torch.no_grad():
    for i in range(n_samples):
        # Sample latent vector
        z = torch.randn(1, Config.latent_dim, device=Config.device)

        # Generate fake data -> shape (40000, 2)
        sample = G(z).cpu().numpy().squeeze()

        # Concatenate col1 and col2 -> shape (80000,)
        #concatenated = np.concatenate([sample[:, 0], sample[:, 1]])

        # Save as .npy
        np.save(os.path.join(output_dir, f"sample_{i+1}.npy"), sample)

        # Plot
        #time = np.arange(concatenated.shape[0])
        X = sample[:,0]
        Y = sample[:,1]
        plt.figure(figsize=(12, 4))
        plt.figure(figsize=(6,5))
        plt.hist2d(X, Y, bins=200, cmap='hot')
        plt.colorbar(label='Density')
        plt.title(f"Heatmap ")
        plt.xlabel("X")
        plt.ylabel("Y")
        #plt.plot(X,Y , color=colors[i % len(colors)])
        #plt.xlabel("Time step")
        #plt.ylabel("Value")
        #plt.title(f"Generated Time Series (Sample {i+1})")
        #plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{i+1}.png"))
        plt.close()





print(f"[Done] Saved {n_samples} generated samples in '{output_dir}'")
