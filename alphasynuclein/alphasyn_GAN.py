import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
# ===============================
# Config
# ===============================
class Config:
    data_file = "downsample_latent_alphasyn.npy"
    save_dir = "wgan_results_iid"
    batch_size = 16
    latent_dim = 1024
    signal_length = 35000
    n_features = 2
    n_epochs = 1000
    lr = 1e-4
    n_critic = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

# ===============================
# Dataset
# ===============================

# ===============================
#class ChunkDataset(Dataset):
#    def __init__(self, npy_file, chunk_size=10000):
#        data = np.load(npy_file)  # e.g., (500000, 2)
#        n_full_chunks = data.shape[0] // chunk_size  # only full chunks
#        data = data[:n_full_chunks * chunk_size]     # drop remainder
#        self.chunks = np.split(data, n_full_chunks)  # list of (chunk_size, 2)

#    def __len__(self):
#        return len(self.chunks)

#    def __getitem__(self, idx):
#        return torch.tensor(self.chunks[idx], dtype=torch.float32)


#dataset = ChunkDataset(Config.data_file, chunk_size=35000)
#loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)




#######################################################################################################################################



#                     IID DATASET



#####################################################################################################################################

import torch
from torch.utils.data import Dataset
import numpy as np

class IIDWindowDataset(Dataset):
    def __init__(self, npy_file, chunk_size=40000, n_samples=100):
        """
        npy_file: path to full dataset (N, 2)
        chunk_size: number of i.i.d. points per sample
        n_samples: number of chunks per epoch
        """
        self.data = np.load(npy_file)
        self.chunk_size = chunk_size
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Sample `chunk_size` points i.i.d. from the full dataset (without replacement)
        indices = np.random.choice(len(self.data), size=self.chunk_size, replace=False)
        chunk = self.data[indices]
        return torch.tensor(chunk, dtype=torch.float32)



dataset = IIDWindowDataset(Config.data_file, chunk_size=35000, n_samples=100)
loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)




######################################################################################################################################


#                           SEQUENTIAL DATASET


######################################################################################################################################


class SequentialWindowDataset(Dataset):
    def __init__(self, npy_file, chunk_size=35000, n_bins=7, n_samples=100):
        """
        npy_file: path to full dataset (N, 2)
        chunk_size: total number of points per chunk
        n_bins: number of bins to split the dataset
        n_samples: number of chunks per epoch
        """
        self.data = np.load(npy_file)
        self.chunk_size = chunk_size
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.bin_size = self.data.shape[0] // n_bins

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        bins = []
        samples_per_bin = self.chunk_size // self.n_bins

        for b in range(self.n_bins):
            start_bin = b * self.bin_size
            end_bin = (b + 1) * self.bin_size if b < self.n_bins - 1 else self.data.shape[0]

            # Random start for sequential window
            max_start = end_bin - samples_per_bin
            start_idx = np.random.randint(start_bin, max_start + 1)
            end_idx = start_idx + samples_per_bin

            bins.append(self.data[start_idx:end_idx])  # sequential slice

        # Shuffle the bins (rows inside bins stay sequential)
        random.shuffle(bins)

        # Stack the bins into final chunk
        chunk = np.vstack(bins)
        return torch.tensor(chunk, dtype=torch.float32)


#dataset = SequentialWindowDataset(Config.data_file, chunk_size=35000, n_bins=2, n_samples=100)
#loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

############################################################################################################################################



# ===============================
# Models
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

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Config.signal_length * Config.n_features, 8192),
            nn.LeakyReLU(0.2),
            #nn.Linear(16384, 8192),
            #nn.LeakyReLU(0.2),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

G = Generator().to(Config.device)
D = Critic().to(Config.device)

opt_G = optim.Adam(G.parameters(), lr=Config.lr)
opt_D = optim.Adam(D.parameters(), lr=Config.lr)

# ===============================
# Gradient Penalty
# ===============================
def gradient_penalty(D, real, fake):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=Config.device)
    interpolates = epsilon * real + (1 - epsilon) * fake
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# ===============================
# Training
# ===============================
for epoch in range(1, Config.n_epochs+1):
    for real in loader:
        real = real.to(Config.device)
        B = real.size(0)

        # Train Critic
        for _ in range(Config.n_critic):
            z = torch.randn(B, Config.latent_dim, device=Config.device)
            fake = G(z).detach()
            d_loss = -(torch.mean(D(real)) - torch.mean(D(fake)))
            gp = gradient_penalty(D, real, fake)
            loss_D = d_loss + 10 * gp

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

    print(f"[Epoch {epoch}/{Config.n_epochs}] D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}",flush=True)

    # Save + generate samples every 100 epochs
    if epoch % 100 == 0:
        torch.save(G.state_dict(), os.path.join(Config.save_dir, f"G_epoch{epoch}.pt"))
        #torch.save(D.state_dict(), os.path.join(Config.save_dir, f"D_epoch{epoch}.pt"))

        G.eval()
        with torch.no_grad():
            for i in range(10):
                z = torch.randn(1, Config.latent_dim, device=Config.device)
                sample = G(z).cpu().numpy().squeeze()  # (10000,2)
              #  np.save(os.path.join(Config.save_dir, f"sample_epoch{epoch}_{i+1}.npy"), sample)

                # Plot
                import matplotlib.pyplot as plt



                import matplotlib.pyplot as plt

                # Assuming sample.shape = (chunk_size, 2)
                time = np.arange(sample.shape[0])  # x-axis for time

                plt.figure(figsize=(12, 4))
                plt.plot(time, sample[:, 0], label="Feature 1")  # line plot for feature 1
                plt.plot(time, sample[:, 1], label="Feature 2")  # line plot for feature 2
                plt.xlabel("Time step")
                plt.ylabel("Value")
                plt.title("Time Series of Features")
                plt.legend()
                plt.savefig(os.path.join(Config.save_dir, f"timeplot_epoch{epoch}_{i+1}.png"))
                plt.close()



                plt.figure(figsize=(12,4))
                plt.scatter(sample[:,0],sample[:,1])
               # plt.plot(sample[:,1], label="feature2")
                plt.title(f"Generated Sample {i+1} at Epoch {epoch}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(Config.save_dir, f"plot_epoch{epoch}_{i+1}.png"))
                plt.close()
        G.train()
