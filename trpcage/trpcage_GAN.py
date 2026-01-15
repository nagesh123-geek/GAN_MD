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
    save_dir = "cWGAN_TRPCAGE_WITHOUT_NORMALIZATION_separate"
    batch_size = 1
    latent_dim = 1024
    signal_length = None   # auto detect
    n_features = None      # auto detect
    n_epochs = 2000
    lr = 1e-4
    n_critic = 5
    embed_dim = 16         # embedding dim for conditional labels
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

# ===============================
# Dataset
# ===============================
class ConditionalFrameDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {folder}")

        # Inspect first file
        first = np.load(os.path.join(folder, self.files[0]), allow_pickle=True)
        T, F = first.shape
        Config.signal_length = T - 1
        Config.n_features = F

        # Map labels R1-R140 to numeric indices
        self.label_map = {f"R{i+1}": i for i in range(len(self.files))}

        print(f"[Dataset] signal_length={Config.signal_length}, n_features={F}, n_files={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.files[idx])
        data = np.load(file_path, allow_pickle=True)  # (40001,2)
        label_row = data[0]  # first row is label
        numeric_data = data[1:].astype(np.float32)  # remaining rows
        # Convert label to numeric index
        label_num = self.label_map[label_row[0]]
        return torch.tensor(numeric_data, dtype=torch.float32), label_num

# ===============================
# DataLoader
# ===============================
folder = "../data_downsample/labeled_AF_data_separate"  # your folder
dataset = ConditionalFrameDataset(folder)
loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

# ===============================
# Conditional Generator
# ===============================
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, signal_len, num_classes, embed_dim=16):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, signal_len * Config.n_features)
        )

    def forward(self, z, labels):
        label_vec = self.label_emb(labels)
        x = torch.cat([z, label_vec], dim=1)
        out = self.model(x)
        return out.view(-1, Config.signal_length, Config.n_features)

# ===============================
# Conditional Critic
# ===============================
class ConditionalCritic(nn.Module):
    def __init__(self, signal_len, num_classes, embed_dim=16):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.model = nn.Sequential(
            nn.Linear(signal_len*Config.n_features + embed_dim, 8192),
            nn.LeakyReLU(0.2),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1)
        )

    def forward(self, x, labels):
        label_vec = self.label_emb(labels)
        x = torch.cat([x.view(x.size(0), -1), label_vec], dim=1)
        return self.model(x)

# ===============================
# Gradient Penalty
# ===============================
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=Config.device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ===============================
# Initialize models
# ===============================
num_classes = len(dataset.files)
G = ConditionalGenerator(Config.latent_dim, Config.signal_length, num_classes, Config.embed_dim).to(Config.device)
D = ConditionalCritic(Config.signal_length, num_classes, Config.embed_dim).to(Config.device)

opt_G = optim.Adam(G.parameters(), lr=Config.lr, betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=Config.lr, betas=(0.5, 0.9))

# ===============================
# Training loop
# ===============================
for epoch in range(1, Config.n_epochs+1):
    for real, labels in loader:
        real = real.to(Config.device)
        labels = labels.to(Config.device)
        B = real.size(0)

        # Train Critic
        for _ in range(Config.n_critic):
            z = torch.randn(B, Config.latent_dim, device=Config.device)
            fake = G(z, labels).detach()
            d_loss = -torch.mean(D(real, labels)) + torch.mean(D(fake, labels))
            gp = compute_gradient_penalty(D, real, fake, labels)
            loss_D = d_loss + 10.0 * gp

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # Train Generator
        z = torch.randn(B, Config.latent_dim, device=Config.device)
        fake = G(z, labels)
        loss_G = -torch.mean(D(fake, labels))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"[Epoch {epoch}] Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}", flush=True)

    # Save + generate samples every 200 epochs
    if epoch % 200 == 0:
        torch.save(G.state_dict(), os.path.join(Config.save_dir, f"G_epoch{epoch}.pt"))
        G.eval()
        with torch.no_grad():
            for label_idx in range(num_classes):
                z = torch.randn(1, Config.latent_dim, device=Config.device)
                lbl = torch.tensor([label_idx], device=Config.device)
                fake = G(z, lbl).squeeze(0).cpu().numpy()

                plt.figure(figsize=(12,6))
                for fi in range(Config.n_features):
                    plt.plot(fake[:, fi], label=f"Feature {fi+1}")
                plt.title(f"Generated Signal - R{label_idx+1}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(Config.save_dir, f"epoch{epoch}_R{label_idx+1}.png"))
                plt.close()
        G.train()
