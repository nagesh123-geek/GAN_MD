# ===============================
# Generate 100 seeded datasets from saved Generator
# ===============================
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Config
# ===============================
class Config:
    base_save_dir = "generated_data"
    saved_model = "../wgan_results/G_epoch1000.pt"   # path to saved generator
    latent_dim = 1024
    total_len = 35000
    signal_length = 35000
    chunk_size = 25000
    n_features = 2
    batch_latent = 1
    n_datasets = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(base_save_dir, exist_ok=True)

# ===============================
# Generator definition (match training!)
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
# Load saved model
# ===============================
G = Generator().to(Config.device)
G.load_state_dict(torch.load(os.path.join(Config.saved_model), map_location=Config.device))
G.eval()
print("Loaded generator")


# ===============================
# Generate 100 datasets with 50% overlap
# ===============================
for idx in range(Config.n_datasets):
    seed = 42 + idx
    torch.manual_seed(seed)
    np.random.seed(seed)

    generated_datasets = []
    prev_tail = None  

    num_chunks = Config.total_len // (Config.chunk_size // 2)  

    with torch.no_grad():
        for c in range(num_chunks):
            z = torch.randn(Config.batch_latent, Config.latent_dim, device=Config.device)
            chunk = G(z).cpu().numpy().reshape(-1, Config.n_features)  # (chunk_size, 2)

            if prev_tail is not None:
                # Take 50% from previous + 50% from new
                combined = np.vstack([prev_tail, chunk[: Config.chunk_size // 2]])
                generated_datasets.append(combined)
            else:
                # First chunk: just take the full thing
                generated_datasets.append(chunk)

            # Save last 50% for next overlap
            prev_tail = chunk[Config.chunk_size // 2 :]

    # Concatenate all overlapping chunks
    dataset_full = np.vstack(generated_datasets)[: Config.total_len]  # ensure exact length

    # Save data with seed in filename
    save_path = os.path.join(Config.base_save_dir, f"data_{seed}.npy")
    np.save(save_path, dataset_full)
    print(f" Saved dataset {idx} as {save_path}, shape {dataset_full.shape}")

    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(dataset_full[:, 0], label="Feature 1")
    plt.plot(dataset_full[:, 1], label="Feature 2")
    plt.title(f"Generated Features (50% overlap) - Seed {seed}")
    plt.xlabel("Sample index")
    plt.ylabel("Feature value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.base_save_dir, f"plot_{seed}.png"))
    plt.close()
