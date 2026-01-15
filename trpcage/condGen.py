import os
import numpy as np
import torch
import torch.nn as nn

# Config 
class Config:
    save_dir = "../cWGAN_TRPCAGE_WITHOUT_NORMALIZATION_separate"
    latent_dim = 1024
    signal_length = 40000
    n_features = 2
    embed_dim = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Conditional Generator 
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

# load the saved model 
model_path = os.path.join(Config.save_dir, "G_epoch2000.pt")
num_classes = 24
G = ConditionalGenerator(Config.latent_dim, Config.signal_length, num_classes, Config.embed_dim).to(Config.device)
G.load_state_dict(torch.load(model_path, map_location=Config.device))
G.eval()

output_dir = "generated_paired_samples"
os.makedirs(output_dir, exist_ok=True)

pairs = [(2*i, 2*i + 1) for i in range(num_classes // 2)]  # 12 pairs

for sample_id in range(1, 11):  # 10 samples
    paired_data = []  # will collect ONE 40000x2 per pair (so final is 12*40000 x 2 = 480000 x 2)

    for p1, p2 in pairs:
       
        z = torch.randn(1, Config.latent_dim, device=Config.device)
        lbl = torch.tensor([p1], device=Config.device)

        with torch.no_grad():
            fake = G(z, lbl).squeeze(0).cpu().numpy()  # shape (40000, 2)

        paired_data.append(fake)

    # Combine all pairs vertically: (12 * 40000, 2) == (480000, 2)
    all_pairs_data = np.vstack(paired_data)

    save_path = os.path.join(output_dir, f"synthetic_pair_sample_{sample_id}.npy")
    np.save(save_path, all_pairs_data)
    print(f" Saved: {save_path}  shape={all_pairs_data.shape}")

print("\n All 10 paired synthetic samples generated successfully!")
