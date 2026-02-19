import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler
import os

device = "cuda"

LATENT_DIR = "latents_overfit"
BATCH = 6
EPOCHS = 300
LR = 5e-5

# ---------- dataset ----------
class LatentDataset(Dataset):
    def __init__(self, path):
        self.files = [os.path.join(path, f) for f in sorted(os.listdir(path))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return torch.load(self.files[i]).squeeze(0)

dataset = LatentDataset(LATENT_DIR)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

# ---------- model ----------
sample = dataset[0]
C, H, W = sample.shape

model = UNet2DModel(
    sample_size=H,
    in_channels=C,
    out_channels=C,
    layers_per_block=2,
    block_out_channels=(128, 256, 256)
).to(device)

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear"
)

opt = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------- training ----------
for epoch in range(EPOCHS):
    total_loss = 0

    for z in loader:
        z = z.to(device)

        noise = torch.randn_like(z)
        t = torch.randint(0, 1000, (z.shape[0],), device=device)

        z_noisy = scheduler.add_noise(z, noise, t)
        pred = model(z_noisy, t).sample

        loss = torch.nn.functional.mse_loss(pred, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    print(f"epoch {epoch} loss {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "ddpm_overfit.pt")
