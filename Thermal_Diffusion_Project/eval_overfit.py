import torch
from diffusers import UNet2DModel, DDPMScheduler
import os

device = "cuda"

LATENT_PATH = "latents_overfit/sample.pt"

z = torch.load(LATENT_PATH).to(device)

C, H, W = z.shape

model = UNet2DModel(sample_size=H, in_channels=C, out_channels=C).to(device)
model.load_state_dict(torch.load("ddpm_overfit.pt"))
model.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(1000)

zt = torch.randn_like(z.unsqueeze(0))

for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = model(zt, t).sample
    zt = scheduler.step(noise_pred, t, zt).prev_sample

z_recon = zt.squeeze(0)

mse = torch.mean((z_recon - z)**2).item()
print("latent reconstruction MSE =", mse)
