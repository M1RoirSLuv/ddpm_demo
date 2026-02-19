import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
import numpy as np
import os

device = "cuda"

VAE_PATH = "your_vae_path"
IMG_PATH = "test.png"
SAVE_PATH = "vae_recon.png"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device)
vae.eval()

img = Image.open(IMG_PATH)
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    z = vae.encode(x).latent_dist.sample()
    z = z * vae.config.scaling_factor
    z = z / vae.config.scaling_factor
    recon = vae.decode(z).sample

recon = (recon.clamp(-1, 1) + 1) / 2
recon = recon.cpu().squeeze().numpy()

from PIL import Image
Image.fromarray((recon * 255).astype(np.uint8)).save(SAVE_PATH)

print("VAE reconstruction saved →", SAVE_PATH)
