import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from pytorch_msssim import ssim
from PIL import Image
from model import UNet   

device = "cuda"

model_path = "ddpm_model.pt"
data_dir = "data_256"           
save_dir = "recon_results"
os.makedirs(save_dir, exist_ok=True)

img_size = 256
T = 1000

# ===== load model =====
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== diffusion schedule =====
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# ===== transform =====
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def reconstruct(x0):
    """从图像 → 加噪 → DDPM反推"""
    x = x0.clone()

    # 正向加噪到T
    for t in range(T):
        noise = torch.randn_like(x)
        x = torch.sqrt(alpha[t]) * x + torch.sqrt(beta[t]) * noise

    # 反向采样
    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=device)
        noise_pred = model(x, t_tensor)

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        x = (1 / torch.sqrt(alpha[t])) * (
            x - (beta[t] / torch.sqrt(1 - alpha_bar[t])) * noise_pred
        ) + torch.sqrt(beta[t]) * noise

    return x

# ===== evaluation =====
ssim_scores = []

for name in os.listdir(data_dir):
    path = os.path.join(data_dir, name)

    img = Image.open(path)
    x0 = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        x_rec = reconstruct(x0)

    # 反归一化
    x0_img = (x0 * 0.5 + 0.5).clamp(0,1)
    x_rec_img = (x_rec * 0.5 + 0.5).clamp(0,1)

    score = ssim(x_rec_img, x0_img, data_range=1.0).item()
    ssim_scores.append(score)

    # ===== 保存重建图 =====
    out = x_rec_img.squeeze().cpu().numpy() * 255
    out = out.astype(np.uint8)

    save_path = os.path.join(save_dir, name)
    cv2.imwrite(save_path, out)

    print(f"{name} SSIM={score:.4f}")

print("=================================")
print("Mean SSIM:", np.mean(ssim_scores))
print("Saved to:", save_dir)
