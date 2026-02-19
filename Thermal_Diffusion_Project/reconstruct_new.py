import torch
import os
import numpy as np
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from model import UNet

device = "cuda"

model_path = "./ddpm_ckpt/ddpm_epoch_99.pt"
data_dir = "./data/raw_256"           
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

def compute_ssim(x_rec, x_gt):
    rec = x_rec.squeeze().detach().cpu().numpy()
    gt = x_gt.squeeze().detach().cpu().numpy()

    score = ssim(gt, rec, data_image=1.0)
    return score

for name in os.listdir(data_dir):
    path = os.path.join(data_dir, name)

    img = Image.open(path)
    x0 = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        x_rec = reconstruct(x0)

    # 反归一化
    x0_img = (x0 * 0.5 + 0.5).clamp(0,1)
    x_rec_img = (x_rec * 0.5 + 0.5).clamp(0,1)
    print(type(x0_img))
    print(type(x_rec_img))

    rec_np = x_rec_img.squeeze().detach().cpu().numpy()
    gt_np = x0_img.squeeze().detach().cpu().numpy

    score = ssim(gt_np, rec_np, data_range = 1.0)
    ssim_scores.append(score)

    # ===== 保存重建图 =====
    out = (x_rec_img.squeeze().cpu().numpy() * 255).astype("uint8")

    save_path = os.path.join(save_dir, name)
    Image.fromarray(out).save(save_path)

    print(f"{name} SSIM={score:.4f}")

print("=================================")
print("Mean SSIM:", np.mean(ssim_scores))
print("Saved to:", save_dir)
