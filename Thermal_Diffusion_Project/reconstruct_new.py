import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

# =====================
# 配置
# =====================

model_path = "./ddpm_ckpt/ddpm_epoch_99.pt"
data_dir = "./ir_256"
device = "cuda"
image_size = 256
T = 1000

# =====================
# 数据集
# =====================

class IRDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        return self.transform(img), self.paths[i]

loader = DataLoader(IRDataset(data_dir), batch_size=1)

# =====================
# 加载模型（使用你的UNet）
# =====================

model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =====================
# 扩散参数（必须与训练一致）
# =====================

betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# =====================
# 单图重建
# =====================

def reconstruct(x0, t_start=300):
    noise = torch.randn_like(x0)
    xt = (
        torch.sqrt(alphas_cumprod[t_start]) * x0 +
        torch.sqrt(1 - alphas_cumprod[t_start]) * noise
    )

    x = xt
    for i in reversed(range(t_start)):
        t = torch.tensor([i], device=device).float()
        pred_noise = model(x, t)

        alpha = alphas[i]
        alpha_bar = alphas_cumprod[i]
        beta = betas[i]

        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
        )

        if i > 0:
            x += torch.sqrt(beta) * torch.randn_like(x)

    return x

# =====================
# 评估
# =====================

ssim_scores = []

for x, path in loader:
    x = x.to(device)

    recon = reconstruct(x)

    x_img = ((x.clamp(-1,1) + 1) / 2).cpu().numpy()[0,0]
    recon_img = ((recon.clamp(-1,1) + 1) / 2).cpu().numpy()[0,0]

    score = ssim(x_img, recon_img, data_range=1.0)
    ssim_scores.append(score)

    print(os.path.basename(path[0]), "SSIM:", score)

print("=================================")
print("Average SSIM:", np.mean(ssim_scores))
