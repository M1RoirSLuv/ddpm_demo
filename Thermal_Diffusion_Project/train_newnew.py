import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# =====================
# 配置
# =====================

data_dir = "./ir_256"        # 256×256红外图目录
save_dir = "./ddpm_ckpt"
sample_dir = "./ddpm_samples"

image_size = 256
batch_size = 8
epochs = 100
lr = 2e-4
T = 1000
device = "cuda"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

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
        return self.transform(img)

dataset = IRDataset(data_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# =====================
# 时间嵌入
# =====================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)

# =====================
# ResBlock
# =====================

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = F.relu(self.conv2(h))
        return h + self.shortcut(x)

# =====================
# 官方风格 UNet
# =====================

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(1, 64, 3, padding=1)

        self.down1 = ResBlock(64, 128, time_dim)
        self.down2 = ResBlock(128, 256, time_dim)
        self.down3 = ResBlock(256, 256, time_dim)

        self.mid = ResBlock(256, 256, time_dim)

        self.up3 = ResBlock(256 + 256, 256, time_dim)
        self.up2 = ResBlock(256 + 128, 128, time_dim)
        self.up1 = ResBlock(128 + 64, 64, time_dim)

        self.out = nn.Conv2d(64, 1, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t):
        t = self.time_mlp(t)

        x0 = self.conv0(x)
        x1 = self.down1(x0, t)
        x2 = self.down2(self.pool(x1), t)
        x3 = self.down3(self.pool(x2), t)

        h = self.mid(self.pool(x3), t)

        h = F.interpolate(h, scale_factor=2)
        h = self.up3(torch.cat([h, x3], dim=1), t)

        h = F.interpolate(h, scale_factor=2)
        h = self.up2(torch.cat([h, x2], dim=1), t)

        h = F.interpolate(h, scale_factor=2)
        h = self.up1(torch.cat([h, x1], dim=1), t)

        return self.out(h)

model = UNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# =====================
# 标准线性噪声调度
# =====================

betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise):
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

# =====================
# 训练循环（双进度条）
# =====================

for epoch in range(epochs):

    epoch_bar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    for x in epoch_bar:
        x = x.to(device)

        t = torch.randint(0, T, (x.size(0),), device=device)
        noise = torch.randn_like(x)
        xt = q_sample(x, t, noise)

        pred_noise = model(xt, t.float())
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_bar.set_postfix(loss=loss.item())

    # 保存模型
    torch.save(model.state_dict(), f"{save_dir}/ddpm_epoch_{epoch}.pt")

    # 采样
    with torch.no_grad():
        x = torch.randn(1, 1, image_size, image_size).to(device)
        for i in reversed(range(200)):
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

        x = (x.clamp(-1, 1) + 1) / 2
        save_image(x, f"{sample_dir}/sample_epoch_{epoch}.png")

print("training complete")
