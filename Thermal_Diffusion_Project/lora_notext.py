import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from diffusers import StableDiffusionPipeline, DDPMScheduler

# ===== 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt"
data_dir = "./data/raw_256"
out_dir = "./sd15_lora_ir"
os.makedirs(out_dir, exist_ok=True)

image_size = 256
batch_size = 4
lr = 1e-4
epochs = 5
device = "cuda"

# ===== 加载 SD（无文本条件）=====
pipe = download_from_original_stable_diffusion_ckpt(
    checkpoint_path=ckpt_path,
    from_safetensors=False,
    device="cuda",
    extract_ema=True
)

unet = pipe.unet
vae = pipe.vae
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# 冻结原模型
for p in unet.parameters():
    p.requires_grad = False
vae.requires_grad_(False)

# ===== LoRA 模块（Conv2d版本）=====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        self.rank = rank

        in_c = conv.in_channels
        out_c = conv.out_channels
        k = conv.kernel_size[0]

        self.lora_down = nn.Conv2d(in_c, rank, k, padding=k//2, bias=False)
        self.lora_up = nn.Conv2d(rank, out_c, 1, bias=False)

        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.conv(x) + self.lora_up(self.lora_down(x))

# 注入 LoRA
def inject_lora(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child).to(device))
        else:
            inject_lora(child)

inject_lora(unet)

# 收集可训练参数
train_params = [p for p in unet.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(train_params, lr=lr)

# ===== 数据集 =====
class InfraredDataset(Dataset):
    def __init__(self, root):
        self.files = [os.path.join(root, f) for f in os.listdir(root)]

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        img = self.tf(img)
        img = img.repeat(3, 1, 1)  # 转3通道
        return img

dataset = InfraredDataset(data_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ===== 训练 =====
print("Start training...")
train_start = time.time()

for epoch in range(epochs):

    epoch_start = time.time()
    epoch_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

    for x in pbar:
        x = x.to(device, dtype=torch.float16)

        with torch.no_grad():
            latents = vae.encode(x).latent_dist.sample()
            latents = latents * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps,
            (latents.size(0),),
            device=device
        )

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = unet(noisy_latents, timesteps).sample

        loss = nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    epoch_time = time.time() - epoch_start

    print(
        f"\nEpoch {epoch+1} | avg loss {epoch_loss/len(loader):.6f} | "
        f"time {epoch_time/60:.2f} min"
    )

    torch.save(unet.state_dict(), f"{out_dir}/lora_unet_epoch{epoch+1}.pt")

total_time = time.time() - train_start
print(f"\nTraining finished. Total time {total_time/3600:.2f} hours")
