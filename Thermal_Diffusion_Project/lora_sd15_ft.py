import os
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
# 新增：引入官方的 Dataset 和 DataLoader 模块
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===== 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt"
clip_path = "./model/clip-vit-large-patch14"
data_dir = "ir_rgb_256"
out_dir = "sd15_ir_lora"
os.makedirs(out_dir, exist_ok=True)

image_size = 256
batch_size = 4
lr = 1e-4
epochs = 5
device = "cuda"

# ===== 修复 1：规范化数据加载器 (Dataset & DataLoader) =====
class ImageDataset(Dataset):
    def __init__(self, data_dir, size):
        self.image_paths = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # 返回字典格式，这样后续调用 batch["pixel_values"] 才不会报错
        return {"pixel_values": self.transform(img)}

dataset = ImageDataset(data_dir, image_size)
# 将 dataset 包装进 DataLoader，支持批量处理和打乱
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===== 纯本地加载基座 =====
print("Loading CLIP (Local)")
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True)

print("Loading SD (Local Single File)")
# 明确关闭 safety_checker，完全断开网络依赖
pipe = StableDiffusionPipeline.from_single_file(
    ckpt_path,
    torch_dtype=torch.float16,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    load_safety_checker=False,  
    local_files_only=True
).to(device)
print("SD loaded")

unet, vae = pipe.unet, pipe.vae
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# 冻结原始权重
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# ===== LoRA 注入（保持你的结构）=====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        in_c, out_c = conv.in_channels, conv.out_channels
        k = conv.kernel_size[0]
        self.down = nn.Conv2d(in_c, rank, k, padding=conv.padding, bias=False)
        self.up = nn.Conv2d(rank, out_c, 1, bias=False)
        
    def forward(self, x):
        return self.conv(x) + self.up(self.down(x))

for name, module in list(unet.named_modules()):
    if isinstance(module, nn.Conv2d):
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAConv2d(module))

unet.to(device)

# ===== 修复 2：统一优化器变量名 =====
# 原代码这里定义的是 opt，但下方循环里用的是 optimizer
optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=lr)

# ===== 训练 =====
unet.train()
for epoch in range(epochs):
    epoch_start = time.time()
    epoch_loss = 0.0

    # 这里传入刚才定义好的 dataloader
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for step, batch in enumerate(progress_bar):
        step_start = time.time()

        # 确保输入图像与 VAE 的 dtype (float16) 保持一致，避免类型冲突报错
        pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)

        # ===== 前向 =====
        latents = vae.encode(pixel_values).latent_dist.sample()
        # 修复 3：使用配置里的 scaling_factor，更加严谨
        latents = latents * vae.config.scaling_factor 

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.size(0),), device=device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = unet(noisy_latents, timesteps).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # ===== 反向 =====
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===== 统计 =====
        step_time = time.time() - step_start
        epoch_loss += loss.item()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "step_time": f"{step_time:.2f}s"
        })

    epoch_time = time.time() - epoch_start
    print(
        f"\nEpoch {epoch+1} finished | "
        f"avg loss = {epoch_loss/len(dataloader):.6f} | "
        f"time = {epoch_time/60:.2f} min\n"
    )

torch.save(unet.state_dict(), os.path.join(out_dir, "lora_unet.pt"))
print("Saved:", os.path.join(out_dir, "lora_unet.pt"))
