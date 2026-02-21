import os, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time

# ===== 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt"
clip_path = "./model/clip-vit-large-patch14"
data_dir = "./data/raw_256"
out_dir = "sd15_ir_lora"
os.makedirs(out_dir, exist_ok=True)

image_size = 256
batch_size = 4
lr = 1e-4
epochs = 5
device = "cuda"

# ===== 加载基座 =====
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(device)

print("Loading SD")
pipe = StableDiffusionPipeline.from_single_file(
    ckpt_path,
    torch_dtype=torch.float16,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    safety_checker=None,
    requires_safety_checker=False,
    local_files_only=True
).to(device)
print("SD loaded")

unet, vae = pipe.unet, pipe.vae
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# 冻结原始权重
for p in unet.parameters():
    p.requires_grad = False
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# ===== LoRA 注入=====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        in_c, out_c = conv.in_channels, conv.out_channels
        
        # 统一使用 1x1 卷积，避免 stride 导致的尺寸不匹配报错
        self.down = nn.Conv2d(in_c, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, out_c, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)
        
    def forward(self, x, *args, **kwargs):
        return self.conv(x) + self.up(self.down(x))

for name, module in list(unet.named_modules()):
    if isinstance(module, nn.Conv2d):
        # 避开降采样和上采样层，防止张量尺寸不匹配
        if "downsample" in name or "upsample" in name:
            continue
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAConv2d(module))

unet.to(device, dtype=torch.float16)
optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=lr)

# ===== 数据 =====
class ImageDataset(Dataset):
    def __init__(self, data_dir, size):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img)}

dataset = ImageDataset(data_dir, image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 提前准备一个空的文本 embedding，解决 missing 'encoder_hidden_states' 的报错
empty_text_inputs = tokenizer([""], return_tensors="pt", padding="max_length", max_length=77).to(device)
with torch.no_grad():
    empty_text_embeddings = text_encoder(empty_text_inputs.input_ids)[0].to(dtype=torch.float16)

# ===== 训练 =====
unet.train()

# 强制 VAE 使用 float32 精度，解决 NaN 
vae.to(device, dtype=torch.float32)

for epoch in range(epochs):
    epoch_start = time.time()
    epoch_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for step, batch in enumerate(progress_bar):
        step_start = time.time()
        bsz = batch["pixel_values"].size(0)

        # 图像数据转入 GPU，使用 float32 喂给 VAE
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)

        # ===== 前向 =====
        with torch.no_grad():
            # 此时 VAE 和 pixel_values 都是 float32，安全编码不会产生 NaN
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
            
        # 编码完成后，降回 float16 给 UNet 使用
        latents = latents.to(dtype=torch.float16)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (bsz,), device=device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 复制空文本向量以匹配 batch_size
        encoder_hidden_states = empty_text_embeddings.repeat(bsz, 1, 1)

        # 传入 noisy_latents, timesteps AND encoder_hidden_states
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 在 float32 下计算 loss 以保稳定
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

        # ===== 反向 =====
        optimizer.zero_grad()
        loss.backward()
        # 加入梯度裁剪，双重保险防 NaN
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
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
