import os, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader

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
    load_safety_checker=False,  
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

# ===== LoRA 注入  =====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        # 使用 1x1 卷积分支，确保任何层级都不会出现 Tensor 尺寸不匹配 (16 vs 32)
        self.down = nn.Conv2d(conv.in_channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, conv.out_channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        
    def forward(self, x, *args, **kwargs):
        # 使用 *args 接收并忽略 UNet 传下来的额外参数，解决参数个数报错
        return self.conv(x) + self.up(self.down(x))

for name, module in list(unet.named_modules()):
    if isinstance(module, nn.Conv2d):
        # 避开降采样层，进一步保证稳定性
        if "downsample" in name or "upsample" in name: continue
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAConv2d(module))

unet.to(device, dtype=torch.float16)
optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=lr)

# ===== 数据  =====
class ImageDataset(Dataset):
    def __init__(self, data_dir, size):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img)}

dataset = ImageDataset(data_dir, image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 预先生成空文本特征，解决 forward() missing encoder_hidden_states 报错
empty_input = tokenizer([""], return_tensors="pt", padding="max_length", max_length=77).to(device)
with torch.no_grad():
    empty_prompt_embeds = text_encoder(empty_input.input_ids)[0].to(dtype=torch.float16)

# ===== 训练  =====
unet.train()

# 核心修复 NaN！将 VAE 设为 float32 精度，它在半精度下极易溢出产生 NaN
vae.to(device, dtype=torch.float32)

for epoch in range(epochs):
    epoch_start = time.time()
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for step, batch in enumerate(progress_bar):
        step_start = time.time()
        bsz = batch["pixel_values"].size(0)

        # VAE 编码前导流。VAE 在 FP32 下运行，输入图像也转 FP32
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)

        # ===== 前向 =====
        with torch.no_grad():
            # 这里的 VAE 逻辑就在这：它把图片变成潜空间 latents
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
            
        # 转回 FP16 给 UNet 使用，节省显存并匹配权重精度
        latents = latents.to(dtype=torch.float16)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 传入准备好的空文本特征，解决参数缺失报错
        encoder_hidden_states = empty_prompt_embeds.repeat(bsz, 1, 1)

        # 执行预测
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 在 float32 精度下算 Loss，保证数值稳定不消失/不溢出
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

        # ===== 反向 =====
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：防止 LoRA 权重在训练中震荡产生 NaN
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        # ===== 统计 =====
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch+1} finished | avg loss = {epoch_loss/len(dataloader):.6f}")

torch.save(unet.state_dict(), os.path.join(out_dir, "lora_unet.pt"))
