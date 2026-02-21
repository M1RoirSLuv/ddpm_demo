import os, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader

# ===== 配置  =====
ckpt_path = "./model/v1-5-pruned.ckpt"
clip_path = "./model/clip-vit-large-patch14"
data_dir = "./model/raw_256"
out_dir = "sd15_ir_lora"
os.makedirs(out_dir, exist_ok=True)

image_size = 256
batch_size = 4
lr = 5e-5  # 适当降低学习率
epochs = 5
device = "cuda"

# ===== 1. 加载基座  =====
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(device)

print("Loading SD (Offline Mode)")
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
for p in unet.parameters(): p.requires_grad = False
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# ===== 2. LoRA 注入 =====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        # 使用 1x1 分支，既能训练特征又不会导致尺寸对齐报错
        self.down = nn.Conv2d(conv.in_channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, conv.out_channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        
    def forward(self, x, *args, **kwargs):
        # *args 用于接收并忽略 UNet 传下来的 timestep 和 context
        return self.conv(x) + self.up(self.down(x))

for name, module in list(unet.named_modules()):
    if isinstance(module, nn.Conv2d):
        if "downsample" in name or "upsample" in name: continue
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAConv2d(module))

unet.to(device, dtype=torch.float16)

# ===== 3. 准备混合精度训练 (AMP) =====
# eps=1e-8 增加数值稳定性，防止 AdamW 内部除零导致 NaN
optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=lr, eps=1e-8)
# 引入梯度缩放器，解决半精度训练下的梯度消失/溢出问题
scaler = torch.cuda.amp.GradScaler()

# ===== 4. 数据加载  =====
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

# 预先生成空文本特征 
empty_input = tokenizer([""], return_tensors="pt", padding="max_length", max_length=77).to(device)
with torch.no_grad():
    empty_prompt_embeds = text_encoder(empty_input.input_ids)[0].to(dtype=torch.float16)

# ===== 5. 训练循环 =====
unet.train()
# 【核心修复】VAE 必须在 FP32 下编码才不会 NaN，且只需在训练开始前转换一次，不影响速度
vae.to(device, dtype=torch.float32)

for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in progress_bar:
        bsz = batch["pixel_values"].size(0)
        
        # 使用 autocast 开启自动混合精度
        with torch.cuda.amp.autocast():
            # VAE 编码 (使用 float32 保证 latents 不会溢出成 NaN)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
            
            # 转回 float16 给 UNet，保持精度一致
            latents = latents.to(dtype=torch.float16)

            # 3. 扩散噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 文本条件
            encoder_hidden_states = empty_prompt_embeds.repeat(bsz, 1, 1)

            # 4. 前向预测
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # ===== 5. 优化步骤  =====
        optimizer.zero_grad()
        # 缩放损失并回传
        scaler.scale(loss).backward()
        
        # 梯度裁剪 
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        
        # 步进并更新缩放因子
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch+1} avg loss = {epoch_loss/len(dataloader):.6f}")

torch.save(unet.state_dict(), os.path.join(out_dir, "lora_unet.pt"))
print("Done.") 
