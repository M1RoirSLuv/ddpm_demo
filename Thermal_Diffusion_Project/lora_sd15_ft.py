import os, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader

# ===== 路径配置  =====
ckpt_path = "./model/v1-5-pruned.ckpt"
clip_path = "./model/clip-vit-large-patch14"
data_dir = "ir_rgb_256"
out_dir = "sd15_ir_lora"
os.makedirs(out_dir, exist_ok=True)

image_size = 256
batch_size = 4
lr = 5e-5 
epochs = 5
device = "cuda"

# ===== 1. 加载基座  =====
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(device, dtype=torch.float16)

pipe = StableDiffusionPipeline.from_single_file(
    ckpt_path,
    torch_dtype=torch.float16, # 基座固定 FP16 节省显存
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    load_safety_checker=False,
    local_files_only=True
).to(device)

unet, vae = pipe.unet, pipe.vae
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# 冻结
unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# ===== 2. LoRA 注入  =====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        # LoRA 权重强制 FP32，解决 GradScaler unscale 报错
        self.down = nn.Conv2d(conv.in_channels, rank, 1, bias=False).to(device, dtype=torch.float32)
        self.up = nn.Conv2d(rank, conv.out_channels, 1, bias=False).to(device, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        
    def forward(self, x, *args, **kwargs):
        # 支路用 FP32 算，结果转回输入 x 的精度
        lora_out = self.up(self.down(x.to(torch.float32))).to(x.dtype)
        return self.conv(x) + lora_out

for name, module in list(unet.named_modules()):
    if isinstance(module, nn.Conv2d) and not any(n in name for n in ["downsample", "upsample"]):
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAConv2d(module))

# ===== 3. 训练器配置  =====
trainable_params = [p for p in unet.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=lr, eps=1e-8)
scaler = torch.cuda.amp.GradScaler() # 它可以愉快地处理 FP32 梯度了

# ===== 4. 数据 & 文本特征 =====
class ImageDataset(Dataset):
    def __init__(self, data_dir, size):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img)}

dataloader = DataLoader(ImageDataset(data_dir, image_size), batch_size=batch_size, shuffle=True)

# 文本特征对齐 FP16 基座
with torch.no_grad():
    empty_prompt_embeds = text_encoder(tokenizer([""], return_tensors="pt").input_ids.to(device))[0].to(dtype=torch.float16)

# ===== 5. 训练循环 =====
unet.train()
vae.to(device, dtype=torch.float32) # VAE 依然保持 FP32 防止编码 NaN

for epoch in range(epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        # 1. 自动混合精度
        with torch.cuda.amp.autocast():
            # VAE 编码
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            latents = latents.to(dtype=torch.float16)

            # 扩散
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 预测 (此时 UNet 内部会处理 FP16 基座与 FP32 LoRA 的交互)
            noise_pred = unet(noisy_latents, timesteps, empty_prompt_embeds.repeat(latents.size(0), 1, 1)).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # 2. 优化
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

torch.save(unet.state_dict(), os.path.join(out_dir, "lora_unet.pt"))
