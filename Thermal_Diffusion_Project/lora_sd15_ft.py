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
lr = 5e-5 
epochs = 5
device = "cuda"

# ===== 1. 加载基座 =====
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(device, dtype=torch.float16)

print("Loading SD (Offline Mode)")
pipe = StableDiffusionPipeline.from_single_file(
    ckpt_path,
    torch_dtype=torch.float16, # 基座保持 FP16
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
unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# ===== 2. LoRA 注入 (强制使用 FP32) =====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        # 修改点：显式指定 LoRA 层使用 float32 精度，这能解决 unscale 报错
        self.down = nn.Conv2d(conv.in_channels, rank, 1, bias=False).to(dtype=torch.float32)
        self.up = nn.Conv2d(rank, conv.out_channels, 1, bias=False).to(dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        
    def forward(self, x, *args, **kwargs):
        # 将输入 x 转为 float32 与 LoRA 权重计算，再转回 x 的原始精度 (fp16)
        orig_dtype = x.dtype
        lora_out = self.up(self.down(x.to(torch.float32)))
        return self.conv(x) + lora_out.to(orig_dtype)

# 注入 LoRA
for name, module in list(unet.named_modules()):
    if isinstance(module, nn.Conv2d):
        if "downsample" in name or "upsample" in name: continue
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]: parent = getattr(parent, p)
        # 替换为 FP32 LoRA 层
        setattr(parent, parts[-1], LoRAConv2d(module))

# 注意：这里千万不要再执行 unet.to(torch.float16) 了！

# ===== 3. 准备训练  =====
# 优化器只负责 LoRA 的 FP32 参数
trainable_params = [p for p in unet.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=lr, eps=1e-8)
scaler = torch.cuda.amp.GradScaler() # 此时 scaler 会正常工作

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

# 预生成空文本特征 (FP16)
empty_input = tokenizer([""], return_tensors="pt", padding="max_length", max_length=77).to(device)
with torch.no_grad():
    empty_prompt_embeds = text_encoder(empty_input.input_ids)[0].to(dtype=torch.float16)

# ===== 5. 训练循环 =====
unet.train()
# VAE 在 FP32 下最稳定
vae.to(device, dtype=torch.float32)

for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in progress_bar:
        bsz = batch["pixel_values"].size(0)
        
        # 开启自动混合精度
        with torch.cuda.amp.autocast():
            # 1. VAE 编码 (FP32)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
            
            latents = latents.to(dtype=torch.float16)

            # 2. 扩散
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = empty_prompt_embeds.repeat(bsz, 1, 1)

            # 3. UNet 预测 (内部会自动处理 FP16 基座和 FP32 LoRA 的交互)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # ===== 4. 优化  =====
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer) # 现在不会报错了，因为 optimizer 里的参数是 FP32
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch+1} avg loss = {epoch_loss/len(dataloader):.6f}")

torch.save(unet.state_dict(), os.path.join(out_dir, "lora_unet.pt"))
print("Done.")
