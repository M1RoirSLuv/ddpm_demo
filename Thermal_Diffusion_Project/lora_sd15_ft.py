import os
import torch
import torch.nn as nn
import time
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader # 必须导入
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

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

# ===== 1. 定义 Dataset 和 DataLoader (修复 NameError) =====
class LocalDataset(Dataset):
    def __init__(self, data_dir, size):
        self.paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img)}

dataset = LocalDataset(data_dir, image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===== 2. 加载基座 (修复本地加载与配置报错) =====
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(device, dtype=torch.float16)

print("Loading SD...")
# 关键：load_safety_checker=False 彻底关闭对 HF 的网络依赖
pipe = StableDiffusionPipeline.from_single_file(
    ckpt_path,
    torch_dtype=torch.float16,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    load_safety_checker=False,
    local_files_only=True
).to(device)

unet, vae = pipe.unet, pipe.vae
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# 冻结
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# ===== 3. LoRA 注入 (保持原逻辑) =====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        in_c = conv.in_channels
        out_c = conv.out_channels
        
        # 无论原卷积是什么，LoRA 分路建议使用 1x1 卷积来避免尺寸变动冲突
        # 这样可以保证 self.down(x) 的输出尺寸永远和 x 一致
        self.down = nn.Conv2d(in_c, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, out_c, kernel_size=1, bias=False)
        
        # 初始化
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        
    def forward(self, x, *args, **kwargs):
        # 支路计算：Input -> Down(1x1) -> Up(1x1) -> Result
        # 主路计算：conv(x)
        lora_out = self.up(self.down(x))
        return self.conv(x) + lora_out

# 遍历 UNet 替换层
for name, module in unet.named_modules():
    if isinstance(module, nn.Conv2d):
        # 重点：避开 Downsample 和 Upsample 模块中的卷积，
        # 因为它们通常带有 stride=2，容易引起维度不匹配
        if "downsample" in name or "upsample" in name:
            continue
            
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        
        # 替换层
        setattr(parent, parts[-1], LoRAConv2d(module, rank=4))

# 转换精度
unet.to(device, dtype=torch.float16)

# 修复变量名不一致：opt -> optimizer
optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=lr)

# ===== 4. 训练循环 (修复数据类型冲突) =====
unet.train()
for epoch in range(epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        # 修复：将输入图像转为 float16，否则 vae 会报错
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 这里的 LoRA 训练通常需要 Text Embedding，即使是空文本
        # 为了让代码跑通，我们准备一个空 batch 的文本输入
        inputs = tokenizer([""] * bsz, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
        encoder_hidden_states = text_encoder(inputs.input_ids)[0]

        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({"loss": loss.item()})

torch.save(unet.state_dict(), os.path.join(out_dir, "lora_unet.pt"))
