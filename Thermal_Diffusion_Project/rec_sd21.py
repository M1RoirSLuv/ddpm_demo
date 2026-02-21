import os, torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from skimage.metrics import structural_similarity as ssim

# ===== 配置 =====
model_id = "stabilityai/stable-diffusion-2-1-base"
lora_path = "sd21_ir_lora/lora_unet.pt"
img_path = "test_ir.png"          # 单张红外图（RGB三通道）
t_start = 100                     # 噪声起点（与你DDPM评估一致）
device = "cuda"

# ===== 加载基座 + LoRA =====
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
pipe.unet.load_state_dict(torch.load(lora_path), strict=False)
vae, unet = pipe.vae, pipe.unet
scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# ===== 预处理 =====
tfm = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
x0 = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

@torch.no_grad()
def reconstruct(x0, t_start):
    # encode → latent
    z0 = vae.encode(x0).latent_dist.sample() * 0.18215

    # 直接采样到 q(z_t | z0)
    noise = torch.randn_like(z0)
    t = torch.full((z0.size(0),), t_start, device=device, dtype=torch.long)
    zt = scheduler.add_noise(z0, noise, t)

    # 逐步去噪
    z = zt
    for i in reversed(range(t_start)):
        ti = torch.full((z.size(0),), i, device=device, dtype=torch.long)
        eps = unet(z, ti, encoder_hidden_states=None).sample
        z = scheduler.step(eps, ti, z).prev_sample

    # decode → image
    x_rec = vae.decode(z / 0.18215).sample
    return x_rec

# ===== 重建 =====
x_rec = reconstruct(x0, t_start)

# ===== 反归一化并计算SSIM =====
def to_np(x):
    x = (x * 0.5 + 0.5).clamp(0,1)
    x = x[0].permute(1,2,0).detach().cpu().numpy()
    return x

gt = to_np(x0)
rc = to_np(x_rec)

# 若你按灰度评估，可转灰度再算
gt_g = np.mean(gt, axis=2)
rc_g = np.mean(rc, axis=2)
score = ssim(gt_g, rc_g, data_range=1.0)
print("SSIM:", score)

Image.fromarray((rc*255).astype("uint8")).save("recon.png")
print("Saved recon.png")
