import torch
import os
import numpy as np
from PIL import Image
from diffusers import UNet2DModel, DDIMScheduler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- 配置区 ---
SCALING_FACTOR = 0.18215 # 建议先按此标准，或用我之前给的脚本计算你的 Std
VAE_CONFIG = "./vae_config.yaml"
VAE_CKPT = "./autoencoder.ckpt"
UNET_WEIGHTS = "./checkpoints/unet_epoch_49.pth"
TEST_IMAGE = "./data/raw_images/test_01.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_ddim_reconstruction():
    # 1. 加载 VAE (强制 CPU 避免显存爆炸)
    config = OmegaConf.load(VAE_CONFIG)
    vae = instantiate_from_config(config.model).to("cpu").eval()
    vae.load_state_dict(torch.load(VAE_CKPT, map_location="cpu")["state_dict"], strict=False)

    # 2. 加载 UNet
    unet = UNet2DModel.from_pretrained(None, config={
        "sample_size": (256, 320), # 注意：这是下采样后的尺寸 (1024/4, 1280/4)
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 2,
        "block_out_channels": (128, 256, 512, 512),
        "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    }).to(DEVICE).eval()
    unet.load_state_dict(torch.load(UNET_WEIGHTS, map_location=DEVICE))

    # 3. 设置 DDIM Scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.0120)
    scheduler.set_timesteps(50) # 50步足够

    # 4. 图像预处理
    img = Image.open(TEST_IMAGE).convert("RGB").resize((1280, 1024))
    img_tensor = (torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0)

    # 5. VAE 编码并缩放
    with torch.no_grad():
        latents = vae.encode(img_tensor).mode() * SCALING_FACTOR
    
    latents = latents.to(DEVICE)
    
    # --- DDIM Inversion (将图片转为确定性噪声) ---
    print("开始 DDIM Inversion...")
    inverted_latents = latents.clone()
    # 逻辑：反向运行 DDIM，从 t=0 到 t=999
    for t in reversed(scheduler.timesteps):
        with torch.no_grad():
            noise_pred = unet(inverted_latents, t).sample
            # 这里使用简化的反向公式
            alpha_prod_t = scheduler.alphas_cumprod[t]
            inverted_latents = (inverted_latents - (1 - alpha_prod_t)**0.5 * noise_pred) / (alpha_prod_t**0.5)

    # --- DDIM Reconstruction (从噪声还原) ---
    print("开始重构生成...")
    curr_latents = inverted_latents
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = unet(curr_latents, t).sample
            curr_latents = scheduler.step(noise_pred, t, curr_latents).prev_sample

    # 6. VAE 解码还原
    with torch.no_grad():
        recon_img_tensor = vae.decode(curr_latents.to("cpu") / SCALING_FACTOR).sample
    
    # 7. 计算指标
    orig = ((img_tensor[0].permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
    recon = ((recon_img_tensor[0].permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
    
    s = ssim(orig, recon, data_range=1.0, channel_axis=-1)
    p = psnr(orig, recon, data_range=1.0)
    
    print(f"✅ 重构完成! SSIM: {s:.4f}, PSNR: {p:.2f}dB")
    
    # 保存对比图
    res = Image.fromarray((recon * 255).astype(np.uint8))
    res.save("ddim_reconstruction_result.png")

if __name__ == "__main__":
    run_ddim_reconstruction()
