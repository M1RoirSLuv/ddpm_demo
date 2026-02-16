import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from diffusers import UNet2DModel, DDPMScheduler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import os

# 1. 加载组件
def load_models(vae_config, vae_ckpt, unet_weights):
    device_gpu = "cuda"
    device_cpu = "cpu"

    # --- VAE 加载到 CPU ---
    print(f"正在加载 VAE 配置: {vae_config}")
    config = OmegaConf.load(vae_config)
    vae = instantiate_from_config(config.model)
    
    print(f"正在加载 VAE 权重: {vae_ckpt}")
    state_dict = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
    vae.load_state_dict(state_dict, strict=False)
    vae.to(device_cpu).eval() # 关键：强制放 CPU
    print("VAE 已加载到 CPU")

    # --- UNet 加载到 GPU ---
    print(f"正在加载 UNet: {unet_weights}")
    unet = UNet2DModel(
        sample_size=(256, 320),
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    unet.load_state_dict(torch.load(unet_weights, map_location="cpu"))
    unet.to(device_gpu).eval()
    print(" UNet 已加载到 GPU")
    
    return vae, unet

def reconstruct_verify():
    # --- 配置 ---
    IMG_PATH = "data/raw/test.jpg"  
    UNET_WEIGHTS = "./checkpoints/unet_epoch_49.pth"
    VAE_CONFIG = "vae_config.yaml"
    VAE_CKPT = "model/autoencoder.ckpt"
    
    # 显式指定设备
    GPU = "cuda"
    CPU = "cpu"
    
    if not os.path.exists(IMG_PATH):
        print(f"错误: 找不到测试图片 {IMG_PATH}")
        return

    # 加载模型
    vae, unet = load_models(VAE_CONFIG, VAE_CKPT, UNET_WEIGHTS)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 2. 准备原图 (在 CPU 处理)
    print(f"正在处理图片: {IMG_PATH}")
    img = Image.open(IMG_PATH).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((1024, 1280)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # 图片保留在 CPU
    x = transform(img).unsqueeze(0).to(CPU)
    
    # 3. VAE 编码 (在 CPU 运行)
    print(" 正在 CPU 上进行 VAE 编码 (可能需要几秒钟)...")
    with torch.no_grad():
        encoder_posterior = vae.encode(x)
        if hasattr(encoder_posterior, 'mode'):
            clean_latents = encoder_posterior.mode()
        else:
            clean_latents = encoder_posterior[0] if isinstance(encoder_posterior, tuple) else encoder_posterior

    # 4. 转移到 GPU 进行加噪和去噪
    print(" 转移到 GPU 进行扩散去噪...")
    clean_latents = clean_latents.to(GPU) # 转移到显卡
    
    # 加噪 (50% 强度)
    noise = torch.randn_like(clean_latents)
    timesteps = torch.tensor([500]).long().to(GPU)
    noisy_latents = scheduler.add_noise(clean_latents, noise, timesteps)

    # UNet 去噪循环
    scheduler.set_timesteps(1000)
    start_step = 1000 - 500
    
    curr_latents = noisy_latents
    for t in scheduler.timesteps[start_step:]:
        with torch.no_grad():
            noise_pred = unet(curr_latents, t).sample
            curr_latents = scheduler.step(noise_pred, t, curr_latents).prev_sample

    # 5. 转移回 CPU 进行解码
    print(" 转移回 CPU 进行最终解码...")
    curr_latents = curr_latents.to(CPU) # 搬回内存
    
    with torch.no_grad():
        reconstruction = vae.decode(curr_latents)
        reconstruction = (reconstruction / 2 + 0.5).clamp(0, 1)
        
    # 保存结果
    res = reconstruction.permute(0, 2, 3, 1).numpy()[0]
    res_img = Image.fromarray((res * 255).astype(np.uint8))
    res_img.save("reconstruction_verify.png")
    print(" 成功！结果已保存至 reconstruction_verify.png")

if __name__ == "__main__":
    reconstruct_verify()
