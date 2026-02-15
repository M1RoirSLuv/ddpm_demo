import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from diffusers import UNet2DModel, DDPMScheduler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import os


# 1. 加载组件 (参考 test.py 和 train_new.py)
def load_models(vae_config, vae_ckpt, unet_weights, device):
    # 加载 VAE
    config = OmegaConf.load(vae_config)
    vae = instantiate_from_config(config.model)
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu")["state_dict"], strict=False)
    vae.to(device).eval()

    # 加载 UNet
    unet = UNet2DModel(
        sample_size=(256, 320),
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnDownBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    unet.load_state_dict(torch.load(unet_weights, map_location="cpu"))
    unet.to(device).eval()

    return vae, unet


def reconstruct_verify():
    # --- 配置 ---
    IMG_PATH = "data/raw/test.jpg"  # 选一张训练集里的原图
    UNET_WEIGHTS = "./checkpoints/unet_epoch_10.pth"
    VAE_CONFIG = "vae_config.yaml"
    VAE_CKPT = "model/autoencoder.ckpt"
    DEVICE = "cuda"

    vae, unet = load_models(VAE_CONFIG, VAE_CKPT, UNET_WEIGHTS, DEVICE)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 2. 处理原图并编码到潜空间 (参考 test.py)
    img = Image.open(IMG_PATH).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((1024, 1280)),  # 对应 1280x1024
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # 获取原图的潜特征
        clean_latents, _, _ = vae.encode(x)

        # 3. 模拟加噪 (验证重建能力，我们可以加 500 步噪声，看看模型能否还原回去)
        noise = torch.randn_like(clean_latents)
        timesteps = torch.tensor([500]).long().to(DEVICE)  # 设置一个中间步数
        noisy_latents = scheduler.add_noise(clean_latents, noise, timesteps)

        # 4. 逐步去噪 (从第 500 步回到第 0 步)
        print("🔄 正在尝试从噪声重建原图...")
        scheduler.set_timesteps(1000)
        # 过滤出小于等于 500 的时间步
        active_timesteps = scheduler.timesteps[scheduler.timesteps <= 500]

        curr_latents = noisy_latents
        for t in active_timesteps:
            # 预测噪声并去噪
            noise_pred = unet(curr_latents, t).sample
            curr_latents = scheduler.step(noise_pred, t, curr_latents).prev_sample

        # 5. VAE 解码回像素
        reconstruction = vae.decode(curr_latents)
        reconstruction = (reconstruction / 2 + 0.5).clamp(0, 1)

    # 保存对比结果
    res = reconstruction.cpu().permute(0, 2, 3, 1).numpy()[0]
    res_img = Image.fromarray((res * 255).astype(np.uint8))
    res_img.save("reconstruction_verify.png")
    print("✅ 重建验证完成，结果已保存至 reconstruction_verify.png")


if __name__ == "__main__":
    reconstruct_verify()