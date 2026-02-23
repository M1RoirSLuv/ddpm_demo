import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as T
import os

# ===== 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt" # VAE 通常集成在单文件 ckpt 中
input_image_path = "ir_rgb_5k/test_01.png" # 选一张你最清晰的红外原图
device = "cuda"

# ===== 1. 加载 VAE =====
# 如果是单文件，我们可以通过 StableDiffusionPipeline 加载后再提取 VAE
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_single_file(ckpt_path, torch_dtype=torch.float32)
vae = pipe.vae.to(device)
vae.eval()

# ===== 2. 图像预处理 =====
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]) # 缩放到 [-1, 1]
])

img = Image.open(input_image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# ===== 3. 执行重建 (VA-VAE 核心逻辑) =====
with torch.no_grad():
    # 编码到潜空间
    # 这里的 0.18215 是 SD 标准的缩放系数
    latents = vae.encode(img_tensor).latent_dist.sample()
    
    # 从潜空间解码回像素
    recon_tensor = vae.decode(latents).sample

# ===== 4. 后处理并保存对比图 =====
def tensor_to_pil(tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2 # 移回到 [0, 1]
    tensor = tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((tensor * 255).astype("uint8"))

recon_img = tensor_to_pil(recon_tensor)

# 拼合原图和重建图进行对比
combined = Image.new('RGB', (512, 256))
combined.paste(img.resize((256, 256)), (0, 0))
combined.paste(recon_img, (256, 0))
combined.save("vae_check_result.png")

print("VAE 重建测试完成！对比图已保存至 vae_check_result.png")
print("左侧为原图，右侧为 VAE 重建图。如果右侧明显模糊，说明你需要微调 VAE。")
