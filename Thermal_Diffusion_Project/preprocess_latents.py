import torch
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

# 1. 自动关联本地 ldm 库路径
# 如果你的 ldm 文件夹在当前目录下，这行是必须的
sys.path.append(os.getcwd())
from ldm.util import instantiate_from_config

def get_vae(config_path, ckpt_path):
    print("🚀 正在加载 VAE (FP16 模式)...")
    config = OmegaConf.load(config_path)
    
    # 【核心修改 1】屏蔽 LPIPS 联网检查
    config.model.params.lossconfig.target = "torch.nn.Identity"
    
    # 实例化模型
    model = instantiate_from_config(config.model)
    
    # 【核心修改 2】解决 PyTorch 2.6+ 的权重加载安全限制 (weights_only=False)
    print(f"📂 读取权重文件: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=False)
    
    # 【核心修改 3】将模型转换为半精度 (FP16) 并移至 GPU
    model = model.cuda().half().eval()
    return model

@torch.no_grad()
def main():
    # --- 配置区域 ---
    IMG_DIR = "./data/raw_images"       # 你的 1280x1024 图片存放文件夹
    SAVE_DIR = "./data/latents"         # 特征向量保存文件夹
    VAE_CONFIG = "vae_config.yaml"
    VAE_CKPT = "model/autoencoder.ckpt"
    # ----------------
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    try:
        vae = get_vae(VAE_CONFIG, VAE_CKPT)
    except Exception as e:
        print(f"❌ 加载 VAE 失败: {e}")
        return
    
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(img_files)} 张图片，开始提取特征...")

    for fname in tqdm(img_files):
        try:
            # 1. 加载并强制转为 RGB (3通道)
            img_path = os.path.join(IMG_DIR, fname)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((1280, 1024)) 
            
            # 2. 归一化并转为 FP16 Tensor
            img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
            # 注意：数据也要转成 .half() 才能与模型匹配
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda().half()
            
            # 3. 执行编码
            # VQModel 编码返回 quant (特征), emb_loss, info
            latent, _, _ = vae.encode(img_tensor)
            
            # 4. 保存 Latent (保存时转回 FP32 可以避免精度累积误差，且占用空间变化不大)
            save_path = os.path.join(SAVE_DIR, os.path.splitext(fname)[0] + ".pt")
            torch.save(latent.squeeze(0).float().cpu(), save_path)
            
        except Exception as e:
            print(f"\n❌ 处理图片 {fname} 时发生错误: {e}")
            continue

    print(f"\n✅ 预处理完成！Latent 文件已保存在: {SAVE_DIR}")

if __name__ == "__main__":
    # 建议在运行前设置此环境变量以减少显存碎片
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    main()