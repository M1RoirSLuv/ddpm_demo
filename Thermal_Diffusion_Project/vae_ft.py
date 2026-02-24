import os, torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import lpips  # 新增：感知损失库

# ===== 1. 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt"
data_dir = "./data/raw_5k"
out_dir = "sd15_ir_vae_lpips"
os.makedirs(out_dir, exist_ok=True)

image_size = 256
batch_size = 2   # 引入 LPIPS 后显存占用增加，建议调小 batch_size
lr = 5e-6
epochs = 10
device = "cuda"

# ===== 2. 加载模型 =====
vae = AutoencoderKL.from_single_file(ckpt_path, torch_dtype=torch.float32).to(device)
vae.requires_grad_(True)
vae.train()

# 新增：初始化 LPIPS 模型
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) 

# ===== 3. 数据准备 (同之前) =====
class VAEDataset(Dataset):
    def __init__(self, data_dir, size):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3) # LPIPS 默认预期 [-1, 1] 范围
        ])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths[idx]).convert("RGB"))

dataloader = DataLoader(VAEDataset(data_dir, image_size), batch_size=batch_size, shuffle=True)

# ===== 4. 优化器 =====
optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
mse_criterion = torch.nn.MSELoss()

# ===== 5. 训练循环 =====
print(f"开始 LPIPS 增强训练... 共 {len(dataloader.dataset)} 张图片")
best_loss = float("inf")

for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in progress_bar:
        img = batch.to(device)
        
        # 重建流程
        latents = vae.encode(img).latent_dist.sample()
        reconstruction = vae.decode(latents).sample
        
        # --- 组合损失函数 ---
        # 1. MSE Loss (像素级准确性)
        mse_loss = mse_criterion(reconstruction, img)
        
        # 2. LPIPS Loss (感知级清晰度)
        # lpips 计算结果通常在 0.1-0.5 之间，可以加一个权重系数
        lpips_loss = loss_fn_vgg(reconstruction, img).mean()
        
        # 总损失：1.0 * MSE + 0.1 * LPIPS (这是一个经典的平衡比例)
        total_loss = mse_loss + 0.1 * lpips_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        progress_bar.set_postfix({
            "mse": f"{mse_loss.item():.4f}", 
            "lpips": f"{lpips_loss.item():.4f}"
        })

    avg_loss = epoch_loss / len(dataloader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        vae.save_pretrained(os.path.join(out_dir, "vae_best_lpips"))
        print(f"--- 性能提升！已保存 ---")

print("VAE LPIPS 微调完成！")
