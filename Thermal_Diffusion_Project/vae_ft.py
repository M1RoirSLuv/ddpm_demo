import os, torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ===== 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt"
data_dir = "./data/raw_5k"      # 使用你的 5k 数据集
out_dir = "sd15_ir_vae_finetuned"
os.makedirs(out_dir, exist_ok=True)

image_size = 256
batch_size = 4
lr = 5e-6          # VAE 非常脆弱，保持极低学习率
epochs = 10        # 建议先跑 10 轮观察效果
device = "cuda"

# ===== 加载 VAE  =====
print(f"正在从 {ckpt_path} 直接加载 VAE...")
# 必须使用 FP32 训练以保证数值稳定性
vae = AutoencoderKL.from_single_file(
    ckpt_path, 
    torch_dtype=torch.float32
).to(device)

vae.requires_grad_(True)
vae.train()

# ===== 数据准备 =====
class VAEDataset(Dataset):
    def __init__(self, data_dir, size):
        # 自动过滤无效文件
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths[idx]).convert("RGB"))

dataloader = DataLoader(VAEDataset(data_dir, image_size), batch_size=batch_size, shuffle=True)

# ===== 优化器与 Loss =====
optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
criterion = torch.nn.MSELoss() # 强制像素级一致

# ===== 训练循环 =====
print(f"开始微调... 共 {len(self.image_paths)} 张图片")
best_loss = float("inf")

for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in progress_bar:
        img = batch.to(device)
        
        # 编码并采样潜变量
        # 0.18215 是 SD 1.5 默认的缩放系数，保持一致
        latents = vae.encode(img).latent_dist.sample()
        
        # 解码重建
        reconstruction = vae.decode(latents).sample
        
        # 计算像素级差异
        loss = criterion(reconstruction, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"mse": f"{loss.item():.6f}"})

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
    
    # 自动保存最优模型 (diffusers 格式)
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_path = os.path.join(out_dir, "vae_best")
        vae.save_pretrained(save_path)
        print(f"--- 性能提升！已保存至 {save_path} ---")

print("VAE 微调任务圆满完成！")
