import os, torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, StableDiffusionPipeline
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ===== 1. 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt"
data_dir = "ir_rgb_5k"
out_dir = "sd15_ir_vae_finetuned"
os.makedirs(out_dir, exist_ok=True)

image_size = 256
batch_size = 4
lr = 5e-6  # VAE 微调建议使用极小的学习率
epochs = 10
device = "cuda"

# ===== 2. 加载 VAE =====
# 从单文件加载，并提取 VAE
pipe = StableDiffusionPipeline.from_single_file(ckpt_path, torch_dtype=torch.float32)
vae = pipe.vae.to(device)

# 确保 VAE 是可训练的
vae.requires_grad_(True)
vae.train()

# ===== 3. 数据准备 =====
class VAEDataset(Dataset):
    def __init__(self, data_dir, size):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3) # 缩放到 [-1, 1]
        ])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths[idx]).convert("RGB"))

dataloader = DataLoader(VAEDataset(data_dir, image_size), batch_size=batch_size, shuffle=True)

# ===== 4. 优化器与损失函数 =====
optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
# 我们使用 MSE 损失来强制重建图像在像素级与原图一致
criterion = torch.nn.MSELoss()

# ===== 5. 训练循环 =====
print("开始微调 VAE...")
best_loss = float("inf")

for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in progress_bar:
        img = batch.to(device)
        
        # --- 重建流程 ---
        # 1. 编码到 Latent
        posterior = vae.encode(img).latent_dist
        z = posterior.sample()
        
        # 2. 解码回像素
        reconstruction = vae.decode(z).sample
        
        # 3. 计算 $MSE$ Loss
        loss = criterion(reconstruction, img)
        
        # --- 反向传播 ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"mse_loss": f"{loss.item():.6f}"})

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
    
    # 保存最优 VAE
    if avg_loss < best_loss:
        best_loss = avg_loss
        vae.save_pretrained(os.path.join(out_dir, "vae_best"))
        print(f"--- 保存当前最优 VAE 模型 ---")

print("VAE 微调完成！")
