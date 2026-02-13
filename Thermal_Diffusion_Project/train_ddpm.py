import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import os
from tqdm import tqdm

# 1. 定义潜空间数据集
class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.files = [os.path.join(latent_dir, f) for f in os.listdir(latent_dir) if f.endswith('.pt')]
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        # 加载 [3, 256, 320] 的 Latent
        return torch.load(self.files[idx])

# 2. 训练配置
def train():
    # --- 参数设置 ---
    LATENT_DIR = "./data/latents"
    BATCH_SIZE = 4   # 1280x1024 经过 f=4 后依然很大，bs 不宜过大
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    DEVICE = "cuda"
    # ----------------

    dataset = LatentDataset(LATENT_DIR)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 初始化 UNet (针对 3通道, 256x320 分辨率)
    model = UNet2DModel(
        sample_size=(256, 320), 
        in_channels=3, 
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", 
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", 
            "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 4. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            clean_latents = batch.to(DEVICE)
            
            # 采样噪声
            noise = torch.randn_like(clean_latents)
            bs = clean_latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=DEVICE).long()
            
            # 向潜空间添加噪声
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
            
            # 预测噪声
            noise_pred = model(noisy_latents, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        # 每 10 个 Epoch 保存一次模型
        if epoch % 10 == 0:
            model.save_pretrained(f"sd_model_epoch_{epoch}")

if __name__ == "__main__":
    train()