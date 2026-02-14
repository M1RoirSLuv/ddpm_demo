import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator
import os
from tqdm import tqdm
import argparse # 用于接收权重路径

class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.files = [os.path.join(latent_dir, f) for f in os.listdir(latent_dir) if f.endswith('.pt')]
        if len(self.files) == 0:
            raise RuntimeError(f"在 {latent_dir} 中没找到任何 .pt 文件！")
        print(f"加载了 {len(self.files)} 个潜空间特征文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="./checkpoints/unet_epoch_10.pth")
    args = parser.parse_args()

    # --- 超参数设置 ---
    LATENT_DIR = "./data/latents"
    OUTPUT_DIR = "./checkpoints"
    BATCH_SIZE = 2 
    GRADIENT_ACCUM = 4 
    LEARNING_RATE = 1e-4
    EPOCHS = 50  # 
    # ------------------

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRADIENT_ACCUM
    )

    dataset = LatentDataset(LATENT_DIR)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet2DModel(
        sample_size=(256, 320),
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 加载第 10 轮的权重并设置起始位置
    start_epoch = 0
    if os.path.exists(args.resume):
        print(f"正在从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)
        start_epoch = 11  # 因为 10 已经跑完了，所以从 11 开始
        print(f"起始点已设为 Epoch {start_epoch}")
    else:
        print(f"未找到权重文件 {args.resume}，将从零开始训练！")

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"开始训练 (从 {start_epoch} 到 {EPOCHS})...")
    
    # 修改循环范围：range(11, 50)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                clean_latents = batch
                noise = torch.randn_like(clean_latents)
                bs = clean_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_latents.device).long()
                noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
                noise_pred = model(noisy_latents, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        # 每 10 轮保存一次，50 结束时也会保存
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(OUTPUT_DIR, f"unet_epoch_{epoch}.pth")
                torch.save(unwrapped_model.state_dict(), save_path)
                print(f"已保存最新权重: {save_path}")

if __name__ == "__main__":
    train()
