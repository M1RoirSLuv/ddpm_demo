import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator
import os
from tqdm import tqdm
import argparse


# 1. 定义潜空间数据集
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


# 2. 训练主函数
def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="权重的路径，例如 ./checkpoints/unet_epoch_10.pth")
    args = parser.parse_args()

    # --- 超参数设置 ---
    LATENT_DIR = "./data/latents"
    OUTPUT_DIR = "./checkpoints"
    BATCH_SIZE = 2
    GRADIENT_ACCUM = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    # ------------------

    # 初始化加速器
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRADIENT_ACCUM
    )

    # 数据加载
    dataset = LatentDataset(LATENT_DIR)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化 UNet
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

    # --- 关键修改：加载旧权重 ---
    start_epoch = 0
    if args.resume:
        print(f"正在加载权重: {args.resume}")
        # 加载权重到 CPU (由 accelerator 自动移到 GPU)
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

        # 自动计算开始的轮数
        # 假设文件名是 unet_epoch_10.pth，说明 0-10 跑完了，下次从 11 开始
        try:
            # 解析文件名里的数字
            file_epoch_num = int(args.resume.split("_")[-1].split(".")[0])
            start_epoch = file_epoch_num + 1
            print(f"上次保存的是 Epoch {file_epoch_num}，将从 Epoch {start_epoch} 继续训练！")
        except:
            print("⚠️ 无法从文件名解析 Epoch，将从 0 开始计数（但权重已加载）。")

    # 使用 accelerator 准备所有组件
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"开始训练 (目标: {EPOCHS} Epochs)...")

    # --- 修改循环范围：从 start_epoch 到 100 ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                clean_latents = batch
                noise = torch.randn_like(clean_latents)
                bs = clean_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,),
                                          device=clean_latents.device).long()

                noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
                noise_pred = model(noisy_latents, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        # 定期保存
        if epoch % 10 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(OUTPUT_DIR, f"unet_epoch_{epoch}.pth")
                torch.save(unwrapped_model.state_dict(), save_path)
                print(f"已保存: {save_path}")


if __name__ == "__main__":
    train()