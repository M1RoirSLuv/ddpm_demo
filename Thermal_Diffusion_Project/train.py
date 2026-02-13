import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator
import os
from tqdm import tqdm


# 1. 定义潜空间数据集
class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        # 自动加载之前预处理好的 .pt 文件
        self.files = [os.path.join(latent_dir, f) for f in os.listdir(latent_dir) if f.endswith('.pt')]
        if len(self.files) == 0:
            raise RuntimeError(f"在 {latent_dir} 中没找到任何 .pt 文件！")
        print(f"📊 加载了 {len(self.files)} 个潜空间特征文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载形状为 [3, 256, 320] 的张量
        return torch.load(self.files[idx])


# 2. 训练主函数
def train():
    # --- 超参数设置 ---
    LATENT_DIR = "./data/latents"
    OUTPUT_DIR = "./checkpoints"
    BATCH_SIZE = 2  # 如果显存够大，可以改为 4
    GRADIENT_ACCUM = 4  # 梯度累积，相当于 Batch Size = 2*4=8
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    # ------------------

    # 初始化加速器 (自动处理 FP16 和多卡环境)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRADIENT_ACCUM
    )

    # 数据加载
    dataset = LatentDataset(LATENT_DIR)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化 UNet
    # sample_size 对应潜空间分辨率 (H, W)
    model = UNet2DModel(
        sample_size=(256, 320),
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),  # 增加通道数以提取复杂特征
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # 在低分辨率层加入 Attention
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 使用 accelerator 准备所有组件
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔥 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                clean_latents = batch  # 形状 [B, 3, 256, 320]

                # 采样噪声
                noise = torch.randn_like(clean_latents)
                bs = clean_latents.shape[0]

                # 随机采样时间步
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,),
                                          device=clean_latents.device).long()

                # 前向加噪
                noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

                # 预测噪声
                noise_pred = model(noisy_latents, timesteps).sample

                # 计算损失 (MSE)
                loss = F.mse_loss(noise_pred, noise)

                # 反向传播
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        # 定期保存
        if epoch % 10 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # 只保存 UNet 权重
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_DIR, f"unet_epoch_{epoch}.pth"))


if __name__ == "__main__":
    train()