import sys
import os
import yaml
import torch
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import ThermalDataset

def reconstruct_demo(checkpoint_path, t_start=500):
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet2DModel.from_pretrained(checkpoint_path).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    dataset = ThermalDataset(config['data']['dataset_path'], image_size=config['data']['image_size'], limit=10)
    x_0 = dataset[0].unsqueeze(0).to(device)

    noise = torch.randn_like(x_0)
    timesteps = torch.tensor([t_start], device=device).long()
    x_t = scheduler.add_noise(x_0, noise, timesteps)

    curr_image = x_t
    model.eval()
    for t in range(t_start, -1, -1):
        t_tensor = torch.tensor([t], device=device).long()
        with torch.no_grad():
            model_output = model(curr_image, t_tensor).sample
            curr_image = scheduler.step(model_output, t, curr_image).prev_sample

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(((x_0.cpu() + 1) / 2).squeeze(), cmap='gray'); axs[0].set_title("Original")
    axs[1].imshow(((x_t.cpu() + 1) / 2).squeeze(), cmap='gray'); axs[1].set_title(f"Noisy t={t_start}")
    axs[2].imshow(((curr_image.cpu() + 1) / 2).squeeze(), cmap='gray'); axs[2].set_title("Reconstructed")
    plt.savefig("reconstruction_result.png")
    print("Result saved to reconstruction_result.png")

if __name__ == "__main__":
    # 请确保路径指向实际存在的 checkpoint 文件夹
    # reconstruct_demo("outputs/checkpoints/epoch_50", t_start=500)
    print("Please run this after training and point to a valid checkpoint.")
