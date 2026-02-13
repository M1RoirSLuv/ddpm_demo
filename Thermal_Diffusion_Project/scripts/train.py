import sys
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import ThermalDataset
from src.model_builder import create_model_and_scheduler

def train():
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = config['training']['device']
    os.makedirs("outputs/checkpoints", exist_ok=True)

    dataset = ThermalDataset(config['data']['dataset_path'], image_size=config['data']['image_size'], limit=config['data']['sample_limit'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    model, noise_scheduler = create_model_and_scheduler(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))

    for epoch in range(config['training']['num_epochs']):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for clean_images in progress_bar:
            clean_images = clean_images.to(device)
            noise = torch.randn_like(clean_images).to(device)
            timesteps = torch.randint(0, 1000, (clean_images.shape[0],), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": loss.item()})

        if (epoch + 1) % config['training']['save_interval'] == 0:
            model.save_pretrained(f"outputs/checkpoints/epoch_{epoch+1}")
    print("Training Complete.")

if __name__ == "__main__":
    train()
