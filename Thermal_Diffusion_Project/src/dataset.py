import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ThermalDataset(Dataset):
    def __init__(self, root_dir, image_size=64, limit=None):
        self.image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff', '.bmp')):
                    self.image_files.append(os.path.join(root, file))

        if limit:
            self.image_files = self.image_files[:limit]

        print(f"Loaded {len(self.image_files)} thermal images.")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        img = Image.open(path).convert('L')
        return self.transform(img)
