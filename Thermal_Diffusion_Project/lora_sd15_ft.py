import os, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ===== 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt"
clip_path = "./model/clip-vit-large-patch14"
data_dir = "ir_rgb_256"
out_dir = "sd21_ir_lora"
os.makedirs(out_dir, exist_ok=True)

image_size = 256
batch_size = 4
lr = 1e-4
epochs = 5
device = "cuda"

# ===== 加载基座 =====
tokenizer = CLIPTokenizer.from_pretrained(clip_path)
text_encoder = CLIPTextModel.from_pretrained(clip_path)

pipe = StableDiffusionPipeline.from_single_file(
    ckpt_path,
    torch_dtype=torch.float16,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    safety_checker=None
).to("cuda")
unet, vae = pipe.unet, pipe.vae
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# 冻结原始权重
for p in unet.parameters():
    p.requires_grad = False

# ===== LoRA 注入（简化版：替换Conv2d）=====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        in_c, out_c = conv.in_channels, conv.out_channels
        k = conv.kernel_size[0]
        self.down = nn.Conv2d(in_c, rank, k, padding=conv.padding, bias=False)
        self.up = nn.Conv2d(rank, out_c, 1, bias=False)
    def forward(self, x):
        return self.conv(x) + self.up(self.down(x))

for name, module in list(unet.named_modules()):
    if isinstance(module, nn.Conv2d):
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAConv2d(module))

opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=lr)

# ===== 数据 =====
tfm = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

imgs = []
for f in os.listdir(data_dir):
    imgs.append(tfm(Image.open(os.path.join(data_dir, f)).convert("RGB")))
dataset = torch.stack(imgs)

# ===== 训练 =====
unet.train()
for ep in range(epochs):
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    for i in range(0, len(dataset), batch_size):
        x = dataset[i:i+batch_size].to(device)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.sample() * 0.18215

        noise = torch.randn_like(z)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                          (z.size(0),), device=device).long()
        zt = noise_scheduler.add_noise(z, noise, t)

        pred = unet(zt, t, encoder_hidden_states=None).sample
        loss = (noise - pred).pow(2).mean()

        opt.zero_grad(); loss.backward(); opt.step()

    print(f"epoch {ep} loss {loss.item():.4f}")

torch.save(unet.state_dict(), os.path.join(out_dir, "lora_unet.pt"))
print("Saved:", os.path.join(out_dir, "lora_unet.pt"))
