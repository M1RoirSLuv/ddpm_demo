import os, torch, torch.nn as nn
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

# ===== 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt"
clip_path = "./model/clip-vit-large-patch14"
lora_weight_path = "sd15_ir_lora/lora_unet_best.pt"
device = "cuda"

# ===== 1. 定义 LoRA 结构 (必须一致) =====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        self.down = nn.Conv2d(conv.in_channels, rank, 1, bias=False).to(device, dtype=torch.float32)
        self.up = nn.Conv2d(rank, conv.out_channels, 1, bias=False).to(device, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
    def forward(self, x, *args, **kwargs):
        lora_out = self.up(self.down(x.to(torch.float32))).to(x.dtype)
        return self.conv(x) + lora_out

# ===== 2. 加载 Pipeline =====
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(device, dtype=torch.float16)

pipe = StableDiffusionPipeline.from_single_file(
    ckpt_path, torch_dtype=torch.float16, tokenizer=tokenizer, 
    text_encoder=text_encoder, load_safety_checker=False, local_files_only=True
).to(device)

# ===== 3. 注入 LoRA =====
unet = pipe.unet
for name, module in list(unet.named_modules()):
    if isinstance(module, nn.Conv2d) and not any(n in name for n in ["downsample", "upsample"]):
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAConv2d(module))

unet.load_state_dict(torch.load(lora_weight_path, map_location=device))

# ===== 4. 生成图片 =====
# 因为训练时用的是空提示词，所以这里 prompt 设为空
print("正在从噪声生成红外图像...")
image = pipe(
    prompt="", 
    num_inference_steps=50, 
    guidance_scale=1.5, # 低引导值，因为没有明确的文本指令
    height=256, 
    width=256
).images[0]

image.save("test_prompt2img_result.png")
print("生成完毕，请检查 test_prompt2img_result.png")
