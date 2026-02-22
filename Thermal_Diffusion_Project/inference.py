import os, torch, torch.nn as nn
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer

# ===== 配置  =====
ckpt_path = "./model/v1-5-pruned.ckpt"
clip_path = "./model/clip-vit-large-patch14"
lora_weight_path = "sd15_ir_lora/lora_unet.pt"
device = "cuda"

# 输入一张你数据集里的红外图作为测试
input_ir_image = "ir_rgb_256/test_01.png" # 请确保路径下有图
output_path = "restored_ir_result.png"

# ===== 定义 LoRA  =====
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        self.down = nn.Conv2d(conv.in_channels, rank, 1, bias=False).to(device, dtype=torch.float32)
        self.up = nn.Conv2d(rank, conv.out_channels, 1, bias=False).to(device, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        
    def forward(self, x, *args, **kwargs):
        # 混合精度桥接：在 FP32 算支路，转回 FP16 加回主路
        lora_out = self.up(self.down(x.to(torch.float32))).to(x.dtype)
        return self.conv(x) + lora_out

# ===== 加载 Pipeline =====
print("Initializing Pipeline...")
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(device, dtype=torch.float16)

# 使用 Img2Img 专用管道
pipe = StableDiffusionImg2ImgPipeline.from_single_file(
    ckpt_path,
    torch_dtype=torch.float16,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    load_safety_checker=False,
    local_files_only=True
).to(device)

# ===== 注入 LoRA 并加载权重 =====
print("Injecting LoRA and loading weights...")
unet = pipe.unet
for name, module in list(unet.named_modules()):
    if isinstance(module, nn.Conv2d) and not any(n in name for n in ["downsample", "upsample"]):
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAConv2d(module))

# 加载你训练出的权重
unet.load_state_dict(torch.load(lora_weight_path, map_location=device))
print("Model Ready!")

# ===== 执行推理  =====
if not os.path.exists(input_ir_image):
    print(f"Error: {input_ir_image} 不存在，请指定一张红外图。")
else:
    init_image = Image.open(input_ir_image).convert("RGB").resize((256, 256))
  
    # 0.1 - 0.3: 只加一点点噪，看模型微调能力
    # 0.5 - 0.7: 加较多噪，看模型根据“残影”重构红外特征的能力
    # 0.9: 几乎全是噪，看模型凭空生成的能力
    test_strengths = [0.3, 0.6, 0.8]
    
    results = []
    for s in test_strengths:
        print(f"Generating with strength {s}...")
        # 因为训练是空提示词，所以这里也用 ""
        output = pipe(
            prompt="", 
            image=init_image, 
            strength=s, 
            guidance_scale=1.5, # 稍微给一点点引导
            num_inference_steps=30
        ).images[0]
        results.append(output)

    # 把结果拼在一起对比展示
    combined = Image.new('RGB', (256 * 4, 256))
    combined.paste(init_image, (0, 0)) # 原图
    for i, res in enumerate(results):
        combined.paste(res, (256 * (i+1), 0))
    
    combined.save("comparison_result.png")
    print("实验完成！结果已保存至 comparison_result.png (顺序: 原图, strength 0.3, 0.6, 0.8)")
