import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

def load_model_from_config(config_path, ckpt_path):
    print(f"正在根据配置加载模型: {config_path}")
    config = OmegaConf.load(config_path)
    # 实例化 VQModel
    model = instantiate_from_config(config.model)
    
    print(f"正在加载权重: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()
    return model

def test_ir_vae():
    # 路径配置
    config_path = "vae_config.yaml"
    ckpt_path = "model/autoencoder.ckpt"
    img_path = "data/raw/test.jpg" # 你的 1280x1024 红外图
    
    device = torch.device("cuda")
    model = load_model_from_config(config_path, ckpt_path)

    # 准备图像：强制转 RGB (3通道)
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((1024, 1280)), # 保持你的 1280x1024
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 三通道归一化
    ])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # VQModel 的编码和解码流程
        # 1. 编码到潜空间
        quant, emb_loss, info = model.encode(x)
        print(f"潜空间维度: {quant.shape}") # 应该是 [1, 3, 256, 320]
        
        # 2. 从潜空间解码
        reconstructions = model.decode(quant)

    # 保存结果
    comparison = torch.cat([x, reconstructions], dim=0)
    save_image(comparison * 0.5 + 0.5, "vqvae_reconstruction.png")
    
    mse = torch.nn.functional.mse_loss(x, reconstructions)
    print(f"✅ 重构完成！MSE Loss: {mse.item():.6f}")

if __name__ == "__main__":
    test_ir_vae()