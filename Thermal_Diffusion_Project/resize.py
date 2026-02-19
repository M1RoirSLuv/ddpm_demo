import os
from PIL import Image

input_dir = "./ir_images"        # 原始红外图目录
output_dir = "./ir_256"          # 输出目录
size = 256

os.makedirs(output_dir, exist_ok=True)

def process_image(path, save_path):
    img = Image.open(path).convert("L")

    # 保持比例缩放到最小边=256
    w, h = img.size
    scale = size / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = img.resize((new_w, new_h), Image.BICUBIC)

    # 中心裁剪 256x256
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    img.save(save_path)

for name in os.listdir(input_dir):
    src = os.path.join(input_dir, name)
    dst = os.path.join(output_dir, name)

    try:
        process_image(src, dst)
        print("processed:", name)
    except Exception as e:
        print("error:", name, e)

print("done")
