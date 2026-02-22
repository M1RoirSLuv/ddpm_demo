import os
import shutil
from tqdm import tqdm

# ===== 配置路径 =====
src_dir = "ir_rgb_10k"       # 你的原始 10k 图片文件夹
dst_dir = "ir_rgb_5k"        # 准备存放前 5k 张图片的新文件夹
limit = 5000                 # 复制的数量限制

# 如果目标文件夹不存在，则创建它
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
    print(f"创建目标文件夹: {dst_dir}")

# 获取所有图片文件列表并排序（确保每次运行结果一致）
# 你可以根据需要修改后缀名限制，如 .jpg, .png 等
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(valid_extensions)])

print(f"在源文件夹中找到 {len(files)} 张图片。")

# 执行复制操作
count = 0
for i in tqdm(range(min(limit, len(files))), desc="正在复制图片"):
    filename = files[i]
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)
    
    shutil.copy2(src_path, dst_path) # copy2 会保留原始文件的元数据（如修改时间）
    count += 1

print(f"复制完成！共将 {count} 张图片复制到了 {dst_dir}。")
