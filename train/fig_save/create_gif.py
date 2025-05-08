import imageio
import os

def create_gif_from_images(image_folder, gif_path, fps=10):
    # 获取文件夹中的所有图片文件
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.png')]
    
    # 对图片文件进行排序，确保顺序正确
    images.sort(key=lambda x: int(os.path.basename(x).split('.')[0].zfill(3)))
    
    # 读取图片并创建GIF
    with imageio.get_writer(gif_path, mode='I', duration=1/fps) as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    # 打印完成信息
    print(f"GIF created at {gif_path}")

# 使用示例
image_folder = 'train/fig_save/16UAV_1'  # 图片所在的文件夹路径
gif_path = 'train/gif/3_drones.gif'  # 输出GIF的路径
fps = 10  # 每秒帧数
create_gif_from_images(image_folder, gif_path, fps)