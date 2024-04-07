import cv2
import numpy as np
from collections import Counter
import time
import gradio as gr


def timedecorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f'Cost time: {time.time()-start}')
    return wrapper

@timedecorator
def process_image(image_path, num_top_colors=6, eps=10):
    # 读取图像
    image = cv2.imread(image_path)

    height, width, _ = image.shape

    # 将图像转换为一维数组
    flattened_image = image.reshape(-1, 3)
    print(flattened_image.shape)

    print(Counter(map(tuple, flattened_image)))
    # 计算最常见的颜色
    colors_set_top_n = Counter(map(tuple, flattened_image)).most_common(num_top_colors)

    # 遍历每个常见颜色
    for idx, target_color in enumerate(colors_set_top_n):
        target_color = np.array(target_color[0])  # 目标颜色

        # 计算每个像素与目标颜色的差异
        diff = np.abs(flattened_image - target_color)
        total_diff = np.sum(diff, axis=1)

        # 创建掩码
        mask = np.where(total_diff < eps, 255, 0).reshape(height, width).astype(np.uint8)

        # 保存掩码
        cv2.imwrite(f'output_image{idx}.jpg', mask)
        print(f'cnt:{np.sum(mask)}')

@timedecorator
def process_image2(image, num_top_colors=6, eps=10):

    height, width, _ = image.shape

    # 将图像转换为一维数组
    flattened_image = image.reshape(-1, 3)
    print(flattened_image.shape)

    print(Counter(map(tuple, flattened_image)))
    # 计算最常见的颜色
    colors_set_top_n = Counter(map(tuple, flattened_image)).most_common(num_top_colors)
    pic_result_list = []
    # 遍历每个常见颜色
    for idx, target_color in enumerate(colors_set_top_n):
        target_color = np.array(target_color[0])  # 目标颜色

        # 计算每个像素与目标颜色的差异
        diff = np.abs(flattened_image - target_color)
        total_diff = np.sum(diff, axis=1)

        # 创建掩码
        mask = np.where(total_diff < eps, 255, 0).reshape(height, width).astype(np.uint8)
        pic_result_list.append(mask)
        # 保存掩码
        cv2.imwrite(f'output_image{idx}.jpg', mask)
        print(f'cnt:{np.sum(mask)}')
    return pic_result_list

def get_img(img):
    print(img)
    print(type(img))
    res = process_image2(img)
    return res

if __name__ == "__main__":
    
    # demo = gr.Interface(get_img, gr.Image(), "image")
    demo = gr.Interface(get_img, gr.Image(), gr.Gallery())
    demo.launch(debug=True)
    # process_image('pic3.png')