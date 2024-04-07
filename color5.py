

import cv2
import numpy as np
import gradio as gr
def quantize_colors(image, color_k=8, iter_num=10):
    '''
        args:
            image: 输入图像
            color_k: 颜色聚类中心数
            iter_num: 迭代次数
    '''
    # 转换颜色空间为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 将图像转换为一维数组形式
    pixels = image_rgb.reshape((-1, 3))
    
    # 使用K均值聚类对颜色进行量化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    print(criteria)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), color_k, None, criteria, iter_num, cv2.KMEANS_RANDOM_CENTERS)
    
    # 统计每个颜色的出现次数
    counts = np.bincount(labels.flatten())
    # print(counts)
    # 找到出现次数最多的颜色的索引
    # most_common_color_label = np.argmax(counts)
    mask_list = []
    for idx,color in enumerate(counts):
        # 创建蒙版
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == idx] = 255
        
        # 将蒙版形状改变为图像形状
        mask = mask.reshape(image.shape[:2])
        mask_list.append(mask)
        cv2.imwrite(f'output_image{idx}.jpg', mask)
    # 返回出现次数最多的颜色的蒙版
    return mask_list

# 加载图像
# image = cv2.imread('pic4.jpg')

# 对图像进行颜色量化并获取出现次数最多的颜色的蒙版
# color_mask = quantize_colors(image)

# # 显示原始图像和出现次数最多的颜色的蒙版
# cv2.imshow('Original Image', image)
# cv2.imshow('Most Common Color Mask', color_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Interface(quantize_colors, 
                         [gr.Image(),
                            gr.Slider(1,10,step=1, label="图片颜色种类", interactive=True),
                            gr.Slider(10,50,step=1, label="迭代次数(分辨率太大的，建议迭代次数控制在30次以内)", interactive=True)], 
                        outputs=gr.Gallery(label="生成结果：",allow_preview=True))
    demo.launch(share=True)
