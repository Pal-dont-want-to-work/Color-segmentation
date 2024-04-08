import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
def quantize_colors(image, color_k=5, iter_num=10):
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
        cv2.imwrite(f'mask{idx}.png', mask)

        # 取反蒙版
        inverted_mask = cv2.bitwise_not(mask)
        cv2.imwrite(f'inverted_mask{idx}.png', inverted_mask)
        # 将取反后的蒙版应用于原始图像
        result = cv2.bitwise_and(image, image, mask=inverted_mask)
        
        # 将蒙版之外的部分设为255（白色）
        result[np.where(inverted_mask != 0)] = 255
        result[np.where(inverted_mask == 0)] = image.copy()[np.where(inverted_mask == 0)]
        cv2.imwrite(f'result_image{idx}.png', result)
    # 返回出现次数最多的颜色的蒙版
    return mask_list

quantize_colors(cv2.imread('pic5.png'))


