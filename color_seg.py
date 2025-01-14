import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import datetime
from PIL import Image


def get_labels(image, color_k=8, iter_num=10):
    '''
    args:
        image: 输入图像
        color_k: 颜色聚类中心数
        iter_num: 迭代次数
    return:
        mask_list: 返回聚类后的mask列表
    '''
    # 转换颜色空间为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 将图像转换为一维数组形式
    pixels = image_rgb.reshape((-1, 3))
    
    # 使用K均值聚类对颜色进行量化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), color_k, None, criteria, iter_num, cv2.KMEANS_RANDOM_CENTERS)

    return labels


def create_dir2(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    cur_time = get_current_datetime()
    file_dir = os.path.join(os.path.curdir,dir_name, cur_time)
    os.makedirs(file_dir)
    return file_dir


def get_current_datetime():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    return formatted_datetime

def get_mask(labels, image):
    # 统计每个颜色的出现次数
    counts = np.bincount(labels.flatten())

    # 找到出现次数最多的颜色的索引
    # most_common_color_label = np.argmax(counts)
    mask_dir = create_dir2('mask')

    mask_list = []
    for idx,color in enumerate(counts):
        # 创建蒙版
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == idx] = 255
        
        # 将蒙版形状改变为图像形状
        mask = mask.reshape(image.shape[:2])
        mask_list.append(mask)
        mask_filename = os.path.join(mask_dir, f'mask{idx}')
        cv2.imwrite(f'{mask_filename}.png', mask)
    return mask_list

def get_inverted_mask(mask_list):
    inverted_mask_dir = create_dir2('inverted_mask')

    inverted_mask_list = []
    for idx, mask in enumerate(mask_list):
        inverted_mask = cv2.bitwise_not(mask)
        inverted_mask_filename = os.path.join(inverted_mask_dir, f'inverted_mask{idx}')
        cv2.imwrite(f'{inverted_mask_filename}.png', inverted_mask)
        inverted_mask_list.append(inverted_mask)
    return inverted_mask_list   


def quantize_colors(image, inverted_mask_list):

    quantize_dir = create_dir2('quantize_dir')
    res_list = []
    for idx, inverted_mask in enumerate(inverted_mask_list):
        # 将取反后的蒙版应用于原始图像
        result = cv2.bitwise_and(image, image, mask=inverted_mask)
        
        # 将蒙版之外的部分设为255（白色）
        result[np.where(inverted_mask != 0)] = 255
        result[np.where(inverted_mask == 0)] = image.copy()[np.where(inverted_mask == 0)]

        
        quantize_filename = os.path.join(quantize_dir, f'mask{idx}')
        cv2.imwrite(f'{quantize_filename}.png', result)

        # 颜色存在误差问题
        result = Image.open(f'{quantize_filename}.png')
        res_list.append(result)
    return res_list


def get_quantize_colors(image=None, color_k=8, iter_num=10):
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    labels = get_labels(image, color_k, iter_num)
    masks = get_mask(labels, image)
    inverted_masks = get_inverted_mask(masks)
    results = quantize_colors(image, inverted_masks)
    return results

def soft_process(image_path, operation='close先膨胀后腐蚀', kernel_size=5):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # 将彩色图像分割为蓝色、绿色和红色通道
    blue_channel, green_channel, red_channel = cv2.split(image)
    
    # 创建形态学内核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 对每个通道分别进行形态学操作
    def apply_morphological_operation(channel):
        if operation == 'dilate膨胀':
            return cv2.dilate(channel, kernel, iterations=1)
        elif operation == 'erode腐蚀':
            return cv2.erode(channel, kernel, iterations=1)
        
        # 先腐蚀后膨胀，用来消除小物体、在纤细点处分离物体、平滑较大物体的边界的同时并不明显改变其面积，消除物体表面的突起。
        elif operation == 'open先腐蚀后膨胀':
            return cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 闭操作就是对图像先膨胀，再腐蚀。闭操作的结果一般是可以将许多靠近的图块相连称为一个无突起的连通域。
        elif operation == 'close先膨胀后腐蚀':
            return cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 对蓝色、绿色和红色通道分别进行形态学操作
    processed_blue_channel = apply_morphological_operation(blue_channel)
    processed_green_channel = apply_morphological_operation(green_channel)
    processed_red_channel = apply_morphological_operation(red_channel)
    
    # 合并处理后的三个通道
    processed_image = cv2.merge([processed_blue_channel, processed_green_channel, processed_red_channel])
    
    return processed_image

def soft_process_single(image_path, operation='close先膨胀后腐蚀', kernel_size=9):
    if not os.path.exists('soft_process_single'):
        os.makedirs('soft_process_single')
    soft_process_img = soft_process(image_path, operation, kernel_size)
    img_num = len(os.listdir('soft_process_single'))
    cv2.imwrite(f'soft_process_single/soft_process_single{img_num+1}.png', soft_process_img)

    # 颜色存在误差问题
    soft_process_img = Image.open(f'soft_process_single/soft_process_single{img_num+1}.png')
    return [soft_process_img]
        


def soft_process_batch(self, image_path, operation='close', kernel_size=5):
    pass
    
    
if __name__ == "__main__": 
    
    with gr.Blocks(analytics_enabled=False) as extract_pic_interface:
        with gr.Column():
            gr.Interface(get_quantize_colors, 
                         [gr.Image(label="输入图片", type='filepath'),
                            gr.Slider(1,10,step=1, label="图片颜色种类", interactive=True),
                            gr.Slider(10,50,step=1, label="迭代次数(分辨率太大的，建议迭代次数控制在30次以内)", interactive=True)], 
                        outputs=gr.Gallery(label="生成结果：",allow_preview=True))
            
    with gr.Blocks(analytics_enabled=False) as soft_process_interface:
        with gr.Column():
            gr.Interface(soft_process_single,
                        [gr.Image(label="输入图片", type='filepath'),
                         gr.Radio(['dilate膨胀', 'erode腐蚀', 'open先腐蚀后膨胀', 'close先膨胀后腐蚀'], label="操作类型"),
                         gr.Radio([3,5,7,9,11], label="内核大小")], 
                        outputs=gr.Gallery(label="生成结果：",allow_preview=True))
                
    interfaces = [
        (extract_pic_interface, '颜色分层', 'extract_pic'),
        (soft_process_interface, '颜色平滑', 'soft_process')
    ]

    with gr.Blocks(analytics_enabled=False) as demo:
        for interface, label,ifid in interfaces:
            with gr.TabItem(label, id=ifid,elem_id=f'tab_{ifid}'):
                interface.render()
    
    demo.launch(share=True)
