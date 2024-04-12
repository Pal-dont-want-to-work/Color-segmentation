import cv2
import numpy as np

def process_image(image_path, output_path, operation='close', kernel_size=5):
    # 读取彩色图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # 将彩色图像分割为蓝色、绿色和红色通道
    blue_channel, green_channel, red_channel = cv2.split(image)
    
    # 创建形态学内核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 对每个通道分别进行形态学操作
    def apply_morphological_operation(channel):
        if operation == 'dilate':
            return cv2.dilate(channel, kernel, iterations=1)
        elif operation == 'erode':
            return cv2.erode(channel, kernel, iterations=1)
        
        # 先腐蚀后膨胀，用来消除小物体、在纤细点处分离物体、平滑较大物体的边界的同时并不明显改变其面积，消除物体表面的突起。
        elif operation == 'open':
            return cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 闭操作就是对图像先膨胀，再腐蚀。闭操作的结果一般是可以将许多靠近的图块相连称为一个无突起的连通域。
        elif operation == 'close':
            return cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 对蓝色、绿色和红色通道分别进行形态学操作
    processed_blue_channel = apply_morphological_operation(blue_channel)
    processed_green_channel = apply_morphological_operation(green_channel)
    processed_red_channel = apply_morphological_operation(red_channel)
    
    # 合并处理后的三个通道
    processed_image = cv2.merge([processed_blue_channel, processed_green_channel, processed_red_channel])
    
    # 保存处理后的彩色图像
    cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    image_path = 'result_image0.png'
    output_path = 'output.jpg'
    process_image(image_path, output_path, operation='erode', kernel_size=9)