import cv2
import numpy as np
from collections import Counter

# 读取图像
image = cv2.imread('pic.png')
print('B:',image[530][380][0])
print('G:',image[530][380][1])
print('R:',image[530][380][2])
print(image.shape)

height, width, channel = image.shape


# dup_set = set()
color_list = []
for i in range(0, height):
    for j in range(0, width):
        # dup_set.add((image[i][j][0], image[i][j][1], image[i][j][2]))
        color_list.append((image[i][j][0], image[i][j][1], image[i][j][2]))

colors = Counter(color_list)
color_set = colors.most_common(6)

# print(dup_set)
print(len(color_set))
print(color_set)

# 定义要提取的颜色的 RGB 值
# cv2 读取的图片通道是BGR
# target_color = np.array([0, 34, 224])
# target_color = np.array([96, 152, 157])
# target_color = np.array([89, 133, 154])


dup_set = [color[0] for color in color_set]
print(dup_set)
eps = 10
for idx,target_color in enumerate(dup_set):

    cnt = 0
    new_image = np.ones((height,width), dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            if image[i][j][0] == target_color[0] and image[i][j][1] == target_color[1] and image[i][j][2] == target_color[2]:
                new_image[i][j] = 0
                cnt += 1
            else:
                new_image[i][j] = 255

    cv2.imwrite(f'output_image{idx}.jpg', new_image)
    print(f'cnt:{cnt}')


# # 计算图像中每个像素与目标颜色的距离
# mask = np.all(image == target_color, axis=-1).astype(np.uint8) * 255
# print(mask)
# print(mask.shape)

# # 显示原始图像和蒙版
# cv2.imshow('Original Image', image)
# cv2.imshow('Mask', mask)
# cv2.imshow('Mask', new_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()