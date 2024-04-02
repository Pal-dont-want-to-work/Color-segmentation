import cv2
import numpy as np

# 读取图像
image = cv2.imread('pic.png')
print(image[530][380][0])
print(image[530][380][1])
print(image[530][380][2])
print(image.shape)
height, width, channel = image.shape

# 定义要提取的颜色的 RGB 值
# BGR
# target_color = np.array([0, 34, 224])
target_color = np.array([96, 152, 157])

cnt = 0
new_image = np.ones((height,width), dtype=np.uint8)
for i in range(0, height):
    for j in range(0, width):
        if image[i][j][0] == target_color[0] and image[i][j][1] == target_color[1] and image[i][j][2] == target_color[2]:
            new_image[i][j] = 255
            cnt += 1
        else:
            new_image[i][j] = 0

print(cnt)
# # 计算图像中每个像素与目标颜色的距离
# mask = np.all(image == target_color, axis=-1).astype(np.uint8) * 255
# print(mask)
# print(mask.shape)

# # 显示原始图像和蒙版
# cv2.imshow('Original Image', image)
# cv2.imshow('Mask', mask)
cv2.imshow('Mask', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()