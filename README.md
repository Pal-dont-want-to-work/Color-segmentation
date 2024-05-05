# 基于聚类的颜色分类

## 功能
### 1. 拥有可交互的界面，可以基于图片的颜色数量自由选择
![本地图片](interaction1.png)
### 2. 可以为图片创建蒙版
蒙版可以在生成的文件夹中看到

<div style="display: flex; justify-content: space-between;">
    <img src="result/mask/mask0.png" alt="Image 1" style="width: 48%;">
    <img src="result/mask/mask1.png" alt="Image 2" style="width: 48%;">
</div>
<div style="display: flex; justify-content: space-between;">
    <img src="result/mask/mask2.png" alt="Image 1" style="width: 48%;">
    <img src="result/mask/mask3.png" alt="Image 2" style="width: 48%;">
</div>

### 3. 颜色平滑
![本地图片](interaction2.png)

# 效果
## 1. 颜色分层

<div style="display: flex; justify-content: space-between;">
    <img src="result/logo/pic.png" alt="Image 1" style="width: 31%;">
    <img src="result/logo/mask1.png" alt="Image 1" style="width: 31%;">
    <img src="result/logo/mask2.png" alt="Image 2" style="width: 31%;">
</div>
<div style="display: flex; justify-content: space-between;">
    <img src="result/logo/mask3.png" alt="Image 1" style="width: 33%;">
    <img src="result/logo/mask4.png" alt="Image 2" style="width: 33%;">
</div>

<div style="display: flex; justify-content: space-between;">
    <img src="result/cat/pic2.jpg" alt="Image 1" style="width: 100%;">
</div>
<div>
    <img src="result/cat/mask0.png" alt="Image 2" style="width: 50%;">
    <img src="result/cat/mask1.png" alt="Image 2" style="width: 49%;">
<div>

<div style="display: flex; justify-content: space-between;">
    <img src="result/flower/pic3.png" alt="Image 1" style="width: 48%;" >
    <img src="result/flower/mask1.png" alt="Image 1" style="width: 48%;">
</div>
<div>
    <img src="result/flower/mask2.png" alt="Image 2" style="width: 50%;">
    <img src="result/flower/mask3.png" alt="Image 2" style="width: 49%;">
<div>


## 2. 颜色平滑
<div style="display: flex; justify-content: space-between;">
    <img src="result/soft_process/landscape.png" alt="Image 1" style="width: 48%;">
    <img src="result/soft_process/soft_process_single69.png" alt="Image 2" style="width: 48%;">
</div>
<div style="display: flex; justify-content: space-between;">
    <img src="result/soft_process/output.jpg" alt="Image 1" style="width: 48%;">
    <img src="result/soft_process/soft_process_single45.png" alt="Image 2" style="width: 48%;">
</div>

# 使用教程
## python3.10
## 第一步创建虚拟环境并安装相应的库
```
python -m venv venv
# windows
.\venv\Scripts\activate

# linux
source venv/bin/activate

pip install -r requirements.txt
```

## 执行程序
```
python color_seg.py
```

# 注意事项
1. 默认精度是0.001
2. 默认单轮最大迭代次数是20次
3. 界面交互可以调节的迭代次数外循环（即每一次迭代次数都会执行单轮最大迭代次数是20次）