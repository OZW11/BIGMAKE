import cv2
import numpy as np
import random

def add_blocky_speckle_noise(image, block_size=10, noise_intensity=0.03):
    # """
    # 向图像添加块状的散斑噪声。

    # :param image: 输入图像（彩色或灰度）
    # :param block_size: 噪声块的大小（以像素为单位）
    # :param noise_intensity: 噪声的强度（0到1之间的浮点数）
    # :return: 添加噪声后的图像
    # """
    # 确保图像是浮点型，以便进行数学运算
    image = image.astype(np.float32) / 255.0
    
    # 获取图像的尺寸
    h, w, c = image.shape if len(image.shape) == 3 else (image.shape[0], image.shape[1], 1)
    
    # 生成噪声图像
    noise_image = np.zeros_like(image)
    
    # 遍历图像的每个块
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            # 确定块的边界，确保不会超出图像范围
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            
            # 为当前块生成一个随机的噪声值
            noise_value = random.uniform(-noise_intensity, noise_intensity)
            
            # 将噪声值应用到当前块
            noise_image[y:y_end, x:x_end, :] += noise_value
    
    # 将噪声图像与原始图像结合，并限制在0到1的范围内
    noisy_image = np.clip(image + noise_image, 0, 1)

    # 将图像转换回uint8类型并返回
    return (noisy_image * 255).astype(np.uint8)

if __name__ == '__main__':

    # 读取图像
    image_path = './test_img/17_Averaged Image.tif'  # 替换为你的图像路径
    image = cv2.imread(image_path)
    
    # 如果图像是彩色的，将其从BGR转换为RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 添加块状散斑噪声
    noisy_image = add_blocky_speckle_noise(image, block_size=5, noise_intensity=0.05)
    
    # 显示原始图像和添加噪声后的图像
    import matplotlib.pyplot as plt
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image)
    plt.title('Image with Blocky Speckle Noise')
    plt.axis('off')
    
    plt.show()