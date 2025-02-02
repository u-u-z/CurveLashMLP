import numpy as np
import matplotlib.pyplot as plt

# 使用已有的曲线数据
A_segments = parse_svg_path(A_d)
B_segments = parse_svg_path(B_d)

A_points = bezier_sample(A_segments)
B_points = bezier_sample(B_segments)

# 创建一个大图，包含10个子图
plt.figure(figsize=(20, 15))

A_curves = []
B_curves = []

for i in range(100):
    # 添加随机噪声
    noise_scale = 0.1
    A_noise = np.random.normal(0, noise_scale, A_points.shape)
    B_noise = np.random.normal(0, noise_scale, B_points.shape)
    
    A_curve = A_points + A_noise
    B_curve = B_points + B_noise
    
    A_curves.append(A_curve)
    B_curves.append(B_curve)
    
    # 每10次迭代画一次图
    if (i + 1) % 10 == 0:
        plt.subplot(2, 5, (i + 1) // 10)
        
        # 画原始曲线（半透明）
        plt.plot(A_points[:, 0], -A_points[:, 1], 'b-', alpha=0.3, label='Original A')
        plt.plot(B_points[:, 0], -B_points[:, 1], 'r-', alpha=0.3, label='Original B')
        
        # 画带噪声的曲线
        plt.plot(A_curve[:, 0], -A_curve[:, 1], 'b--', label=f'Noisy A {i+1}')
        plt.plot(B_curve[:, 0], -B_curve[:, 1], 'r--', label=f'Noisy B {i+1}')
        
        plt.title(f'Iteration {i+1}')
        plt.grid(True)
        if i == 0:  # 只在第一个子图显示图例
            plt.legend()
        plt.axis('equal')
        plt.xlim(-10, 190)
        plt.ylim(-70, 10)

plt.tight_layout()
plt.show()

# 打印一些统计信息
print("噪声统计信息：")
A_noise_avg = np.mean([np.mean(np.sqrt(np.sum((A - A_points)**2, axis=1))) for A in A_curves])
B_noise_avg = np.mean([np.mean(np.sqrt(np.sum((B - B_points)**2, axis=1))) for B in B_curves])
print(f"A曲线平均噪声幅度: {A_noise_avg:.4f} 像素")
print(f"B曲线平均噪声幅度: {B_noise_avg:.4f} 像素")
