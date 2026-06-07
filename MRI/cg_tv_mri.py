import numpy as np
import cv2
from skimage.data import shepp_logan_phantom
from scipy.sparse.linalg import LinearOperator, cg
import matplotlib.pyplot as plt


def fft_forward(image):
    """图像到k空间的FFT变换"""
    return np.fft.fft2(image)

def fft_backward(k_space):
    """k空间到图像的逆FFT变换"""
    return np.fft.ifft2(k_space)

def apply_A(I, mask):
    """A·I：FFT后应用采样掩膜"""
    return np.fft.fftshift(fft_forward(I)) * mask

def apply_AH(b, mask):
    """A^H·b：补零后逆FFT"""
    return fft_backward(np.fft.ifftshift(b * mask))

def tv_gradient(I, lambda_tv, epsilon=1e-6):
    # 前向差分计算梯度
    grad_x = np.roll(I, -1, axis=0) - I
    grad_y = np.roll(I, -1, axis=1) - I
    norm_grad = np.sqrt(grad_x**2 + grad_y**2 + epsilon)
    # 后向差分计算散度
    div_x = (grad_x / norm_grad) - np.roll(grad_x / norm_grad, 1, axis=0)
    div_y = (grad_y / norm_grad) - np.roll(grad_y / norm_grad, 1, axis=1)
    return -lambda_tv * (div_x + div_y)

def cg_tv_with_fft(b, mask, lambda_tv, max_iter=100, epsilon=1e-6):
    """
    基于FFT加速的CG-TV MRI重建
    :param b: 欠采样的k空间数据（复数，已按掩膜采样）
    :param mask: k空间采样掩膜（与b同尺寸）
    :param lambda_tv: TV正则化强度
    :param max_iter: 最大迭代次数
    :param epsilon: TV光滑化参数
    :return: 重建图像（复数）
    """
    # 初始化解（零填充重建）
    I_init = apply_AH(b, mask)
    I = np.real(I_init)  # 仅优化实部（假设相位平滑）

    def grad(I_flat):
        """目标函数梯度计算"""
        I = I_flat.reshape(I_init.shape)
        # 数据保真项梯度（FFT加速）
        residual = apply_A(I, mask) - b
        grad_data = 2 * np.real(apply_AH(residual, mask))
        # TV正则化梯度
        grad_tv = tv_gradient(I, lambda_tv, epsilon)
        return (grad_data + grad_tv).flatten()

    # 定义Hessian近似算子（频域对角矩阵）
    def hessian_matvec(x):
        """H·x ≈ 2A^H A x + TV项的二阶近似"""
        x = x.reshape(I.shape)
        # 数据保真项Hessian（2A^H A x）
        Ax = apply_A(x, mask)
        AH_Ax = 2 * np.real(apply_AH(Ax, mask))
        # TV项Hessian（近似为梯度算子的对角占优项）
        tv_hessian = 1e-3 * x  # 简化假设（实际需更精确近似）
        return (AH_Ax + tv_hessian).flatten()

    # 牛顿-共轭梯度法求解
    H = LinearOperator((I.size, I.size), matvec=hessian_matvec, dtype=np.float64)
    I_flat, info = cg(H, -grad(I.flatten()), maxiter=max_iter, tol=1e-6)
    I_recon = I_flat.reshape(I.shape)
    return I_recon + 1j * np.imag(I_init)  # 恢复相位信息

mask = np.ones((400, 400), dtype=np.complex128)
mask[100:156, 100:156] = 1  # 中心区域采样
b = apply_A(shepp_logan_phantom(), mask)  # 生成仿真k空间数据
lambda_tv = 0.1
reconstructed_image = cg_tv_with_fft(b, mask, lambda_tv, max_iter=50)


I_CG_TV=np.abs(reconstructed_image)
phantom = shepp_logan_phantom()
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1), plt.imshow(phantom, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(I_CG_TV, cmap='gray'), plt.title('NCG TV FFT')
plt.tight_layout()
plt.show()
