import numpy as np
from skimage.data import shepp_logan_phantom
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt

# 生成模体（256x256）
phantom = shepp_logan_phantom()
plt.imshow(phantom, cmap='gray')
plt.title("Shepp-Logan Phantom")
plt.show()

#共轭梯度法程序
#k=0
#给定初始解x0
#定义求残差函数r(x)
#现在就可以求得r0,p0=-r0,
#定义步长函数a(x)=-(rktrk)/(pktr(pk))
#定义向量函数b(x)=(rk+1trk+1)/(rktrk)
#开始进行迭代
#xk+1=xk+akpk
#pk+1=-rk+1+bk+1pk
#k=k+1
import numpy as np

def conjugate_gradient(b, x0=None, max_iter=1000, tol=1e-6):
    n = len(b)
    x = np.zeros(n, dtype=complex) if x0 is None else x0.copy()
    r = np.fft.fft2(x) - b
    p = -r  # 初始搜索方向 p₀ = -r₀
    k = 0

    while np.linalg.norm(r) > tol and k < max_iter:
        Ap = np.fft.fft2(p)  # 计算 A*pₖ
        denominator = np.vdot(p, Ap) # 分母 pₖᵀ A pₖ

        alpha = -np.vdot(r, p) / denominator  # αₖ = -(rₖᵀ pₖ) / (pₖᵀ A pₖ)
        x = x + alpha * p  # 更新解 xₖ₊₁ = xₖ + αₖ pₖ
        r_new = np.fft.fft2(x) - b  # 计算新残差 rₖ₊₁ = A xₖ₊₁ - b

        # 计算 βₖ₊₁ = (rₖ₊₁ᵀ A pₖ) / (pₖᵀ A pₖ)
        numerator = np.vdot(r_new, Ap)
        beta = numerator / denominator

        p = -r_new + beta * p  # 更新搜索方向 pₖ₊₁ = -rₖ₊₁ + βₖ₊₁ pₖ
        r = r_new  # 更新残差
        k += 1

        return x

# 可视化
k_space_full = np.fft.fft2(phantom)
x0=np.zeros_like(phantom)
I_CG=conjugate_gradient(k_space_full,x0)
I_CG=np.real(I_CG)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1), plt.imshow(phantom, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(I_CG, cmap='gray'), plt.title('CG FFT')
plt.tight_layout()
plt.show()