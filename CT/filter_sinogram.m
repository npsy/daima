function P_filt = filter_sinogram(P_para, dSize)
% 对平行束投影进行斜坡滤波（Ram-Lak, 全带宽）
%
% 输入：
%   P_para - 平行束投影矩阵 [N_s, N_theta] (行: s, 列: θ)
%   d_s    - 探测器采样间距（s 方向的步长）
% 输出：
%   P_filt - 滤波后的投影矩阵，尺寸相同

[N_s, N_theta] = size(P_para);

% 1. FFT 长度：取 ≥ 2*N_s-1 的下一个 2 的幂，避免循环卷积混叠
N_fft = 2^nextpow2(2 * N_s - 1);

% 2. 频率轴（单位：1/长度）
%    采样间隔 dSize，采样频率 1/dSize，奈奎斯特频率为 1/(2*dSize)
f_max = 1 / (2 * dSize);
f = linspace(-f_max, f_max, N_fft)';   % 列向量，方便后面点乘

% 3. 构造 Ram-Lak 滤波器 H(f) = |f|，全带宽（cutoff=1）
H = abs(f);

% 4. 逐角滤波
P_filt = zeros(N_s, N_theta);
for i = 1:N_theta
    proj = P_para(:, i);
    % 补零
    proj_pad = [proj; zeros(N_fft - N_s, 1)];
    % FFT → 乘 H → IFFT
    proj_f = fftshift(fft(proj_pad));
    proj_f = proj_f .* H;                % 频域相乘
    proj_ifft = ifft(ifftshift(proj_f));
    proj_ifft = real(proj_ifft);          % 舍入误差导致的极小虚部可忽略
    % 截取回原长度
    P_filt(:, i) = proj_ifft(1:N_s);
end
end