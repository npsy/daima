function img = fbp_backproject(P_filt, theta_para, s_para)
% =====================================================
% 平行束滤波反投影 (Backprojection)
%
% 输入：
%   P_filt      - 滤波后的投影矩阵 [N_s, N_theta]
%                 行: 探测器位置 s, 列: 角度 theta
%   theta_para  - 投影角度向量 (弧度), 长度 N_theta
%   s_para      - 探测器位置向量 (长度 N_s)
%
% 输出：
%   img         - 重建图像 [pNum, pNum]
%
% 依赖全局变量：pNum, pSize
% =====================================================

global pNum pSize

% 图像参数
h = pNum;
img_center = (h + 1) / 2;

% 角度步长
dtheta = theta_para(2) - theta_para(1);

% 探测器参数
N_s = length(s_para);
s_min = s_para(1);
d_s   = s_para(2) - s_para(1);

% 初始化图像
img = zeros(h, h);

% 图像物理坐标（与 forward_project_cone 一致）
% x 向右，y 向上；但图像矩阵行索引向下，所以 y 方向取反
x_vals = ((1:h) - img_center) * pSize;            % 行向量
y_vals = (img_center - (1:h)) * pSize;            % 行向量

% 对每个像素进行反投影
for iy = 1:h          % 行，对应 y
    y = y_vals(iy);
    for ix = 1:h      % 列，对应 x
        x = x_vals(ix);
        sum_val = 0;
        for i_theta = 1:length(theta_para)
            th = theta_para(i_theta);
            % 平行束投影坐标
            s = x * cos(th) + y * sin(th);
            % 映射到探测器索引 (浮点)
            s_idx = (s - s_min) / d_s + 1;
            % 线性插值
            if s_idx >= 1 && s_idx <= N_s
                sum_val = sum_val + interp1(1:N_s, P_filt(:, i_theta), ...
                                            s_idx, 'linear', 0);
            end
        end
        img(iy, ix) = sum_val * dtheta;   % 积分乘角度步长
    end
end
end