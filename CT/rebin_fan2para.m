function [P_para, theta_para, s_para] = rebin_fan2para(proj)
% =====================================================
% 重排函数：等距探测器扇束 → 平行束
% 
% 输入：
%   proj - 扇束投影数据，维度 [dNum, views] (行: 探测器通道γ, 列: 视角β)
% 输出：
%   P_para     - 重排后的平行束投影矩阵 [N_s, N_theta]
%   theta_para - 平行束视角向量 (长度 N_theta, 单位弧度, 0 ~ π-Δθ)
%   s_para     - 平行束探测器位置向量 (长度 N_s, 单位与SOD一致)
%
% 依赖全局变量：
%   SOD, SDD, dNum, dSize, views
% =====================================================

global SOD SDD dNum dSize views

% ---------- 基础几何参数 ----------
dh = dNum * dSize / 2;                        % 探测器半宽
gamma_max = asin(dh / sqrt(SDD^2 + dh^2));    % 最大扇角

% ---------- 扇束采样网格（与forward_project_cone一致）----------
detX = (-dh + dSize/2 : dSize : dh - dSize/2)';  % 探测器坐标列向量
gamma_vals = asin(detX ./ sqrt(SDD^2 + detX.^2));% 射线角γ (dNum×1)
beta_vals  = 2*pi*(0:views-1)/views;              % 视角β (1×views)

% ---------- 定义平行束网格 ----------
% 角度方向：0 到 π-Δθ, 步长 Δθ = Δβ = 2π/views
delta_beta = 2*pi / views;
N_theta = floor(views/2);           % 平行束视角数
if mod(views,2) ~= 0
    N_theta = N_theta + 1;          % views为奇数时向上取整
end
theta_para = linspace(0, pi - pi/N_theta, N_theta);  % 行向量

% 探测器方向：s 均匀分布，范围[-SOD*sin(γ_max), SOD*sin(γ_max)]
N_s = dNum;
s_max = SOD * sin(gamma_max);
s_para = linspace(-s_max, s_max, N_s)';     % 列向量

% ---------- 构造查询网格并计算对应扇束坐标 ----------
[THETA, S] = meshgrid(theta_para, s_para);  % 网格矩阵 (N_s × N_theta)

GAMMA = asin(S / SOD);                       % γ = arcsin(s/SOD)
BETA  = THETA + GAMMA;                       % β = θ + γ
BETA  = mod(BETA, 2*pi);                     % 归化到 [0, 2π)

% ---------- 处理β的周期性：扩展一列 ----------
proj_ext = [proj, proj(:,1)];                % 在β=2π处复制0°投影
beta_ext = [beta_vals, 2*pi];                % 扩展的β网格

% ---------- 二维线性插值 ----------
% interp2(X, Y, V, Xq, Yq): X对应列网格(β), Y对应行网格(γ)
P_para = interp2(beta_ext, gamma_vals, proj_ext, ...
                 BETA, GAMMA, 'linear', 0);
% 结果 P_para 为 N_s × N_theta 矩阵
end