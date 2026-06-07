clc;
close all;

%% ============================================================
% Geometry
%% ============================================================
global SOD SDD ODD dNum dSize pNum pSize views

SOD   = 1000;
SDD   = 1500;
ODD   = SDD - SOD;

dNum  = 256;
dSize = 1.0;

pNum  = 256;
pSize = 1.0;

views = 180;

%% ============================================================
% Phantom
%% ============================================================
dh = dNum * dSize / 2;
R_fov = SOD * dh / sqrt(dh^2 + SDD^2);

h = pNum;
img_center = (h + 1) / 2;
u = ((1:h) - img_center) * pSize;
v = (img_center - (1:h)) * pSize;
[imgX, imgY] = meshgrid(u, v);

P = phantom('Modified Shepp-Logan', h);
P = max(P, 0);
P = P ./ max(P(:));

margin = 1.0;
mask = (imgX.^2 + imgY.^2) <= (R_fov - margin)^2;
P = P .* mask;
P = max(P, 0);
if max(P(:)) > 0
    P = P ./ max(P(:));
end

figure;
imagesc(u, v, P);
axis image off;
colormap gray;
colorbar;
title('Ground Truth (FOV-cropped)');

%% ============================================================
% Projection
%% ============================================================
fprintf('Generating projection...\n');
p = forward_project_cone(P);

p = real(p);
p(isnan(p)) = eps;
p(p<=0) = eps;

% ========== 有限角修改（只保留 0°~120° 的投影）==========
angle_vec = (0:views-1) * 360/views;          % 原始角度向量
limit_angle = 120;                            % 有限角范围（度）
idx_keep = angle_vec <= limit_angle;          % 逻辑索引
p = p(:, idx_keep);                           % 截取正弦图
views = sum(idx_keep);                        % 更新全局变量 views !!!
% ===========================================================

figure;
imagesc((0:views-1)*360/views,1:dNum,p);
axis tight;
colormap gray;
colorbar;
xlabel('View (deg)');
ylabel('Detector');
title('Sinogram');

%% ============================================================
% Initialization
%% ============================================================
img = ones(pNum,pNum)*mean(P(:));

%% ============================================================
% OS‑EM Reconstruction (with EBPN framework)
%% ============================================================
fprintf('\nStarting OS‑EM Reconstruction...\n');

max_EBPN = 200;               % 最大等效反投影次数
tol_img  = 1e-5;              % 图像变化停机阈值
enable_early_stop = false;    % 画完整曲线时设为 false

metrics = struct('EBPN', [], 'MSE', [], 'PSNR', [], 'SSIM', []);
img_prev = img;
ebpn = 0;

best_psnr = -inf;
best_ssim = -inf;
best_iter = 1;
best_img  = img;

tic;

while ebpn < max_EBPN

    % ------ 一次 OS‑EM 迭代（1轮子集 = 1次等效反投影）------
    img_new = OSEM(img, p, 1);   % nIter=1，遍历所有子集一次

    % ------ 稳定性处理 ------
    img_new = real(img_new);
    img_new(isnan(img_new)) = 0;
    img_new(isinf(img_new)) = 0;
    img_new = max(img_new, 0);

    % ------ 累计等效反投影次数 ------
    ebpn = ebpn + 1;
    metrics(ebpn).time = toc;
    
    % ------ 记录指标 ------
    mse_now  = mean((img_new(:) - P(:)).^2);
    psnr_now = psnr(img_new, P);
    ssim_now = ssim(img_new, P);

    metrics(ebpn).EBPN = ebpn;
    metrics(ebpn).MSE  = mse_now;
    metrics(ebpn).PSNR = psnr_now;
    metrics(ebpn).SSIM = ssim_now;

    % ------ 更新最佳图像 ------
    if psnr_now > best_psnr
        best_psnr = psnr_now;
        best_ssim = ssim_now;
        best_img  = img_new;
        best_iter = ebpn;
    end

    fprintf('EBPN %3d | PSNR = %.3f dB | SSIM = %.5f\n', ...
        ebpn, psnr_now, ssim_now);

    % ------ 图像变化检查 ------
    rel_change = norm(img_new(:) - img_prev(:)) / norm(img_new(:));
    if enable_early_stop && ebpn > 1 && rel_change < tol_img
        fprintf('Converged due to image change < %.1e at EBPN %d.\n', tol_img, ebpn);
        break;
    end

    % ------ 更新 ------
    img = img_new;
    img_prev = img;

    % ------ 保存特定 EBPN 的图像 ------
    if ismember(ebpn, [1,5,10,20,50,100,200])
        save(sprintf('OSEM_EBPN_%d.mat', ebpn), 'img_new');
    end

end

runtime = toc;

%% ============================================================
% Results
%% ============================================================
fprintf('\n====================================\n');
fprintf('Best EBPN      : %d\n', best_iter);
fprintf('Best PSNR      : %.3f dB\n', best_psnr);
fprintf('Best SSIM      : %.5f\n', best_ssim);
fprintf('Runtime        : %.3f sec\n', runtime);
fprintf('====================================\n');

%% ============================================================
% Final Reconstruction (best image)
%% ============================================================
figure('Position',[100 100 1200 400]);

subplot(1,3,1)
imagesc(P);
axis image off;
colormap gray;
colorbar;
title('Ground Truth');

subplot(1,3,2)
imagesc(best_img);
axis image off;
colormap gray;
colorbar;
title(sprintf('Best OS‑EM (EBPN=%d)', best_iter));

subplot(1,3,3)
imagesc(abs(best_img - P));
axis image off;
colormap hot;
colorbar;
title('Error Map');

%% ============================================================
% Convergence Curves
%% ============================================================
ebpn_arr = [metrics.EBPN];
psnr_arr = [metrics.PSNR];
ssim_arr = [metrics.SSIM];

figure('Position',[200 200 1200 400]);

subplot(1,2,1)
plot(ebpn_arr, psnr_arr, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Equivalent Backprojection Number');
ylabel('PSNR (dB)');
title('PSNR Convergence');
grid on;

subplot(1,2,2)
plot(ebpn_arr, ssim_arr, 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Equivalent Backprojection Number');
ylabel('SSIM');
title('SSIM Convergence');
grid on;

%% ============================================================
% Save Data
%% ============================================================
Result.views      = views;
Result.best_iter  = best_iter;
Result.best_psnr  = best_psnr;
Result.best_ssim  = best_ssim;
Result.runtime    = runtime;
Result.best_img   = best_img;
Result.metrics    = metrics;
Result.P          = P;
Result.p          = p;

save('OSEM_Result_LimitedAngle.mat', 'Result');

fprintf('\nFinished.\n');