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
% 计算有效 FOV 半径（与 EM 几何一致）
dh = dNum * dSize / 2;
R_fov = SOD * dh / sqrt(dh^2 + SDD^2);   % 单位 mm

% 图像物理坐标（与重建代码一致）
h = pNum;
img_center = (h + 1) / 2;
u = ((1:h) - img_center) * pSize;
v = (img_center - (1:h)) * pSize;
[imgX, imgY] = meshgrid(u, v);

% 生成原始 Shepp-Logan（填满整个图像区域）
P = phantom('Modified Shepp-Logan', h);
P = max(P, 0);
P = P ./ max(P(:));

% 圆形 FOV mask（半径略小于 R_fov 以避免边界插值问题）
margin = 1.0;   % 留 1 mm 安全余量
mask = (imgX.^2 + imgY.^2) <= (R_fov - margin)^2;

% 应用 mask，保证体模不超出 FOV
P = P .* mask;

% 保持非负归一化
P = max(P, 0);
if max(P(:)) > 0
    P = P ./ max(P(:));
end

% 显示检查
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
nIter = 30;

img = ones(pNum,pNum)*mean(P(:));

mse_history  = zeros(nIter,1);
psnr_history = zeros(nIter,1);
ssim_history = zeros(nIter,1);

best_psnr = -inf;
best_ssim = -inf;
best_iter = 1;
best_img  = img;

%% ============================================================
% EM Reconstruction
%% ============================================================
fprintf('\nStarting EM Reconstruction...\n');

tic;

for iter = 1:nIter

    img = EM(img,p,1);

    %% ---------- Stability ----------
    img = real(img);

    img(isnan(img)) = 0;
    img(isinf(img)) = 0;

    img = max(img,0);

    %% ---------- Metrics ----------
    mse = mean((img(:)-P(:)).^2);

    mse_history(iter) = mse;

    psnr_history(iter) = psnr(img,P);

    ssim_history(iter) = ssim(img,P);

    %% ---------- Save Best ----------
    if psnr_history(iter) > best_psnr

        best_psnr = psnr_history(iter);
        best_ssim = ssim_history(iter);

        best_img  = img;
        best_iter = iter;

    end

    fprintf('Iter %2d | PSNR = %.3f dB | SSIM = %.5f\n',...
        iter,...
        psnr_history(iter),...
        ssim_history(iter));

end

runtime = toc;

%% ============================================================
% Results
%% ============================================================
fprintf('\n====================================\n');
fprintf('Best Iteration : %d\n',best_iter);
fprintf('Best PSNR      : %.3f dB\n',best_psnr);
fprintf('Best SSIM      : %.5f\n',best_ssim);
fprintf('Runtime        : %.3f sec\n',runtime);
fprintf('====================================\n');

%% ============================================================
% Final Reconstruction
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
title(sprintf('Best EM (Iter=%d)',best_iter));

subplot(1,3,3)
imagesc(abs(best_img-P));
axis image off;
colormap hot;
colorbar;
title('Error Map');

%% ============================================================
% Convergence Curves
%% ============================================================
figure('Position',[200 200 1200 400]);

subplot(1,2,1)

plot(1:nIter,...
     psnr_history,...
     'b-o',...
     'LineWidth',2,...
     'MarkerSize',6);

xlabel('Iteration');
ylabel('PSNR (dB)');
title('PSNR Convergence');
grid on;

subplot(1,2,2)

plot(1:nIter,...
     ssim_history,...
     'r-o',...
     'LineWidth',2,...
     'MarkerSize',6);

xlabel('Iteration');
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

save('EM_Result.mat','Result');

fprintf('\nFinished.\n');